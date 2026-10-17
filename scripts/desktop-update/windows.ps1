# windows.ps1 -- repo-owned Windows Desktop update hand-off.
#
# WHY THIS EXISTS (the frozen-binary problem): the Desktop's Update button
# used to hand off exclusively to the staged Tauri binary
# (%HERMES_HOME%\hermes-setup.exe). That binary has no self-update path --
# copy_self_to_hermes_home deliberately no-ops during --update -- so every
# updater-side fix (cache refresh #67369, marker self-adopt #74782, straggler
# handling) only reaches users when a new installer is built, signed, and
# published. In practice binaries go months stale and users hit long-fixed
# bugs on every update (the 2026-08-09 incident chain).
#
# This script lives in the repo checkout, so EVERY `hermes update` refreshes
# the very code that drives the next update. The Desktop spawns it through a
# `cmd start` wrapper (see wrapHandoffForDetachedConsole in
# apps/desktop/electron/updater-process.ts -- a bare detached+hidden
# powershell dies before -File runs) and exits; only PowerShell itself -- an
# OS component -- is "frozen".
#
# CONTRACT (keep in sync with apps/desktop/electron/main.ts):
#   cmd /d /s /c start "" /b powershell -NoProfile -ExecutionPolicy Bypass
#     -File scripts\desktop-update\windows.ps1
#     -InstallRoot <path>   repo checkout (HERMES_HOME\hermes-agent)
#     [-Branch <ref> | -Channel stable|canary|main]  default: branch main
#     -DesktopPid <pid>     the Electron main process to wait out
#     [-RelaunchExe <path>] Hermes.exe to start when done (omit = no relaunch)
#     [-NoUi]               headless (tests); default shows a progress window
#     [-NoMarkerCleanup]    leave .hermes-update-in-progress in place (tests)
#     [-ProbeTimeoutSeconds <n>] launcher probe bound, default 60 (tests)
#     [-HandoffRun <id>]    protocol 2: the run id of the Desktop's bridge claim
#
# hermes-handoff-protocol: 2
# (exact line above: the Desktop reads it to learn this script speaks hand-off
# protocol 2 -- it accepts -HandoffRun and the -MarkerOp helper below.)
#
# HELPER OPS (Electron's only way to mutate the marker; runs right after
# marker.ps1 loads -- no UI, log rotation, result or relaunch):
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File windows.ps1
#     -MarkerOp reclaim|withdraw -InstallRoot <root> [-DesktopPid P] [-HandoffRun R]
#   prints ONE line (absent | reclaimed | live <pid> | busy | withdrawn |
#   taken <pid> | foreign) and exits 0; 64 on bad usage.
#   (-MarkerOp custody -CustodyOf <pid> is internal: Start-MarkerCustodian.)
#
# SAFETY POSTURE: both preflight gates FAIL CLOSED. A Desktop that never
# exits, or a venv shim that never unlocks, aborts the hand-off without
# mutating the install -- a skipped update is recoverable, a half-updated
# venv is not. Every exit path (success, abort, crash) writes
# .hermes-update-result.json for the relaunched Desktop to surface, and
# relaunches the Desktop so the user is never left stranded. The one
# exception is a refused run (exit 2): it changed nothing and writes no result.
#
# Marker (contract C1 v2 + A7, marker.ps1): claiming HERMES_HOME\.hermes-update-
# in-progress is the FIRST thing the script does -- before any Add-Type, UI
# or probe -- and every read/judge/mutate of it happens inside one hold of
# the kernel lock on "<marker>.lock". With -HandoffRun it only adopts the
# Desktop's bridge for that run; with only -DesktopPid (an old packaged
# Desktop) it accepts that Desktop's cmd.exe launcher claim by lineage;
# otherwise it claims fresh, reclaiming a dead marker. Any other live owner
# refuses the run (exit 2, nothing changed, no result file). The `hermes
# update` child is named as the line-4 delegate BEFORE it is resumed.
# Release deletes our claim, or hands it to a still-running delegate.

param(
    [string]$InstallRoot,
    [string]$Branch = "main",
    [ValidateSet("stable", "canary", "main")]
    [string]$Channel,
    [int]$DesktopPid = 0,
    [string]$RelaunchExe = "",
    [switch]$NoUi,
    [switch]$NoMarkerCleanup,
    [switch]$NoGateway,
    [int]$ProbeTimeoutSeconds = 60,
    [switch]$SelfTestUi,
    [switch]$SelfTestPipeDrain,
    [switch]$SelfTestRelaunchWindow,
    [switch]$SelfTestMarker,
    [switch]$SelfTestWorkingDirectory,
    [string]$HandoffRun = "",
    [string]$MarkerOp = "",
    [int]$CustodyOf = 0
)

if ($MarkerOp -and -not $InstallRoot) {
    [Console]::Error.WriteLine("-MarkerOp needs -InstallRoot")
    exit 64
}

if ($PSBoundParameters.ContainsKey("Branch") -and $PSBoundParameters.ContainsKey("Channel")) {
    throw "-Branch and -Channel are mutually exclusive"
}
$targetArgs = if ($Channel) { @("--channel", $Channel.ToLowerInvariant()) } else { @("--branch", $Branch) }

if (-not $SelfTestUi -and -not $SelfTestPipeDrain -and -not $SelfTestRelaunchWindow -and -not $InstallRoot) {
    # Mandatory in spirit; relaxed in the signature only so the self-test
    # switches can drive the UI / the pipe drain without a checkout.
    throw "-InstallRoot is required"
}

$ErrorActionPreference = "Continue"
$TempDir = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
$HermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } elseif ($InstallRoot) { Split-Path -Parent $InstallRoot } else { $TempDir }
$env:HERMES_HOME = $HermesHome
$MarkerPath = Join-Path $HermesHome ".hermes-update-in-progress"
$LogDir = Join-Path $HermesHome "logs"
$LogPath = Join-Path $LogDir "desktop-update-handoff.log"
$ResultPath = Join-Path $HermesHome ".hermes-update-result.json"

function Write-HandoffLog([string]$Message) {
    $line = "{0:yyyy-MM-ddTHH:mm:ssK} {1}" -f (Get-Date), $Message
    try { Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 } catch {}
    if ($MarkerOp) { return }   # a helper op's stdout is its one verdict line
    if ($script:ConsoleInput -and [HermesHandoff.ConsoleInput]::Selecting()) { return }
    Write-Host $line
}

# Update marker (contract C1 v2 + amendments A1-A4) lives in marker.ps1 next
# to this script. It is pure PowerShell + CIM: the claim runs before the
# first Add-Type. A hand-off that cannot load it changes nothing.
New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
try { . (Join-Path $PSScriptRoot 'marker.ps1') } catch {
    Write-HandoffLog "Update aborted: $PSScriptRoot\marker.ps1 could not be loaded ($($_.Exception.Message)). Nothing was changed. Repair the installation and try again."
    exit 3
}

# Helper op (SPEC 6): one verdict line, before any UI, result or relaunch.
if ($MarkerOp -ceq 'custody') { Invoke-MarkerCustody $CustodyOf; exit 0 }   # Start-MarkerCustodian's watcher
if ($MarkerOp) { exit (Invoke-MarkerOpCli $MarkerOp $DesktopPid $HandoffRun) }

# The Desktop's identity is pinned now: a reused pid later is not "still open".
$script:DesktopCt = $null
$script:DesktopSeenAlive = $false
if ($DesktopPid -gt 0) {
    $desktopProbe = Get-LiveProcessCt $DesktopPid
    $script:DesktopSeenAlive = $desktopProbe.Alive
    $script:DesktopCt = $desktopProbe.Ct
}
if (-not $SelfTestUi -and -not $SelfTestPipeDrain) {
    $script:MarkerClaim = Invoke-MarkerClaim
}

# Foreground helpers: the script is spawned via `cmd start /b` and inherits
# the wrapper's hidden console, so its WinForms window comes up backgrounded
# unless we explicitly claim focus --
# and after the update we must hand focus TO the relaunched Desktop (a
# WMI-spawned process starts unfocused). AllowSetForegroundWindow lets us
# pass our foreground right on to the new Hermes.exe pid.
try {
    Add-Type -Namespace HermesHandoff -Name Win32 -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(System.IntPtr hWnd);
[DllImport("user32.dll")] public static extern bool AllowSetForegroundWindow(int dwProcessId);
[DllImport("user32.dll")] public static extern bool ShowWindow(System.IntPtr hWnd, int nCmdShow);
'@ -ErrorAction Stop
    $script:Win32 = $true
} catch { $script:Win32 = $false }
# Console selection must never hold the hand-off (#103222). The console is
# hidden by design (wrapHandoffForDetachedConsole), but an older Desktop or a
# manual run can leave it visible, and conhost blocks every write to it while a
# selection is active: the child output replayed after `hermes update` exited
# stalled on Write-Host until the user pressed Esc, and the result, marker
# cleanup and relaunch waited behind it. QuickEdit goes off for the run (so a
# stray click cannot start a selection) and the console echo is skipped while
# one is active; the log file keeps every line either way.
try {
    Add-Type -Namespace HermesHandoff -Name ConsoleInput -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] public struct SelectionInfo { public uint Flags; public uint Anchor; public ulong Window; }
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
static extern IntPtr CreateFile(string name, uint access, uint share, IntPtr attributes, uint disposition, uint flags, IntPtr template);
[DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
[DllImport("kernel32.dll")] static extern bool GetConsoleMode(IntPtr handle, out uint mode);
[DllImport("kernel32.dll")] static extern bool SetConsoleMode(IntPtr handle, uint mode);
[DllImport("kernel32.dll")] static extern bool GetConsoleSelectionInfo(out SelectionInfo info);

// stdin is NUL under the Desktop spawn, so open the console input buffer itself.
static uint? Swap(Func<uint, uint> change) {
    IntPtr input = CreateFile("CONIN$", 0xC0000000, 3, IntPtr.Zero, 3, 0, IntPtr.Zero);
    if (input == new IntPtr(-1)) return null;
    try {
        uint mode;
        if (!GetConsoleMode(input, out mode) || !SetConsoleMode(input, change(mode))) return null;
        return mode;
    } finally { CloseHandle(input); }
}
// ENABLE_EXTENDED_FLAGS (0x80) makes conhost honour the cleared ENABLE_QUICK_EDIT_MODE (0x40).
public static uint? DisableQuickEdit() { return Swap(mode => (mode & ~0x40u) | 0x80u); }
public static void Restore(uint mode) { Swap(_ => mode); }
public static bool Selecting() {
    SelectionInfo info;
    return GetConsoleSelectionInfo(out info) && (info.Flags & 1u) != 0; // CONSOLE_SELECTION_IN_PROGRESS
}
'@ -ErrorAction Stop
    $script:ConsoleInput = $true
} catch { $script:ConsoleInput = $false }
# Render UTF-8 glyphs (checkmarks, arrows) correctly in our own console echo
# too; the legacy conhost default OEM codepage shows them as mojibake.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}
$script:Ui = $null
$script:UiStage = "Hermes will open once done."   # until the first gate; matches ui.html
$script:UiStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# ── The shim: repo-owned HTML in a chromeless default-browser app window ───
# The window is a veneer, not a participant: the update runs identically with
# or without it (default browser missing/failed degrades to the WinForms card below,
# then log-only). It never consumes child output; it polls /progress for the
# current hand-off stage or a terminal event and reacts. The loopback listener
# is not a web server in any meaningful sense; it exists because file:// pages
# cannot receive events from a detached process. Salvaged from the web-shell
# spike (Co-authored-by: teknium1), reshaped to the quiet update-surface
# contract (#75895/#83634): loader, one title, one line, no dashboard.
$script:UiState = [hashtable]::Synchronized(@{
    status     = "running"      # running | done | manual | error
    message    = $script:UiStage
    clock      = $script:UiStopwatch
    receipt    = $null
    acknowledged_receipt = $null
})
$script:UiServer = $null     # @{ Listener; Runspace; PowerShell; Port; BrowserProc; Profile }

function Get-UiHtmlPath {
    # Lives next to this script in the checkout. Missing file = fall back to
    # WinForms (old checkouts mid-update, partial syncs).
    $p = Join-Path $PSScriptRoot "ui.html"
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Get-DefaultBrowserExe {
    # The OS default browser, read from the ProgId that the Windows Settings
    # app writes (https first, http as fallback). Windows 11 25H2 writes only
    # UserChoiceLatest\ProgId and leaves the legacy UserChoice key stale or
    # without a value, so the newer key is read first. Only
    # Chromium-family browsers (ChromeHTML / MSEdgeHTM) support the
    # --app + --user-data-dir combo the shim relies on; any other
    # default browser returns $null and degrades to the WinForms card.
    $progId = $null
    foreach ($proto in @("https", "http")) {
        foreach ($sub in @("UserChoiceLatest\ProgId", "UserChoice")) {
            try {
                $progId = (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\$proto\$sub" -Name ProgId -ErrorAction Stop).ProgId
            } catch { continue }
            if ($progId) { break }
        }
        if ($progId) { break }
    }
    if (-not $progId) { return $null }
    $family = switch ($progId) {
        "ChromeHTML" { "Google\Chrome\Application\chrome.exe" }
        "MSEdgeHTM"  { "Microsoft\Edge\Application\msedge.exe" }
        default      { $null }
    }
    if (-not $family) { return $null }
    # Exact path from the ProgId's open command first, then standard roots.
    try {
        $cmd = (Get-ItemProperty -Path "Registry::HKEY_CLASSES_ROOT\$progId\shell\open\command" -ErrorAction Stop).'(default)'
        if ($cmd -and $cmd -match '"([^"]+\.exe)"') {
            $exe = $Matches[1]
            if (Test-Path -LiteralPath $exe) { return $exe }
        }
    } catch {}
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
        if (-not $root) { continue }
        $p = Join-Path $root $family
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Start-UiServer([string]$HtmlPath) {
    # In-process HTTP on a loopback ephemeral port, served from a dedicated
    # runspace so the main thread never blocks on Accept. Plain TcpListener
    # instead of HttpListener: no URL ACL / netsh reservation semantics to
    # trip over, and two GET routes don't need more.
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $listener.Start()
        $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port

        $rs = [runspacefactory]::CreateRunspace()
        $rs.Open()
        $rs.SessionStateProxy.SetVariable("Listener", $listener)
        $rs.SessionStateProxy.SetVariable("State", $script:UiState)
        $rs.SessionStateProxy.SetVariable("HtmlBytes", [System.IO.File]::ReadAllBytes($HtmlPath))

        $ps = [powershell]::Create()
        $ps.Runspace = $rs
        [void]$ps.AddScript({
            function Send-Response($Stream, [string]$Status, [string]$ContentType, [byte[]]$Body) {
                $head = "HTTP/1.1 $Status`r`nContent-Type: $ContentType`r`nContent-Length: $($Body.Length)`r`nCache-Control: no-store`r`nConnection: close`r`n`r`n"
                $headBytes = [System.Text.Encoding]::ASCII.GetBytes($head)
                $Stream.Write($headBytes, 0, $headBytes.Length)
                $Stream.Write($Body, 0, $Body.Length)
                $Stream.Flush()
            }
            while ($true) {
                try { $client = $Listener.AcceptTcpClient() } catch { break }  # Stop() ends the loop
                try {
                    $client.ReceiveTimeout = 2000
                    $stream = $client.GetStream()
                    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
                    $request = $reader.ReadLine()
                    # Drain headers so the client doesn't see a reset mid-send.
                    while ($true) { $h = $reader.ReadLine(); if ($null -eq $h -or $h -eq "") { break } }
                    if ($request -match "^GET /progress HTTP/1\.[01]$") {
                        $elapsed = [Math]::Floor($State.clock.Elapsed.TotalSeconds)
                        $snapshot = @{
                            status          = $State.status
                            message         = $State.message
                            elapsed_seconds = $elapsed
                            receipt         = $State.receipt
                        } | ConvertTo-Json -Compress
                        Send-Response $stream "200 OK" "application/json; charset=utf-8" ([System.Text.Encoding]::UTF8.GetBytes($snapshot))
                    } elseif ($request -match "^POST /ack/([^ /?]+) HTTP/1\.[01]$") {
                        $receipt = $Matches[1]
                        if ($State.status -in @("done", "manual", "error") -and $State.receipt -and $receipt -ceq $State.receipt) {
                            # Flush acceptance before waking the owner that will
                            # close the listener. No request body is needed.
                            Send-Response $stream "204 No Content" "text/plain" ([byte[]]@())
                            $State.acknowledged_receipt = $receipt
                        } else {
                            Send-Response $stream "409 Conflict" "text/plain" ([System.Text.Encoding]::ASCII.GetBytes("unknown terminal receipt"))
                        }
                    } elseif ($request -match "^GET / HTTP/1\.[01]$") {
                        Send-Response $stream "200 OK" "text/html; charset=utf-8" $HtmlBytes
                    } else {
                        Send-Response $stream "404 Not Found" "text/plain" ([System.Text.Encoding]::ASCII.GetBytes("not found"))
                    }
                } catch {
                    # Per-connection failure: drop it, keep serving.
                } finally {
                    try { $client.Close() } catch {}
                }
            }
        })
        [void]$ps.BeginInvoke()

        # Readiness handshake. BeginInvoke returns before the runspace has
        # opened its pipeline and JIT'd the script block — on a loaded machine
        # that is seconds, during which the kernel ACCEPTS connections into
        # the listener's backlog and nobody answers them. Anything that
        # trusted "listener bound" as "server serving" (the browser window
        # opening to a page that never loads; the -SelfTestUi URL that CI
        # polls) raced that gap. Prove one /progress round-trip before
        # handing the port out, so the URL means "serving", not "bound".
        $ready = $false
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(15)
        while (-not $ready -and [DateTime]::UtcNow -lt $readyDeadline) {
            try {
                $probe = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$port/progress")
                $probe.Timeout = 1000
                $probe.ReadWriteTimeout = 1000
                $probe.KeepAlive = $false
                $resp = $probe.GetResponse()
                try { $ready = ([int]$resp.StatusCode -eq 200) } finally { $resp.Close() }
            } catch {
                Start-Sleep -Milliseconds 100
            }
        }
        if (-not $ready) {
            Write-HandoffLog "progress server did not answer /progress within 15s; continuing without UI"
            try { $listener.Stop() } catch {}
            try { $ps.Stop() } catch {}
            try { $rs.Close() } catch {}
            return $null
        }

        return @{ Listener = $listener; Runspace = $rs; PowerShell = $ps; Port = $port; BrowserProc = $null; Profile = $null }
    } catch {
        try { if ($listener) { $listener.Stop() } } catch {}
        return $null
    }
}

function Stop-UiServer([switch]$LeaveWindow) {
    if (-not $script:UiServer) { return }
    try { $script:UiServer.Listener.Stop() } catch {}
    try { $script:UiServer.PowerShell.Stop() } catch {}
    try { $script:UiServer.Runspace.Close() } catch {}
    # On success the window closes itself out from under the user (the whole
    # point); on error we LEAVE it — the page holds the failure state and the
    # user closes it when they've read it.
    if (-not $LeaveWindow) {
        try {
            if ($script:UiServer.BrowserProc -and -not $script:UiServer.BrowserProc.HasExited) {
                $script:UiServer.BrowserProc.CloseMainWindow() | Out-Null
            }
        } catch {}
    }
    # Best-effort removal of the dedicated browser profile dirs: this run's
    # profile plus hermes-update-ui-<pid> leftovers whose hand-off process is
    # gone. %TEMP% is shared by every HERMES_HOME of this user, and the marker
    # only serialises one home, so a dir whose pid is still running belongs to
    # another live hand-off and is left alone. A browser that is still
    # shutting down may hold the lock; the delete then silently no-ops.
    try {
        $profileDirs = @()
        if ($script:UiServer.Profile) { $profileDirs += $script:UiServer.Profile }
        Get-ChildItem -LiteralPath $TempDir -Directory -Filter "hermes-update-ui-*" -ErrorAction SilentlyContinue |
            ForEach-Object {
                $owner = 0
                if ($_.Name -match '^hermes-update-ui-([0-9]+)$' -and [int]::TryParse($Matches[1], [ref]$owner) -and
                    $owner -ne $PID -and -not (Get-Process -Id $owner -ErrorAction SilentlyContinue)) {
                    $profileDirs += $_.FullName
                }
            }
        foreach ($dir in ($profileDirs | Select-Object -Unique)) {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    } catch {}
    $script:UiServer = $null
}

function Publish-UiEvent([string]$Status, [string]$Message) {
    # A background browser can miss a fixed 900ms delivery window. Retain the
    # terminal event until the page acknowledges applying this exact receipt.
    # Older/headless clients cannot acknowledge, so teardown remains bounded.
    $receipt = [Guid]::NewGuid().ToString('N')
    $script:UiState.receipt = $receipt
    $script:UiState.acknowledged_receipt = $null
    $script:UiState.message = $Message
    $script:UiState.status = $Status
    if ($script:UiServer) {
        $deliveryWait = [System.Diagnostics.Stopwatch]::StartNew()
        while ($script:UiState.acknowledged_receipt -cne $receipt -and $deliveryWait.Elapsed.TotalSeconds -lt 10) {
            Start-Sleep -Milliseconds 50
        }
        if ($script:UiState.acknowledged_receipt -ceq $receipt) {
            Write-HandoffLog "shim: terminal state '$Status' acknowledged by the window"
        } else {
            Write-HandoffLog "shim: terminal state '$Status' was not acknowledged within 10s; closing the progress server"
        }
    }
}

function Get-UiElapsedText {
    $elapsed = [Math]::Floor($script:UiStopwatch.Elapsed.TotalSeconds)
    if ($elapsed -lt 60) { return "${elapsed}s elapsed" }
    $minutes = [Math]::Floor($elapsed / 60)
    $seconds = $elapsed % 60
    return "${minutes}m ${seconds}s elapsed"
}

function Get-UiProgressLine {
    return "$script:UiStage`r`n$(Get-UiElapsedText)"
}

function Publish-UiProgress([string]$Message) {
    # Stages come from the orchestrator's own control flow. Child stdout and
    # stderr remain asynchronously drained in Invoke-HermesStep and are never
    # read or parsed for UI updates.
    $script:UiStage = $Message
    $script:UiState.message = $Message
    $script:UiState.status = "running"
    $script:UiState.receipt = $null
    $script:UiState.acknowledged_receipt = $null
    if ($script:Ui) {
        try {
            $script:Ui.Sub.Text = Get-UiProgressLine
            [System.Windows.Forms.Application]::DoEvents()
        } catch {}
    }
}

# ── Fallback card (no Edge / no HTML): same shape in WinForms ──────────────
# Matches the shim pixel-for-pixel in spirit -- loader, one title, one live
# stage/elapsed line, OS light/dark -- so degrading is invisible to the user.
function Get-AppsUseLightTheme {
    try {
        $v = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" -Name AppsUseLightTheme -ErrorAction Stop
        return [int]$v.AppsUseLightTheme -ne 0
    } catch { return $true }
}

function Show-ProgressWindow {
    if ($NoUi) { return }

    # ── Primary: the HTML shim in a chromeless default-browser app window ──
    # Same footprint as the card (280x320), spawned as a normal window: it
    # claims attention once by appearing, then competes with nothing.
    $htmlPath = Get-UiHtmlPath
    $browser = Get-DefaultBrowserExe
    if ($htmlPath -and $browser) {
        $server = Start-UiServer $htmlPath
        if ($server) {
            try {
                # Dedicated tiny profile dir: guarantees a NEW WINDOW + process
                # we own (a default-profile launch delegates to an existing
                # browser and returns instantly, leaving nothing to close), and
                # avoids touching the user's real browser profile.
                $browserProfile = Join-Path $TempDir ("hermes-update-ui-{0}" -f $PID)
                $browserArgs = @(
                    "--app=http://127.0.0.1:$($server.Port)/",
                    "--user-data-dir=$browserProfile",
                    "--no-first-run", "--no-default-browser-check",
                    "--disable-features=msImplicitSignin",
                    "--window-size=280,320"
                )
                $server.BrowserProc = Start-Process -FilePath $browser -ArgumentList $browserArgs -PassThru
                $server.Profile = $browserProfile
                $script:UiServer = $server
                Write-HandoffLog "shim: default-browser app window on 127.0.0.1:$($server.Port)"
                return
            } catch {
                try { $server.Listener.Stop() } catch {}
                # fall through to WinForms
            }
        }
    }

    try {
        Add-Type -AssemblyName System.Windows.Forms | Out-Null
        Add-Type -AssemblyName System.Drawing | Out-Null
        $light = Get-AppsUseLightTheme
        # Dark seeds are the settled installer palette: neutral charcoal,
        # never brand blue.
        if ($light) {
            $back = [System.Drawing.Color]::White
            $fore = [System.Drawing.ColorTranslator]::FromHtml("#1A1A1A")
            $mute = [System.Drawing.ColorTranslator]::FromHtml("#6B6B6B")
        } else {
            $back = [System.Drawing.ColorTranslator]::FromHtml("#232323")
            $fore = [System.Drawing.ColorTranslator]::FromHtml("#F5F5F5")
            $mute = [System.Drawing.ColorTranslator]::FromHtml("#A8A8A8")
        }
        $form = New-Object System.Windows.Forms.Form
        $form.Text = "Hermes"
        $form.FormBorderStyle = "FixedSingle"
        $form.MaximizeBox = $false
        $form.MinimizeBox = $false
        $form.ControlBox = $false
        $form.ClientSize = New-Object System.Drawing.Size(280, 320)
        $form.StartPosition = "CenterScreen"
        $form.BackColor = $back

        $bar = New-Object System.Windows.Forms.ProgressBar
        $bar.Style = "Marquee"
        $bar.MarqueeAnimationSpeed = 30
        $bar.SetBounds(60, 128, 160, 8)
        $title = New-Object System.Windows.Forms.Label
        $title.Text = "Updating Hermes"
        $title.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 12)
        $title.ForeColor = $fore
        $title.TextAlign = "MiddleCenter"
        $title.SetBounds(16, 156, 248, 28)
        $sub = New-Object System.Windows.Forms.Label
        $sub.Text = Get-UiProgressLine
        $sub.Font = New-Object System.Drawing.Font("Segoe UI", 9)
        $sub.ForeColor = $mute
        $sub.TextAlign = "TopCenter"
        $sub.SetBounds(24, 190, 232, 48)
        $form.Controls.Add($bar)
        $form.Controls.Add($title)
        $form.Controls.Add($sub)
        $form.Show()
        # `cmd start /b` spawned us backgrounded, so the card comes up
        # behind everything without one explicit activation. Claim it ONCE
        # (so the user knows the update started), then never again — the
        # window is decoration and competes with nothing (no TopMost).
        try {
            $form.Activate()
            if ($script:Win32) { [HermesHandoff.Win32]::SetForegroundWindow($form.Handle) | Out-Null }
        } catch {}
        [System.Windows.Forms.Application]::DoEvents()
        $script:Ui = [pscustomobject]@{ Form = $form; Bar = $bar; Title = $title; Sub = $sub; Timer = $null }
        $timer = New-Object System.Windows.Forms.Timer
        $timer.Interval = 1000
        $timer.Add_Tick({
            if ($script:Ui -and $script:Ui.Sub) {
                $script:Ui.Sub.Text = Get-UiProgressLine
            }
        })
        $script:Ui.Timer = $timer
        $timer.Start()
    } catch {
        # Headless session / WinForms unavailable: degrade to log-only.
        $script:Ui = $null
    }
}

function Show-ErrorFinale([string]$Message) {
    # Terse by design: a title + the debug-share pointer. No error text, no
    # log tail -- `hermes debug share` uploads the real evidence and the
    # relaunched Desktop surfaces the result message.
    if ($script:UiServer) {
        # The shim renders the error state itself; leave the window up for
        # the user to read and close. Nothing to hold for — the page keeps
        # the state after the listener dies.
        Publish-UiEvent "error" $Message
        Stop-UiServer -LeaveWindow
        return
    }
    if (-not $script:Ui) { return }
    try {
        $ui = $script:Ui
        if ($ui.Timer) { $ui.Timer.Stop() }
        $ui.Bar.Visible = $false
        $ui.Title.Text = "Failed to update"
        $ui.Sub.Text = "Run `"hermes debug share`" in a terminal to send a report."
        $close = New-Object System.Windows.Forms.Button
        $close.Text = "Close"
        $close.SetBounds(100, 252, 80, 28)
        $close.FlatStyle = "Flat"
        $close.ForeColor = $ui.Title.ForeColor
        $script:ErrorDismissed = $false
        $close.Add_Click({ $script:ErrorDismissed = $true })
        $ui.Form.Controls.Add($close)
        $ui.Form.AcceptButton = $close
        try {
            $ui.Form.Activate()
            if ($script:Win32) { [HermesHandoff.Win32]::SetForegroundWindow($ui.Form.Handle) | Out-Null }
        } catch {}
        # Hold for dismissal so the failure is actually seen, but never park
        # forever -- the marker is already cleaned up and the relaunched
        # Desktop re-surfaces the failure, so walking away costs nothing.
        $deadline = (Get-Date).AddMinutes(5)
        while (-not $script:ErrorDismissed -and (Get-Date) -lt $deadline -and $ui.Form.Visible) {
            [System.Windows.Forms.Application]::DoEvents()
            Start-Sleep -Milliseconds 100
        }
    } catch {}
}

function Show-ManualFinale([string]$Message) {
    # Update landed but the Desktop did not verifiably come back. Same terse
    # shape as the error finale, success glyph semantics: the shim renders
    # `manual` itself; the WinForms card swaps its copy. Held so the user
    # actually sees the instruction — this window is the only surface until
    # they reopen Hermes themselves.
    if ($script:UiServer) {
        Publish-UiEvent "manual" $Message
        Stop-UiServer -LeaveWindow
        return
    }
    if (-not $script:Ui) { return }
    try {
        $ui = $script:Ui
        if ($ui.Timer) { $ui.Timer.Stop() }
        $ui.Bar.Visible = $false
        $ui.Title.Text = "Update complete"
        $ui.Sub.Text = $Message
        $close = New-Object System.Windows.Forms.Button
        $close.Text = "Close"
        $close.SetBounds(100, 252, 80, 28)
        $close.FlatStyle = "Flat"
        $close.ForeColor = $ui.Title.ForeColor
        $script:ErrorDismissed = $false
        $close.Add_Click({ $script:ErrorDismissed = $true })
        $ui.Form.Controls.Add($close)
        $ui.Form.AcceptButton = $close
        try {
            $ui.Form.Activate()
            if ($script:Win32) { [HermesHandoff.Win32]::SetForegroundWindow($ui.Form.Handle) | Out-Null }
        } catch {}
        $deadline = (Get-Date).AddMinutes(5)
        while (-not $script:ErrorDismissed -and (Get-Date) -lt $deadline -and $ui.Form.Visible) {
            [System.Windows.Forms.Application]::DoEvents()
            Start-Sleep -Milliseconds 100
        }
    } catch {}
}

function Close-ProgressWindow {
    if ($script:UiServer) {
        # Success event: the shim flips to the checkmark, then the window
        # closes out from under the user as the Desktop comes back.
        Publish-UiEvent "done" ""
        Stop-UiServer
    }
    if ($script:Ui) {
        try {
            if ($script:Ui.Timer) {
                $script:Ui.Timer.Stop()
                $script:Ui.Timer.Dispose()
            }
            $script:Ui.Form.Close()
        } catch {}
        $script:Ui = $null
    }
}

function Write-Result([bool]$Ok, [int]$Code, [string]$Message, [bool]$ManualAction = $false) {
    # Consumed (read + deleted) by the relaunched Desktop on boot so the
    # user actually SEES how a detached update ended. $ManualAction marks an
    # ok result the user still must act on -- the Desktop surfaces those in
    # a dialog, not just the log (same protocol as posix.sh).
    # Atomic (tmp + rename over): a reader never sees a torn file, and the
    # previous result survives until this one is complete. run_id matches the
    # marker across heartbeat rewrites; started_at stays for older consumers.
    # A failed replace leaves no "<result>.<pid>.tmp" behind (finally).
    $tmp = "$ResultPath.$PID.tmp"
    try {
        $obj = @{
            ok         = $Ok
            exit_code  = $Code
            manual     = $ManualAction
            message    = $Message
            branch     = $Branch
            channel    = $Channel
            run_id     = $script:ResultRunId
            started_at = $script:StartedAt
            warnings   = @($script:Warnings)
            finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        } | ConvertTo-Json -Compress
        [System.IO.File]::WriteAllText($tmp, $obj, (New-Object System.Text.UTF8Encoding $false))
        if ([System.IO.File]::Exists($ResultPath)) {
            [System.IO.File]::Replace($tmp, $ResultPath, [NullString]::Value)
        } else {
            try { [System.IO.File]::Move($tmp, $ResultPath) } catch { [System.IO.File]::Replace($tmp, $ResultPath, [NullString]::Value) }
        }
    } catch {
        Write-HandoffLog "WARNING: could not write the update result: $($_.Exception.Message)"
    } finally {
        if ([System.IO.File]::Exists($tmp)) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    }
}

# Post-commit follow-ups (contract C3): each failed step after `hermes update`
# exited 0 becomes a warning, never a failed update.
$script:Warnings = New-Object System.Collections.Generic.List[string]
$script:FollowupText = New-Object System.Collections.Generic.List[string]
$script:ManualFollowup = $false
$script:Committed = $false
$script:UpdateInterrupted = $false
function Add-Followup([string]$Warning, [string]$Sentence, [switch]$Manual) {
    $script:Warnings.Add($Warning)
    $script:FollowupText.Add($Sentence)
    if ($Manual) { $script:ManualFollowup = $true }
    Write-HandoffLog "WARNING: $Warning"
}

function Test-DesktopAlive {
    if ($DesktopPid -le 0 -or -not $script:DesktopSeenAlive) { return $false }
    return Test-ProcessIdentityLive $DesktopPid $script:DesktopCt
}

function Wait-DesktopExit([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline -and (Test-DesktopAlive)) {
        Start-Sleep -Milliseconds 300
        if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
    }
    return -not (Test-DesktopAlive)
}

function Show-DesktopWindow {
    # Bring the relaunched Desktop's window to the foreground, and make sure it is
    # actually VISIBLE. Returns $false only when the process died, which is a
    # failed launch; a timeout is a visibility miss, not a failed launch.
    #
    # Two properties of the underlying API this has to work around:
    #
    #  * Process.MainWindowHandle is Zero unless the process owns a window that
    #    IsWindowVisible. Electron creates its BrowserWindow hidden and shows it
    #    on `ready-to-show`, so during startup a real HWND exists that this
    #    property reports as "no window" — verified directly: a process owning
    #    four windows, none visible, reports MainWindowHandle 0. Polling it is
    #    still the right way to WAIT for the app to show something, but the
    #    timeout is a real outcome and must be reported rather than swallowed:
    #    silently expiring leaves a running Desktop with no window while the
    #    handoff reports a clean relaunch.
    #  * SW_RESTORE un-minimizes an existing window; it is not the "make this
    #    visible" call for one that was never shown. SW_SHOWNORMAL does both.
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [int]$TimeoutSeconds = 20
    )

    if (-not $script:Win32) { return $true }

    [HermesHandoff.Win32]::AllowSetForegroundWindow($ProcessId) | Out-Null
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $hwnd = [System.IntPtr]::Zero
        try {
            $proc = Get-Process -Id $ProcessId -ErrorAction Stop
            $hwnd = $proc.MainWindowHandle
        } catch {
            Write-HandoffLog "WARNING: relaunched desktop exited before its window appeared"
            return $false
        }
        if ($hwnd -ne [System.IntPtr]::Zero) {
            [HermesHandoff.Win32]::ShowWindow($hwnd, 9) | Out-Null  # SW_RESTORE
            [HermesHandoff.Win32]::ShowWindow($hwnd, 1) | Out-Null  # SW_SHOWNORMAL
            [HermesHandoff.Win32]::SetForegroundWindow($hwnd) | Out-Null
            Write-HandoffLog "focused relaunched desktop window"
            return $true
        }
        Start-Sleep -Milliseconds 400
    }

    Write-HandoffLog ("WARNING: relaunched desktop (pid $ProcessId) is running but produced no visible window within ${TimeoutSeconds}s - the app is up with no GUI. Reopening Hermes from the Start menu restores the existing window.")
    return $true
}

function Start-DesktopRelaunch {
    # Returns $true only when a launch VERIFIABLY happened (WMI accepted and
    # the pid exists, or the fallback spawn returned a live process). The
    # finally block downgrades the on-screen/on-disk outcome when it didn't
    # — the sibling truth contract to posix.sh's launch acceptance.
    if (-not $RelaunchExe) { return $false }
    # electron-builder replaces win-unpacked in place. After a successful
    # update it can remove the old Hermes.exe before writing the replacement,
    # so a one-shot existence check races the rebuild and strands the user.
    $relaunchDeadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Path -LiteralPath $RelaunchExe)) {
        if ((Get-Date) -ge $relaunchDeadline) {
            Write-HandoffLog "WARNING: desktop relaunch executable did not reappear within 120s: $RelaunchExe"
            return $false
        }
        Start-Sleep -Milliseconds 500
        if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
    }
    Write-HandoffLog "relaunching desktop: $RelaunchExe"
    # DO NOT spawn Hermes.exe as our child: Electron/Chromium calls
    # AttachConsole(ATTACH_PARENT_PROCESS) at boot, so a Desktop launched
    # directly from this console PowerShell latches onto OUR console --
    # the console window then outlives the script (it can't close while
    # an attached process lives), and closing it kills the freshly
    # relaunched GUI with it. Create the process via WMI instead: the
    # parent becomes WmiPrvSE.exe and there is no console to inherit or
    # attach -- same detachment explorer.exe gives a normal launch.
    $spawned = $false
    try {
        $workDir = Split-Path -Parent $RelaunchExe
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine      = ('"{0}"' -f $RelaunchExe)
            CurrentDirectory = $workDir
        } -ErrorAction Stop
        if ($r -and $r.ReturnValue -eq 0) {
            Write-HandoffLog "desktop relaunched detached (pid $($r.ProcessId))"
            $spawned = $true
            # Hand our foreground rights to the new Desktop and focus its
            # main window once it exists. A WMI-spawned process starts
            # unfocused, and Windows only lets the CURRENT foreground
            # owner (us, while the progress window is up / just closed)
            # delegate that right.
            try {
                if (-not (Show-DesktopWindow -ProcessId ([int]$r.ProcessId))) {
                    # Process died before showing a window — that is a
                    # failed launch, not merely an unfocused one.
                    $spawned = $false
                }
            } catch {
                Write-HandoffLog "WARNING: could not focus relaunched desktop: $($_.Exception.Message)"
            }
        } else {
            Write-HandoffLog "WARNING: WMI relaunch returned $($r.ReturnValue); falling back"
        }
    } catch {
        Write-HandoffLog "WARNING: WMI relaunch failed: $($_.Exception.Message); falling back"
    }
    if (-not $spawned) {
        # Middle rung: explorer.exe-mediated launch. On some machines
        # Win32_Process.Create fails outright (observed ReturnValue 8,
        # "unknown failure"), and the tethered fallback below re-attaches the
        # Desktop to this console — its stdout then floods the console and the
        # window can't close while the app lives. Explorer re-parents the
        # target exactly like a normal shell launch, giving the same
        # no-console detachment WMI would have. Explorer returns no pid, so
        # verify by watching for a fresh Hermes process.
        try {
            $exeName = [System.IO.Path]::GetFileNameWithoutExtension($RelaunchExe)
            $before = @(Get-Process -Name $exeName -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
            Start-Process -FilePath 'explorer.exe' -ArgumentList ('"{0}"' -f $RelaunchExe) | Out-Null
            $explorerDeadline = (Get-Date).AddSeconds(15)
            while ((Get-Date) -lt $explorerDeadline) {
                $fresh = @(Get-Process -Name $exeName -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id })
                if ($fresh.Count -gt 0) {
                    Write-HandoffLog "desktop relaunched detached via explorer (pid $($fresh[0].Id))"
                    $spawned = $true
                    # Same foreground hand-off as the WMI rung: the new process
                    # starts unfocused and only the current foreground owner
                    # (us) can delegate that right.
                    try {
                        [void](Show-DesktopWindow -ProcessId ([int]$fresh[0].Id))
                    } catch {
                        Write-HandoffLog "WARNING: could not focus relaunched desktop: $($_.Exception.Message)"
                    }
                    break
                }
                Start-Sleep -Milliseconds 400
                if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
            }
            if (-not $spawned) {
                Write-HandoffLog "WARNING: explorer relaunch did not produce a $exeName process; falling back"
            }
        } catch {
            Write-HandoffLog "WARNING: explorer relaunch failed: $($_.Exception.Message); falling back"
        }
    }
    if (-not $spawned) {
        try {
            # Fallback keeps the old behavior (console tie-in and all) --
            # a tethered Desktop beats no Desktop.
            $p = Start-Process -FilePath $RelaunchExe -WorkingDirectory (Split-Path -Parent $RelaunchExe) -PassThru
            Start-Sleep -Milliseconds 1500
            if ($p -and -not $p.HasExited) { $spawned = $true }
            elseif ($p) { Write-HandoffLog "WARNING: fallback relaunch exited immediately" }
        } catch {
            Write-HandoffLog "WARNING: desktop relaunch failed: $($_.Exception.Message)"
        }
    }
    return $spawned
}

# How long a step's pipes get to reach EOF AFTER the step process itself has
# exited (#90455). This is not a step timeout -- the step is already gone by
# the time the clock starts, and everything it wrote is sitting in the pipe
# buffer ready to read, so the grace only has to cover the final drain.
#
# It exists because pipe EOF is not the child's to give. Windows hands the
# write end of a redirected pipe to the child as an INHERITABLE handle, so
# every descendant that is spawned without its own redirection gets a
# duplicate -- and the read side does not see EOF until the last of them
# closes it. `hermes update` deliberately runs its build steps with stdout
# inherited (hermes_cli/main.py, the tee-stderr runner), so the tree under a
# step is arbitrarily deep and not something this script can enumerate. When
# one of those descendants is a resident gateway, the pipe stays open for the
# life of the gateway, i.e. forever.
#
# Overridable so the pipe-drain self-test does not have to sit out the real
# grace; not documented as a user knob.
$script:StepDrainGraceSeconds = 20
if ($env:HERMES_UPDATE_PIPE_DRAIN_SECONDS) {
    $parsedGrace = 0
    if ([int]::TryParse($env:HERMES_UPDATE_PIPE_DRAIN_SECONDS, [ref]$parsedGrace) -and $parsedGrace -ge 0) {
        $script:StepDrainGraceSeconds = $parsedGrace
    }
}

# A live step also needs a ceiling. The pipe-drain bound above only starts
# after the child exits, so it cannot recover a child that completed its visible
# work and then parks forever inside finalization (#95589). Silence is only the
# cancellation trigger, never evidence that the process tree is safe to overlap:
# every step is assigned to a private, non-breakaway Windows job and a timed-out
# step is retryable only after that job reports zero active processes.
$script:StepIdleTimeoutSeconds = 600
if ($env:HERMES_UPDATE_STEP_IDLE_SECONDS) {
    $parsedIdle = 0
    if ([int]::TryParse($env:HERMES_UPDATE_STEP_IDLE_SECONDS, [ref]$parsedIdle) -and $parsedIdle -gt 0) {
        $script:StepIdleTimeoutSeconds = $parsedIdle
    }
}

# The Desktop's quit can first join a managed SSH update it is running, which
# legitimately outlasts a fixed 30 s. Past this ceiling the hand-off still
# refuses, so nothing is replaced under a live Desktop. Overridable so the
# self-tests need not sit it out; not documented as a user knob.
$script:DesktopExitSeconds = 150
if ($env:HERMES_UPDATE_DESKTOP_EXIT_SECONDS) {
    $parsedExit = 0
    if ([int]::TryParse($env:HERMES_UPDATE_DESKTOP_EXIT_SECONDS, [ref]$parsedExit) -and $parsedExit -gt 0) {
        $script:DesktopExitSeconds = $parsedExit
    }
}

# Silence on the pipes is NOT silence in the update. `hermes update` captures
# the (very loud) Electron/vite build into logs/update.log instead of its own
# stdout (hermes_cli/update_cmd.py, the update-log tee), so a real update is
# routinely stdout-silent for 40+ minutes while demonstrably progressing. An
# idle ceiling that watched only stdout/stderr would cancel every healthy
# large update at StepIdleTimeoutSeconds. The drain therefore also counts
# growth of this file (size or mtime) as progress before declaring a stall.
# Overridable so the pipe-drain self-test can point it at its own file; not
# documented as a user knob.
$script:StepProgressLogPath = Join-Path $LogDir "update.log"
if ($env:HERMES_UPDATE_PROGRESS_LOG) {
    $script:StepProgressLogPath = $env:HERMES_UPDATE_PROGRESS_LOG
}

function Get-StepProgressLogStamp {
    # Size + mtime fingerprint of the update log; $null when absent or
    # unreadable. Comparing fingerprints between passes is how the idle
    # watchdog sees a build that streams to update.log instead of stdout.
    try {
        $fi = New-Object System.IO.FileInfo($script:StepProgressLogPath)
        if (-not $fi.Exists) { return $null }
        return ('{0}:{1}' -f $fi.Length, $fi.LastWriteTimeUtc.Ticks)
    } catch {
        return $null
    }
}

. (Join-Path $PSScriptRoot 'update-job.ps1')

function Step-PipeDrain($Reader, [ref]$Task, $Buffer, $Sink, [ref]$Moved) {
    # Advance one redirected pipe by whatever has already arrived, without
    # ever blocking. Returns $true once the pipe has reached EOF (or its read
    # faulted), $false while more may still come. Sets $Moved when this call
    # actually consumed bytes, so the caller can tell a busy pipe from a quiet
    # one and skip its idle wait.
    #
    # The chunked ReadAsync loop is the point: ReadToEndAsync().Result cannot
    # hand back a partial read, so abandoning it loses the whole step's output.
    # Draining into a StringBuilder means an abandoned pipe still yields every
    # byte that arrived before we gave up.
    if ($null -eq $Task.Value) { return $true }
    if (-not $Task.Value.IsCompleted) { return $false }
    $count = 0
    try {
        $count = $Task.Value.Result
    } catch {
        # Faulted/cancelled read: treat as EOF rather than retrying forever.
        $Task.Value = $null
        return $true
    }
    if ($count -le 0) { $Task.Value = $null; return $true }
    [void]$Sink.Append($Buffer, 0, $count)
    $Moved.Value = $true
    $Task.Value = $Reader.ReadAsync($Buffer, 0, $Buffer.Length)
    return $false
}

function Invoke-HermesStep([string]$Exe, [string[]]$HermesArgs, [string]$Tag) {
    # The window does not stream child output, so no line-pump: both pipes
    # drain asynchronously (no deadlock however chatty the child) while a small
    # DoEvents loop keeps the marquee animating through long silent
    # stretches (pip installs) -- the old EndOfStream pump blocked on quiet
    # children and froze it. Full output still lands in the hand-off log
    # afterwards, where `hermes debug share` picks it up.
    #
    # The drain is bounded once the step exits (#90455). Waiting for pipe EOF
    # is waiting on the step's whole surviving descendant tree, and this
    # function sits upstream of every terminal obligation the hand-off has --
    # .hermes-update-result.json, clearing .hermes-update-in-progress,
    # relaunching the Desktop. One resident grandchild holding an inherited
    # handle used to strand all three and leave the Desktop on "Updating
    # Hermes" until the user killed something by hand. Losing the tail of a
    # log is the strictly better failure.
    # System.Diagnostics.Process directly: Start-Process's .ExitCode is
    # unreliably $null under PS 5.1 even with the Handle-touch workaround.
    # CREATE_SUSPENDED closes the startup race: no updater instruction can run
    # before the process is assigned to its private job and resumed.
    $arguments = ($HermesArgs | ForEach-Object { '"{0}"' -f ($_ -replace '"', '\"') }) -join ' '
    # CreateProcess inherits this process's environment. Set Python's encoding
    # and buffering only for the atomic launch, then restore the hand-off host.
    # Historical user-bin publication could be a command file rather than a
    # native launcher. Keep the wrapper inside the same supervised job.
    if ([IO.Path]::GetExtension($Exe) -eq '.cmd') {
        if ($Exe -match '[%!"\x0D\x0A]' -or @($HermesArgs | Where-Object { $_ -match '[%!"\x0D\x0A]' }).Count) {
            throw 'The legacy command launcher cannot safely quote this update target; refresh the installation launcher first.'
        }
        $arguments = '/d /s /c ""' + $Exe + '" ' + $arguments + '"'
        $Exe = $env:ComSpec
    }
    $savedPythonIoEncoding = $env:PYTHONIOENCODING
    $savedPythonUtf8 = $env:PYTHONUTF8
    $savedPythonUnbuffered = $env:PYTHONUNBUFFERED
    try {
        $env:PYTHONIOENCODING = "utf-8"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONUNBUFFERED = "1"
        $started = [HermesUpdateJob]::StartAssigned($Exe, $arguments)
    } finally {
        if ($null -eq $savedPythonIoEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $savedPythonIoEncoding }
        if ($null -eq $savedPythonUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $savedPythonUtf8 }
        if ($null -eq $savedPythonUnbuffered) { Remove-Item Env:PYTHONUNBUFFERED -ErrorAction SilentlyContinue } else { $env:PYTHONUNBUFFERED = $savedPythonUnbuffered }
    }
    $proc = $started.Process
    # C1 rule 6 + SPEC 5: the update child is the marker's delegate from its
    # first instruction. It is still SUSPENDED here (and its job kills it if
    # this script dies first), so no update work can run before the delegate
    # line is published under the marker lock -- and none runs when it can't be.
    if ($Tag -eq 'update') {
        $delegate = Add-MarkerDelegate @($proc.Id)
        if ($delegate -notin @('published', 'kept')) {
            [void][HermesUpdateJob]::TerminateAndWait($started.Job, 1, 5000)
            [HermesUpdateJob]::Close($started.Job)
            throw "could not name the update process as the update marker's delegate ($delegate); nothing was run"
        }
    }
    [HermesUpdateJob]::Resume($started)
    $stdoutReader = $started.StandardOutput
    $stderrReader = $started.StandardError
    $job = $started.Job
    # A job gives cancellation a kernel-enforced tree boundary. We deliberately
    # do NOT set KILL_ON_JOB_CLOSE: successful updates may start detached
    # services that are meant to outlive this pipe reader. Descendants cannot
    # break away from a default job, but survive when its handle is closed after
    # a normal step.

    $outSink = New-Object System.Text.StringBuilder
    $errSink = New-Object System.Text.StringBuilder
    $outBuffer = New-Object char[] 16384
    $errBuffer = New-Object char[] 16384
    $outTask = $stdoutReader.ReadAsync($outBuffer, 0, $outBuffer.Length)
    $errTask = $stderrReader.ReadAsync($errBuffer, 0, $errBuffer.Length)
    $abandonAt = $null
    $abandoned = $false
    $lastProgressAt = Get-Date
    $progressLogStamp = Get-StepProgressLogStamp
    $jobActivity = [HermesUpdateJob]::Activity($job)
    $stalled = $false
    while ($true) {
        $moved = $false
        $outDone = Step-PipeDrain $stdoutReader ([ref]$outTask) $outBuffer $outSink ([ref]$moved)
        $errDone = Step-PipeDrain $stderrReader ([ref]$errTask) $errBuffer $errSink ([ref]$moved)
        if ($moved) {
            $lastProgressAt = Get-Date
        } elseif ($job -ne [IntPtr]::Zero) {
            # CPU or I/O spent inside the job is progress too: a pipe-silent
            # wheel download or extraction is busy, not stalled. One cheap
            # kernel query per idle pass; never on the hot drain path.
            $currentActivity = [HermesUpdateJob]::Activity($job)
            if ($currentActivity -ne $jobActivity) {
                $jobActivity = $currentActivity
                $lastProgressAt = Get-Date
            }
        }
        if ($proc.HasExited) {
            if ($outDone -and $errDone) { break }
            # Clock starts at the step's exit, not at its start: a slow step is
            # not a stuck one, and only a pipe outliving its process is.
            if ($null -eq $abandonAt) {
                $abandonAt = (Get-Date).AddSeconds($script:StepDrainGraceSeconds)
            } elseif ((Get-Date) -ge $abandonAt) {
                $abandoned = $true
                break
            }
        } elseif (-not $stalled -and $job -ne [IntPtr]::Zero -and ((Get-Date) - $lastProgressAt).TotalSeconds -ge $script:StepIdleTimeoutSeconds) {
            # Quiet pipes are how a healthy `hermes update` looks for 40+
            # minutes: its build output streams to logs/update.log, not the
            # child's stdout. Growth of that file is progress -- reset the
            # clock instead of cancelling. Stat'd only once the ceiling is
            # otherwise reached (at most once per 150ms pass after that), so
            # the hot drain path never touches the filesystem.
            $currentLogStamp = Get-StepProgressLogStamp
            if ($currentLogStamp -ne $progressLogStamp) {
                $progressLogStamp = $currentLogStamp
                $lastProgressAt = Get-Date
            } else {
                # The child is alive but has produced no observable progress
                # -- neither on its pipes nor in the update log -- for the
                # whole bound. Terminate the job, not just its direct process:
                # retrying while a descendant still mutates the checkout,
                # venv, or release tree can overlap two installers and
                # corrupt the install.
                Write-HandoffLog ("{0}!| step stalled: no stdout/stderr, no update.log growth and no CPU/IO in its process tree for {1}s while pid {2} remained alive; cancelling its process tree." -f $Tag, $script:StepIdleTimeoutSeconds, $proc.Id)
                $stalled = [HermesUpdateJob]::TerminateAndWait($job, 124, 10000)
                if (-not $stalled) {
                    Write-HandoffLog ("{0}!| process-tree cancellation could not prove quiescence; refusing the timeout retry." -f $Tag)
                    $script:TreeSafeToFinalize = $false
                    $script:UnquiescedPids = @([HermesUpdateJob]::ProcessIds($job))
                    [HermesUpdateJob]::Close($job)
                    throw "Unable to quiesce stalled update process tree"
                }
            }
        }
        # Only idle when both pipes came up empty this pass, and idle on the
        # reads themselves rather than on the clock.
        #
        # Sleeping after a chunk that DID arrive meters the drain at one buffer
        # per tick (16 KiB / 150ms ~ 107 KB/s), and because the pipe then backs
        # up that is backpressure on the running step, not just a slow read --
        # a chatty step blocks on write() waiting for us. Waiting for EOF and
        # trickling toward it are two ways to make a fast step slow, and this
        # function is upstream of the hand-off's obligations either way.
        #
        # A flat sleep is not enough on its own: a freshly issued ReadAsync is
        # rarely complete by the very next pass, so the loop would sleep 150ms
        # between chunks anyway. WaitAny returns the instant either pipe has
        # something (and immediately if one already does), and expires on its
        # own so a silent step still animates the marquee and still advances
        # the abandon deadline.
        if (-not $moved) {
            $live = @($outTask, $errTask) | Where-Object { $null -ne $_ }
            if ($live.Count -gt 0) {
                [void][System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]$live, 150)
            } else {
                Start-Sleep -Milliseconds 150
            }
        }
        if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
        Update-MarkerHeartbeat
    }
    # Bounded overload deliberately: the argument-less overload also waits on
    # redirected streams, which is the very wait we just bounded. HasExited is
    # already true here, so this call only settles ExitCode.
    [void]$proc.WaitForExit(5000)
    if ($abandoned) {
        Write-HandoffLog ("{0}!| pipe drain abandoned after {1}s: '{0}' exited but a surviving descendant still holds its stdout/stderr handles. Continuing the hand-off with the output captured so far (#90455)." -f $Tag, $script:StepDrainGraceSeconds)
    }
    $outText = $outSink.ToString()
    $errText = $errSink.ToString()
    foreach ($ln in ($outText -split "`r?`n")) {
        if ($ln.Trim()) { Write-HandoffLog ("{0}| {1}" -f $Tag, $ln) }
    }
    foreach ($ln in ($errText -split "`r?`n")) {
        if ($ln.Trim()) { Write-HandoffLog ("{0}!| {1}" -f $Tag, $ln) }
    }
    $all = $outText
    if ($errText) { $all += "`n" + $errText }
    $code = if ($stalled) { 124 } else { $proc.ExitCode }
    [HermesUpdateJob]::Close($job)
    return @{ Code = $code; Output = $all; TreeQuiesced = (-not $stalled -or $proc.HasExited); StartedAfterJobAssignment = $true }
}

# `hermes update` can COMPLETE (its output carries "✓ Update complete!") and
# still be killed with the idle-watchdog sentinel 124: the post-update phase
# (gateway restart hand-off) stayed alive and silent past the ceiling, so
# Invoke-HermesStep terminated the tree (#96205). The install is done; failing
# would keep the old Desktop and a legacy install would re-run the whole update.
# Surface success so the hand-off verifies, restores the gateways and relaunches.
# Only 124 is remapped, and never when anything after the banner reports a
# failure: the restart/verify phase prints "✗ Update not complete", "Update
# incomplete — …", "✗ <unit> failed to come back after restart" or
# "verification incomplete" there. \u2717 (✗) stays an escape: Windows
# PowerShell reads this BOM-less script as ANSI, never as UTF-8.
function Resolve-HermesUpdateOutcome($StepResult) {
    $banner = if ($StepResult.Output) { $StepResult.Output.LastIndexOf('Update complete!') } else { -1 }
    if ($StepResult.Code -eq 124 -and $banner -ge 0 -and $StepResult.Output.Substring($banner) -notmatch 'incomplete|not complete|\u2717') {
        Write-HandoffLog "update completed before the idle watchdog killed its finalizing step (exit 124); treating it as success, not retrying (#96205)"
        $StepResult.Code = 0
        $script:UpdateInterrupted = $true
    }
    return $StepResult
}

# -- The commit point (contract C3) --------------------------------------------
# `hermes update` exits 0 once committed, except an interrupt (130) or a parked
# autostash (1) after the commit point, and a crash of a committed run. Its own
# receipt says which (same rule as posix.sh::update_committed_after_exit): the
# run's receipt -- matched by the correlation id handed to it, finalized
# (finished_at set) -- with outcome success | partial (the user still has to
# act) | interrupted (Ctrl-C after the code moved). A reconciled "interrupted"
# record keeps finished_at null, so it never matches.
function Get-CommittedReceiptOutcome {
    $root = $HermesHome
    # hermes_constants.get_default_hermes_root: a <root>\profiles\<name> home files under <root>
    $parent = Split-Path -Parent $root
    if ($parent -and (Split-Path -Leaf $parent) -eq 'profiles') { $root = Split-Path -Parent $parent }
    $path = Join-Path $root 'logs\update_receipts\latest.json'
    $receipt = $null
    try { $receipt = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8) | ConvertFrom-Json } catch { return $null }
    if ($null -eq $receipt -or -not $script:UpdateCorrelation) { return $null }
    if ([string]$receipt.correlation_id -cne $script:UpdateCorrelation -or -not $receipt.finished_at) { return $null }
    $outcome = [string]$receipt.outcome
    if ($outcome -cin @('success', 'partial', 'interrupted')) { return $outcome }
    return $null
}

function Set-InstallRootCurrentDirectory([string]$Root) {
    $resolved = [System.IO.Path]::GetFullPath($Root)
    [Environment]::CurrentDirectory = $resolved
    return $resolved
}

$finalCode = 1
$manualAction = $false
$finalMsg = "update did not complete"
$script:TreeSafeToFinalize = $true

# ── -SelfTestRelaunchWindow: prove the relaunch focus path reports what it did ──
# Guards the state this fixture was written from. After an update the Desktop
# process is up but owns only HIDDEN windows, and Process.MainWindowHandle
# returns Zero unless a window is visible -- so the poll never yields a handle,
# the loop expires, and with no timeout branch nothing is logged at all. The
# hand-off then reports a clean relaunch while the user has a running app and
# no window. Exits before any marker/desktop machinery, like the others.
#
# Three arms, each a real child process:
#
#   visible -- creates and shows a window. Must log "focused relaunched desktop
#              window" and return $true.
#   hidden  -- creates a window and never shows it (the observed failure). Must
#              log the timeout WARNING and still return $true: the process is
#              running, so this is a visibility miss, not a failed launch, and
#              the caller must not fall through to launching a second instance.
#   dead    -- exits immediately. Must log "exited before its window appeared"
#              and return $false so the caller can fall back.
#
# The child signals readiness with a file once its window exists, and the probe
# only runs after that. Without the handshake the arm races PowerShell's own
# startup (Add-Type of WinForms alone can take seconds) and a genuinely visible
# window reads as a timeout -- which is what the first version of this fixture
# did, on this machine.
if ($SelfTestRelaunchWindow) {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    $windowTimeout = 3
    if ($env:HERMES_SELFTEST_WINDOW_TIMEOUT) { $windowTimeout = [int]$env:HERMES_SELFTEST_WINDOW_TIMEOUT }
    $powershell = Join-Path $PSHOME "powershell.exe"
    $stamp = [Guid]::NewGuid().ToString("N")
    $childPs1 = Join-Path $TempDir "hermes-relaunch-window-$stamp.ps1"
    # A literal here-string: the child's own $f/$Mode must not expand here.
    $childSource = @'
param([string]$Mode, [string]$ReadyFile)
if ($Mode -eq "dead") { exit 0 }
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.Form
$f.Text = "HermesRelaunchProbe"
$f.Width = 640
$f.Height = 480
$null = $f.Handle
if ($Mode -eq "visible") { $f.Show() | Out-Null }
# Signal only once the HWND exists (and is shown, for the visible arm).
Set-Content -LiteralPath $ReadyFile -Value "ready" -Encoding UTF8
if ($Mode -eq "visible") {
    # A form with no message loop never finishes rendering; pump it so the
    # window is genuinely on screen rather than a pending Show().
    $until = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $until) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
} else {
    Start-Sleep -Seconds 30
}
'@
    Set-Content -LiteralPath $childPs1 -Value $childSource -Encoding UTF8

    $results = @()
    foreach ($mode in @("visible", "hidden", "dead")) {
        $readyFile = Join-Path $TempDir "hermes-relaunch-ready-$stamp-$mode"
        Remove-Item -LiteralPath $readyFile -Force -ErrorAction SilentlyContinue
        $before = @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue).Count
        $child = Start-Process -FilePath $powershell `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $childPs1,
                            "-Mode", $mode, "-ReadyFile", $readyFile) `
            -PassThru
        # Wait for the window to exist rather than guessing at startup cost.
        $readyDeadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $readyDeadline) {
            if (Test-Path -LiteralPath $readyFile) { break }
            if ($child.HasExited) { break }
            Start-Sleep -Milliseconds 200
        }
        $shown = Show-DesktopWindow -ProcessId ([int]$child.Id) -TimeoutSeconds $windowTimeout
        $tail = ((@(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue) |
            Select-Object -Skip $before) -join " | ")
        $results += [pscustomobject]@{ Mode = $mode; Returned = $shown; Logged = $tail }
        try { if (-not $child.HasExited) { $child | Stop-Process -Force -ErrorAction SilentlyContinue } } catch {}
        Remove-Item -LiteralPath $readyFile -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $childPs1 -Force -ErrorAction SilentlyContinue

    $visible = @($results | Where-Object { $_.Mode -eq "visible" })[0]
    $hidden  = @($results | Where-Object { $_.Mode -eq "hidden" })[0]
    $dead    = @($results | Where-Object { $_.Mode -eq "dead" })[0]

    $failures = @()
    if (-not $visible.Returned) { $failures += "visible: expected `$true" }
    if ($visible.Logged -notmatch "focused relaunched desktop window") { $failures += "visible: expected the focus log line" }
    if (-not $hidden.Returned) { $failures += "hidden: expected `$true -- the process is up, this is a visibility miss" }
    if ($hidden.Logged -notmatch "no visible window") { $failures += "hidden: expected the timeout WARNING naming the symptom" }
    if ($dead.Returned) { $failures += "dead: expected `$false" }
    if ($dead.Logged -notmatch "exited before its window appeared") { $failures += "dead: expected the exited-before-window WARNING" }

    foreach ($r in $results) {
        Write-Host "SELF-TEST relaunch-window[$($r.Mode)] returned=$($r.Returned)"
        if ($r.Logged) { Write-Host "  log: $($r.Logged)" }
    }
    if ($failures.Count -gt 0) {
        foreach ($f in $failures) { Write-Host "SELF-TEST FAIL: $f" }
        exit 1
    }
    Write-Host "SELF-TEST relaunch-window: all arms passed"
    exit 0
}

# ── -SelfTestUi: drive the shim to both terminal states, no update ─────────
# Manual QA for the Edge shell without a checkout or a real update. Exits
# before the marker/desktop/venv machinery — touches nothing. Off Windows
# (or without Edge) the loopback server still starts and the URL prints, so
# the page can be QA'd in any browser; HERMES_SELFTEST_FAIL=1 exercises the
# error state, HERMES_SELFTEST_HOLD_SECONDS delays the terminal event.
if ($SelfTestUi) {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    Show-ProgressWindow
    if (-not $script:UiServer) {
        $htmlPath = Get-UiHtmlPath
        if ($htmlPath) {
            $script:UiServer = Start-UiServer $htmlPath
        }
    }
    if ($script:UiServer) {
        Write-Host "SELF-TEST: shim at http://127.0.0.1:$($script:UiServer.Port)/"
    }
    Write-HandoffLog "SELF-TEST: shim simulation (no update will run)"
    $hold = 6
    if ($env:HERMES_SELFTEST_HOLD_SECONDS) { $hold = [int]$env:HERMES_SELFTEST_HOLD_SECONDS }
    Publish-UiProgress "Testing quiet update"
    Start-Sleep -Seconds $hold
    if ($env:HERMES_SELFTEST_FAIL) {
        Show-ErrorFinale "self-test error state"
    } else {
        Close-ProgressWindow
    }
    exit 0
}

# -SelfTestPipeDrain: prove Invoke-HermesStep survives a leaked pipe ------
# The #90455 deadlock needs no update, no checkout and no Hermes install to
# reproduce -- only a step whose grandchild outlives it holding the inherited
# write end of the redirected pipe. That is exactly what this builds, so the
# fix has an executable proof on Windows instead of a source-grep. Exits
# before any marker/desktop machinery, same as -SelfTestUi; touches nothing
# but its own temp files.
#
# Three arms cover the independent wait modes:
#
#   leak  -- a step whose grandchild outlives it. Guards the #90455 deadlock:
#            the drain must abandon rather than wait out the descendant.
#   flood -- a chatty step that leaks nothing. Guards the other cliff: a drain
#            that idles after every chunk it reads is metered at one buffer per
#            tick, which backpressures the running step. Waiting for EOF and
#            trickling toward it are both ways to make a fast step slow.
#   stall -- a step that remains alive after its visible work and emits no more
#            output. Guards #95589: the hand-off must terminate it and reach its
#            retry/finally recovery rather than strand the Desktop.
#   logstall -- a step that is silent on its pipes but keeps growing the
#            update log, the shape of every real `hermes update` build (output
#            goes to logs/update.log, not stdout, for 40+ minutes). Guards the
#            watchdog's other cliff: the idle ceiling must count update.log
#            growth as progress and must NOT kill the healthy step.
if ($SelfTestPipeDrain) {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    $hold = 60
    if ($env:HERMES_SELFTEST_HOLD_SECONDS) { $hold = [int]$env:HERMES_SELFTEST_HOLD_SECONDS }
    $floodKb = 8192
    if ($env:HERMES_SELFTEST_FLOOD_KB) { $floodKb = [int]$env:HERMES_SELFTEST_FLOOD_KB }
    # $PSHOME is this interpreter's own directory -- no hardcoded system path.
    $powershell = Join-Path $PSHOME "powershell.exe"
    $stamp = [Guid]::NewGuid().ToString("N")
    $childPs1 = Join-Path $TempDir "hermes-pipe-drain-$stamp.ps1"
    $floodPs1 = Join-Path $TempDir "hermes-pipe-flood-$stamp.ps1"
    $pidFile = Join-Path $TempDir "hermes-pipe-drain-$stamp.pid"
    $stallPs1 = Join-Path $TempDir "hermes-step-stall-$stamp.ps1"
    $stallPidFile = Join-Path $TempDir "hermes-step-stall-$stamp.pid"
    $stallGrandchildPidFile = Join-Path $TempDir "hermes-step-stall-grandchild-$stamp.pid"
    $logStallPs1 = Join-Path $TempDir "hermes-step-logstall-$stamp.ps1"
    $logStallProgress = Join-Path $TempDir "hermes-step-logstall-$stamp.update.log"
    # UseShellExecute=$false with no redirection is what makes the grandchild
    # inherit our stdout/stderr -- the whole point of the fixture. Anything
    # that redirects (Start-Process, subprocess with stdout=DEVNULL) would
    # close the handle and the deadlock would not reproduce.
    $childSource = @'
param([int]$Hold, [string]$PidFile)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = Join-Path $PSHOME "powershell.exe"
$psi.Arguments = "-NoProfile -Command Start-Sleep -Seconds $Hold"
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$grandchild = [System.Diagnostics.Process]::Start($psi)
[System.IO.File]::WriteAllLines($PidFile, @([string]$grandchild.Id, [string][System.Diagnostics.Stopwatch]::GetTimestamp()))
Write-Output "pipe-drain step output"
[Console]::Out.Flush()
exit 7
'@
    # Writes straight to the console stream, holding nothing: a step that is
    # merely loud. `hermes update` is this shape -- the Electron/vite build
    # alone is megabytes. Few large lines rather than many small ones on
    # purpose: Write-HandoffLog is one Add-Content per line and runs inside the
    # measured window, so line-heavy output would time the logger instead of
    # the drain.
    $floodSource = @'
param([int]$Kb)
$chunk = "x" * (131072 - 1)
for ($i = 0; $i -lt [Math]::Ceiling($Kb / 128); $i++) { [Console]::Out.Write($chunk + "`n") }
[Console]::Out.Flush()
exit 5
'@
    $stallSource = @'
param([int]$Hold, [string]$PidFile, [string]$GrandchildPidFile)
[System.IO.File]::WriteAllText($PidFile, [string]$PID)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = Join-Path $PSHOME "powershell.exe"
$psi.Arguments = "-NoProfile -Command Start-Sleep -Seconds $Hold"
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$grandchild = [System.Diagnostics.Process]::Start($psi)
[System.IO.File]::WriteAllText($GrandchildPidFile, [string]$grandchild.Id)
Write-Output "step entered silent finalization"
[Console]::Out.Flush()
Start-Sleep -Seconds $Hold
exit 0
'@
    # Pipe-silent but log-writing: one stdout line, then only Add-Content to
    # the progress log every second. With Hold far above the idle ceiling,
    # surviving to exit 3 proves the watchdog counted the log growth.
    $logStallSource = @'
param([int]$Hold, [string]$ProgressLog)
Write-Output "silent but logging"
[Console]::Out.Flush()
for ($i = 0; $i -lt $Hold; $i++) { Add-Content -LiteralPath $ProgressLog -Value ("build tick {0}" -f $i); Start-Sleep -Seconds 1 }
exit 3
'@
    [System.IO.File]::WriteAllText($childPs1, $childSource)
    [System.IO.File]::WriteAllText($floodPs1, $floodSource)
    [System.IO.File]::WriteAllText($stallPs1, $stallSource)
    [System.IO.File]::WriteAllText($logStallPs1, $logStallSource)
    # The leak arm measures post-exit draining, not cold PowerShell startup.
    $savedIdle = $script:StepIdleTimeoutSeconds
    try {
        $script:StepIdleTimeoutSeconds = 120
        $res = Invoke-HermesStep $powershell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $childPs1,
            "-Hold", [string]$hold, "-PidFile", $pidFile
        ) "pipedrain"
    } finally {
        $script:StepIdleTimeoutSeconds = $savedIdle
    }
    $returnedAt = [System.Diagnostics.Stopwatch]::GetTimestamp()
    $elapsed = [double]::PositiveInfinity
    $leakPid = 0
    if (Test-Path -LiteralPath $pidFile) {
        $leakReceipt = @(Get-Content -LiteralPath $pidFile)
        [void][int]::TryParse($leakReceipt[0].Trim(), [ref]$leakPid)
        if ($leakReceipt.Count -eq 2) {
            $elapsed = [Math]::Round(($returnedAt - [long]$leakReceipt[1]) / [double][System.Diagnostics.Stopwatch]::Frequency, 2)
        }
    }
    $leakAlive = $false
    if ($leakPid -gt 0) {
        $leakAlive = [bool](Get-Process -Id $leakPid -ErrorAction SilentlyContinue)
        Stop-Process -Id $leakPid -Force -ErrorAction SilentlyContinue
    }

    $floodSw = [System.Diagnostics.Stopwatch]::StartNew()
    $flood = Invoke-HermesStep $powershell @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $floodPs1,
        "-Kb", [string]$floodKb
    ) "pipeflood"
    $floodSw.Stop()
    $floodElapsed = [Math]::Round($floodSw.Elapsed.TotalSeconds, 2)
    $floodBytes = $flood.Output.Length

    $stallSw = [System.Diagnostics.Stopwatch]::StartNew()
    $stall = Invoke-HermesStep $powershell @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $stallPs1,
        "-Hold", [string]$hold, "-PidFile", $stallPidFile,
        "-GrandchildPidFile", $stallGrandchildPidFile
    ) "stepstall"
    $stallSw.Stop()
    $stallElapsed = [Math]::Round($stallSw.Elapsed.TotalSeconds, 2)
    $stallPid = 0
    if (Test-Path -LiteralPath $stallPidFile) {
        [void][int]::TryParse((Get-Content -LiteralPath $stallPidFile -Raw).Trim(), [ref]$stallPid)
    }
    $stallAlive = $stallPid -gt 0 -and [bool](Get-Process -Id $stallPid -ErrorAction SilentlyContinue)
    if ($stallAlive) { Stop-Process -Id $stallPid -Force -ErrorAction SilentlyContinue }
    $stallGrandchildPid = 0
    if (Test-Path -LiteralPath $stallGrandchildPidFile) {
        [void][int]::TryParse((Get-Content -LiteralPath $stallGrandchildPidFile -Raw).Trim(), [ref]$stallGrandchildPid)
    }
    $stallGrandchildAlive = $stallGrandchildPid -gt 0 -and [bool](Get-Process -Id $stallGrandchildPid -ErrorAction SilentlyContinue)
    if ($stallGrandchildAlive) { Stop-Process -Id $stallGrandchildPid -Force -ErrorAction SilentlyContinue }

    # logstall arm: point the watchdog's progress log at the fixture's file
    # for exactly this step, restore afterwards so the other arms' contract
    # (no update.log in play) is untouched.
    $savedProgressLogPath = $script:StepProgressLogPath
    $script:StepProgressLogPath = $logStallProgress
    $logStallSw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $logstall = Invoke-HermesStep $powershell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $logStallPs1,
            "-Hold", [string]$hold, "-ProgressLog", $logStallProgress
        ) "logstall"
    } finally {
        $script:StepProgressLogPath = $savedProgressLogPath
    }
    $logStallSw.Stop()
    $logStallElapsed = [Math]::Round($logStallSw.Elapsed.TotalSeconds, 2)

    Remove-Item -LiteralPath $childPs1, $floodPs1, $stallPs1, $logStallPs1, $pidFile, $stallPidFile, $stallGrandchildPidFile, $logStallProgress -Force -ErrorAction SilentlyContinue

    # The grandchild still being alive at return is what makes this a proof
    # rather than a timing coincidence: the pipe was demonstrably still open.
    $budget = $script:StepDrainGraceSeconds + 30
    # A sleep-per-chunk drain moves 16 KiB/150ms ~ 107 KB/s, so 8 MiB takes
    # ~76s. Generous enough for a loaded CI runner, far under the trickle.
    $floodBudget = 25
    $problems = @()
    if (-not $leakAlive) { $problems += "handle-holding grandchild was not alive on return (fixture did not reproduce the leak)" }
    if ($elapsed -ge $budget) { $problems += "leak arm returned in ${elapsed}s, over the ${budget}s budget" }
    if ($res.Code -ne 7) { $problems += "leak arm exit code $($res.Code), expected 7" }
    if ($res.Output -notmatch "pipe-drain step output") { $problems += "leak arm step output was lost" }
    if ($floodElapsed -ge $floodBudget) { $problems += "flood arm returned in ${floodElapsed}s, over the ${floodBudget}s budget -- the drain is metering itself, which backpressures the step" }
    if ($flood.Code -ne 5) { $problems += "flood arm exit code $($flood.Code), expected 5" }
    if ($floodBytes -lt ($floodKb * 1024)) { $problems += "flood arm captured $floodBytes bytes of $($floodKb * 1024)" }
    $stallBudget = $script:StepIdleTimeoutSeconds + 30
    if ($stallElapsed -ge $stallBudget) { $problems += "stall arm returned in ${stallElapsed}s, over the ${stallBudget}s budget" }
    if ($stall.Code -ne 124) { $problems += "stall arm exit code $($stall.Code), expected 124" }
    if ($stall.Output -notmatch "step entered silent finalization") { $problems += "stall arm step output was lost" }
    if ($stallAlive) { $problems += "stalled child pid $stallPid remained alive after Invoke-HermesStep returned" }
    if ($stallGrandchildAlive) { $problems += "stalled descendant pid $stallGrandchildPid remained alive after Invoke-HermesStep returned" }
    if (-not $stall.TreeQuiesced) { $problems += "stall arm returned without proving its process tree quiescent" }
    if (-not $stall.StartedAfterJobAssignment) { $problems += "stall arm started before cancellation-job assignment" }
    $logStallBudget = $hold + 60
    if ($logstall.Code -ne 3) { $problems += "logstall arm exit code $($logstall.Code), expected 3 -- the idle watchdog killed a pipe-silent step whose progress was visible as update.log growth (the shape of every real 40+ min build)" }
    if ($logstall.Output -notmatch "silent but logging") { $problems += "logstall arm step output was lost" }
    if ($logStallElapsed -ge $logStallBudget) { $problems += "logstall arm returned in ${logStallElapsed}s, over the ${logStallBudget}s budget" }

    $detail = "leak: elapsed=${elapsed}s budget=${budget}s code=$($res.Code) grandchildAlive=$leakAlive | flood: ${floodKb}KB in ${floodElapsed}s budget=${floodBudget}s bytes=$floodBytes code=$($flood.Code) | stall: elapsed=${stallElapsed}s budget=${stallBudget}s code=$($stall.Code) childAlive=$stallAlive descendantAlive=$stallGrandchildAlive quiesced=$($stall.TreeQuiesced) | logstall: elapsed=${logStallElapsed}s budget=${logStallBudget}s code=$($logstall.Code)"
    if ($problems.Count -gt 0) {
        Write-Host "PIPE-DRAIN SELF-TEST: FAIL $detail -- $($problems -join '; ')"
        exit 1
    }
    Write-Host "PIPE-DRAIN SELF-TEST: PASS $detail"
    exit 0
}

$savedConsoleInputMode = if ($script:ConsoleInput) { [HermesHandoff.ConsoleInput]::DisableQuickEdit() } else { $null }
try {
    # -- 0. The marker was claimed before anything else (top of script) -----
    if ($script:MarkerClaim -eq "refused") {
        # A4: this run changed nothing and owns no result -- the other update
        # (or the Desktop that gave up on this hand-off) reports its own.
        $finalCode = 2
        $blocker = if ($script:MarkerBlocker -gt 0) { " (process $($script:MarkerBlocker))" } else { "" }
        $finalMsg = "Another Hermes update is already running$blocker, or the Desktop gave up on this hand-off. Nothing was changed."
        Write-HandoffLog $finalMsg
        exit $finalCode
    }
    # The previous result is replaced atomically at finish, never deleted here.
    Show-ProgressWindow
    Write-HandoffLog "hand-off start: root=$InstallRoot branch=$Branch channel=$Channel desktopPid=$DesktopPid pid=$PID marker=$($script:MarkerClaim)"

    if ($SelfTestMarker) {
        $finalCode = 0
        $finalMsg = "marker self-test complete"
        exit 0
    }
    Start-MarkerCustodian

    # StartAssigned passes a null CreateProcess currentDirectory, so children
    # inherit the hand-off process directory rather than PowerShell's $PWD.
    # Desktop launches us from HERMES_HOME; pin the process directory to the
    # checkout before any update child can resolve files against the wrong tree.
    try {
        $resolvedInstallRoot = Set-InstallRootCurrentDirectory $InstallRoot
        Write-HandoffLog "process cwd set to install root: $resolvedInstallRoot"
    } catch {
        $finalCode = 3
        $finalMsg = "Update aborted: cannot enter the install root ($InstallRoot). Nothing was changed."
        Write-HandoffLog $finalMsg
        exit $finalCode
    }

    # Exercise the production cwd setup and native launcher without updating.
    if ($SelfTestWorkingDirectory) {
        $expectedRoot = [System.IO.Path]::GetFullPath($InstallRoot)
        $probeExe = Join-Path $PSHOME "powershell.exe"
        $probe = Invoke-HermesStep $probeExe @("-NoProfile", "-Command", "[Environment]::CurrentDirectory; [Console]::IsInputRedirected") "cwd"
        $observed, $stdinRedirected = @($probe.Output.Trim() -split "`r?`n" | ForEach-Object { $_.Trim() })
        if ($probe.Code -ne 0 -or -not [string]::Equals($observed, $expectedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            $finalMsg = "WORKING-DIRECTORY SELF-TEST: FAIL expected=$expectedRoot observed=$observed code=$($probe.Code)"
            Write-Host $finalMsg
            exit 1
        }
        # A step that can read the hand-off console can block on a prompt nobody sees.
        if ($stdinRedirected -ne "True") {
            $finalMsg = "WORKING-DIRECTORY SELF-TEST: FAIL step stdin is an interactive console"
            Write-Host $finalMsg
            exit 1
        }
        $finalCode = 0
        $finalMsg = "WORKING-DIRECTORY SELF-TEST: PASS $observed"
        Write-Host $finalMsg
        exit 0
    }

    $HermesProbeTimeoutSeconds = $ProbeTimeoutSeconds
    . (Join-Path $PSScriptRoot 'runtime.ps1')
    $legacyInstall = -not (Test-Path -LiteralPath (Join-Path $InstallRoot 'pm') -PathType Container)
    try {
        $runtimeCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallRoot)
    } catch {
        $finalCode = 3
        $finalMsg = $_.Exception.Message
        Write-HandoffLog $finalMsg
        exit $finalCode
    }

    # -- 1. Wait for the Desktop to exit (FAIL CLOSED) ----------------------
    Publish-UiProgress "Waiting for Hermes to close"
    if ($DesktopPid -gt 0) {
        # Identity, not pid: the Desktop's creation time was pinned at start.
        if (-not (Wait-DesktopExit $script:DesktopExitSeconds)) {
            # The running Desktop still owns application outputs being replaced.
            $finalCode = 4
            $finalMsg = "Update aborted: the Hermes window (pid $DesktopPid) did not exit within $($script:DesktopExitSeconds)s. Nothing was changed. Close Hermes fully and try again."
            Write-HandoffLog $finalMsg
            exit $finalCode
        }
        Write-HandoffLog "desktop exited"
    }

    # PM creates a new dependency generation. Live old Python readers do not
    # block it; Desktop exit above protects the application output replacement.
    $pythonExe = $runtimeCommand[0]
    $runtimeArgs = @($runtimeCommand | Select-Object -Skip 1)
    # --gateway restarts the local messaging gateway after the update. The
    # Desktop passes -NoGateway when it is served by a remote gateway
    # (#117529): restarting a local one there is never wanted, and with the
    # same channel credentials as the remote host it becomes a competing
    # long-poll consumer (e.g. Telegram rejects one of the two getUpdates
    # callers).
    $gatewayArg = @("--gateway")
    if ($NoGateway) {
        $gatewayArg = @()
        Write-HandoffLog "update requested without --gateway (remote-served Desktop)"
    }
    # --force precedes the target (the hand-off contract test reads the argv in this order).
    $forceArg = @()
    if ($legacyInstall) { $forceArg = @('--force') }
    $updateArgs = $runtimeArgs + @('update', '--yes') + $gatewayArg + $forceArg + $targetArgs
    # --keep-stash: never re-apply local source edits after the update (they
    # stay parked in git stash). Probe --help first: the flag ships with newer
    # backends and an unknown flag would abort argparse with exit 2, which
    # collides with the "close all Hermes windows" sentinel.
    try {
        $helpProbe = Invoke-HermesProbe $pythonExe (@($runtimeArgs) + @('update', '--help'))
        if ($helpProbe.TimedOut) {
            Write-HandoffLog "update --help probe timed out; running without --keep-stash"
        } elseif ($helpProbe.Output -match "--keep-stash") {
            $updateArgs += "--keep-stash"
        } else {
            Write-HandoffLog "installed hermes predates --keep-stash; running without it"
        }
    } catch {
        Write-HandoffLog "could not probe update --help; running without --keep-stash"
    }
    # The update's receipt carries this id (update_receipt._launcher_correlation_id):
    # Get-CommittedReceiptOutcome finds THIS run's receipt by it.
    $script:UpdateCorrelation = if ($env:HERMES_UPDATE_CORRELATION_ID) { $env:HERMES_UPDATE_CORRELATION_ID } else { $script:ResultRunId }
    $env:HERMES_UPDATE_CORRELATION_ID = $script:UpdateCorrelation
    Write-HandoffLog ("running: python " + ($updateArgs -join " "))
    Publish-UiProgress "Updating code and dependencies"
    $res = Invoke-HermesStep $pythonExe $updateArgs "update"
    Write-HandoffLog "hermes update exit code: $($res.Code)"
    $res = Resolve-HermesUpdateOutcome $res

    # Retry only the identified pre-PM update-boundary transition. Current
    # update/build failures propagate and must not trigger another owner.
    if ($legacyInstall -and $res.Code -ne 0 -and $res.Code -ne 2) {
        Write-HandoffLog "legacy update failed; retrying once from the updated installation"
        Publish-UiProgress "Retrying update"
        $runtimeCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallRoot)
        $pythonExe = $runtimeCommand[0]
        $runtimeArgs = @($runtimeCommand | Select-Object -Skip 1)
        # Same request as the first attempt (--force included): the installation is still the
        # legacy one being converted until this run succeeds.
        $updateArgs = $runtimeArgs + @('update', '--yes') + $gatewayArg + $forceArg + $targetArgs
        $res = Invoke-HermesStep $pythonExe $updateArgs 'update'
        $res = Resolve-HermesUpdateOutcome $res
    }

    $committedAfterExit = $null
    if ($res.Code -ne 0 -and $res.Code -ne 2) { $committedAfterExit = Get-CommittedReceiptOutcome }
    if ($res.Code -ne 0 -and -not $committedAfterExit) {
        $finalCode = $res.Code
        $finalMsg = "Update failed (exit $($res.Code)). Run `hermes debug share` in a terminal to send a report."
        exit $finalCode
    }

    # -- Commit point: `hermes update` exited 0 (contract C3). The install is
    # on the new version; every step below is follow-up work whose failure is
    # a warning on an ok result, never "still on the previous version".
    $script:Committed = $true
    $finalCode = 0
    $finalMsg = "Update complete."
    if ($script:UpdateInterrupted) {
        Add-Followup "post-update steps (gateway resume) were interrupted; run hermes update again" "its post-update steps (gateway resume) were interrupted. Run 'hermes update' again to finish them." -Manual
    }
    if ($committedAfterExit) {
        # Past the commit point with a nonzero exit: installed, with an owed follow-up.
        $sentence = switch ($committedAfterExit) {
            'partial' { "one step is left for you: your local source changes may still be parked in git stash. Run 'hermes update' in a terminal for the exact commands." }
            'interrupted' { "its post-update steps were interrupted. The next launch or 'hermes update' finishes them." }
            default { "'hermes update' exited with code $($res.Code) afterwards. Run 'hermes update' in a terminal to finish any remaining steps." }
        }
        Add-Followup "update: hermes update exited $($res.Code) after the commit point (receipt outcome: $committedAfterExit)" $sentence -Manual
    }

    # Pre-PM updates reported a successful exit with a failed build warning.
    # Keep that historical transition here only; current failures propagate.
    $desktopBuildFailed = $false
    if ($legacyInstall -and $res.Output -match "Desktop build failed") {
        Write-HandoffLog "hermes update reported a desktop build failure; retrying build"
        Publish-UiProgress "Rebuilding Desktop"
        $rebuildFailure = $null
        try {
            $runtimeCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallRoot)
            $rebuildArgs = @($runtimeCommand | Select-Object -Skip 1) + @('desktop', '--force-build', '--build-only')
            $rebuild = Invoke-HermesStep $runtimeCommand[0] $rebuildArgs 'rebuild'
            Write-HandoffLog "desktop rebuild exit code: $($rebuild.Code)"
            if ($rebuild.Code -ne 0) { $rebuildFailure = "exit $($rebuild.Code)" }
        } catch {
            $rebuildFailure = $_.Exception.Message
        }
        if ($rebuildFailure) {
            $desktopBuildFailed = $true
            Add-Followup "desktop rebuild: $rebuildFailure" "the Desktop app rebuild failed, so you may still be running the previous build. Run 'hermes desktop --force-build' in a terminal to retry." -Manual
        }
    }

    # Contract C3: a Desktop build that failed after the code committed is an owed follow-up
    # (exit 0); the CLI prints one whole "Desktop app build owed:" line for it. The user is on
    # the new Hermes but this app was not rebuilt: a manual outcome, never plain success.
    if (-not $desktopBuildFailed -and $res.Output -match '(?m)^\s*Desktop app build owed: ') {
        Add-Followup "build: the Desktop app build is owed by the committed update" "the Desktop app could not be rebuilt, so it still runs its old build. Run 'hermes desktop --force-build' in a terminal to rebuild it; the update log has the build error." -Manual
    }

    # Every other owed follow-up of the committed update (a gateway still on the old code, a
    # Windows resume, a lost completion...) prints one whole "Update follow-up '<step>' did not
    # finish:" line (hermes_cli/update_receipt.record_followup): never a plain success either.
    $owedSteps = @([regex]::Matches(($res.Output -join "`n"), "Update follow-up '([A-Za-z0-9_]+)' did not finish: ") |
        ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique)
    if ($owedSteps.Count -gt 0) {
        $owedHint = if ($owedSteps -contains 'gateway_restart') { " Run 'hermes gateway restart' to move the messaging gateway onto the new code now." } else { '' }
        Add-Followup ("followup: owed by the committed update: " + ($owedSteps -join ', ')) ("some follow-up steps did not finish (" + ($owedSteps -join ', ') + "). The next launch or 'hermes update' retries them; the update log has the details." + $owedHint) -Manual
    }

    # A zero-exit update is not proof that the runtime survived the update.
    if (-not $desktopBuildFailed) {
        $verifyFailure = $null
        try {
            $verifyCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallRoot -Module 'hermes_cli.desktop_update_verify')
            $verifyArgs = @($verifyCommand | Select-Object -Skip 1)
            $verify = Invoke-HermesStep $verifyCommand[0] $verifyArgs 'verify'
            if ($verify.Code -ne 0) { $verifyFailure = "exit $($verify.Code)" }
        } catch {
            $verifyFailure = $_.Exception.Message
        }
        if ($verifyFailure) {
            Add-Followup "verify: $verifyFailure" "the new Desktop build could not be verified. Nothing was removed. If Hermes does not start normally, run 'hermes desktop --force-build' in a terminal to rebuild it." -Manual
        }
    }

    # Desktop stopped every locally running profile gateway before handing off
    # so their venv launchers could not hold the update lock. That happens
    # before `hermes update` captures its Windows pause inventory, leaving the
    # updater nothing to resume on its normal success path. Restore the same
    # all-profile fleet after the update -- also when a follow-up above failed:
    # the code is committed and the gateways must not stay down. A remote-served
    # Desktop must stay passive: its -NoGateway hand-off owns no local poller.
    if (-not $NoGateway) {
        $gatewayFailure = $null
        try {
            # Resolve again after update: PM may have published a new generation,
            # and its command can include an isolation/bootstrap prefix.
            $gatewayCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallRoot)
            $gatewayArgs = @($gatewayCommand | Select-Object -Skip 1) + @("gateway", "start", "--all")
            $gatewayRestart = Invoke-HermesStep $gatewayCommand[0] $gatewayArgs "gateway restart"
            if ($gatewayRestart.Code -ne 0) { $gatewayFailure = "exit $($gatewayRestart.Code)" }
        } catch {
            $gatewayFailure = $_.Exception.Message
        }
        if ($gatewayFailure) {
            Add-Followup "gateway start: $gatewayFailure" "it could not restart every messaging gateway. Run `hermes gateway start --all` in a terminal." -Manual
        }
    }

    exit $finalCode
} catch {
    # An unexpected throw. Before the commit point the update did not land
    # (exit 1 below). After it, the install IS updated: report the broken
    # follow-up as a warning on an ok result (contract C3); an unquiesced
    # tree gets its own warning in the finally block.
    Write-HandoffLog "hand-off error: $($_.Exception.Message)"
    if ($script:Committed -and $script:TreeSafeToFinalize) {
        Add-Followup "hand-off: $($_.Exception.Message)" "a post-update step failed unexpectedly. Run 'hermes update' again to finish it." -Manual
        $finalCode = 0
    }
} finally {
    # Truth ordering (sibling contract to posix.sh finish()):
    #   1. durable result + marker removal (the relaunched Desktop consumes
    #      the result on boot and must not park on our marker);
    #   2. attempt the relaunch and require ACCEPTANCE;
    #   3. only then the terminal UI state — done means "Hermes is back",
    #      manual means "it is not, reopen it", error is error (and still
    #      tries to bring the app back after showing itself).
    if ($script:MarkerClaim -eq "refused") {
        # No result (A4). An older Desktop quits right after spawning us: bring
        # it back when it is gone; one that gave up on this run is still open.
        if (-not (Test-DesktopAlive)) { [void](Start-DesktopRelaunch) }
    } elseif (-not $script:TreeSafeToFinalize) {
        # A failed job termination means a mutating descendant may still own
        # checkout/install files. Keep the marker LIVE for as long as a
        # surviving member does and do not relaunch into that state.
        [void](Add-MarkerDelegate $script:UnquiescedPids)
        if ($script:Committed) {
            # C3: the update landed; only a follow-up step outlived its cancellation.
            $finalCode = 0
            Add-Followup "follow-up processes could not be stopped" "a follow-up step's processes could not be stopped, so Hermes was not reopened. Reopen Hermes once they finish, or restart Windows first." -Manual
            $finalMsg = "Hermes was updated, but " + ($script:FollowupText -join " Also, ")
            Write-Result $true $finalCode $finalMsg $true
            Write-HandoffLog $finalMsg
            Show-ManualFinale $finalMsg
        } else {
            $finalCode = 7
            $finalMsg = "Update recovery could not stop every updater process. Hermes was not restarted to avoid overlapping the active install. Wait for it to finish or restart Windows, then reopen Hermes."
            Write-Result $false $finalCode $finalMsg
            Write-HandoffLog $finalMsg
            Show-ErrorFinale $finalMsg
        }
        Close-ProgressWindow
    } else {
        if ($finalCode -eq 0 -and $script:FollowupText.Count -gt 0) {
            $finalMsg = "Hermes was updated, but " + ($script:FollowupText -join " Also, ")
        }
        $manualAction = $finalCode -eq 0 -and $script:ManualFollowup
        Write-Result ($finalCode -eq 0) $finalCode $finalMsg $manualAction
        Invoke-MarkerRelease
        # The R6 wait can last hours, and the Desktop drops a non-manual result
        # whose finished_at is 30 minutes old: publish it again with the real
        # finish time (unless something already consumed it).
        if ($script:MarkerReleaseWaited -and [System.IO.File]::Exists($ResultPath)) {
            Write-Result ($finalCode -eq 0) $finalCode $finalMsg $manualAction
        }
        if ($finalCode -ne 0) {
            Show-ErrorFinale $finalMsg
            Close-ProgressWindow
            if (Test-DesktopAlive) {
                # Exit 4 and friends: the old window never closed. A second
                # instance on top of it is never the fix.
                Write-HandoffLog "desktop pid $DesktopPid is still running; not relaunching a second instance"
            } else {
                [void](Start-DesktopRelaunch)
            }
        } else {
            Publish-UiProgress "Opening Hermes"
            $cameBack = Start-DesktopRelaunch
            if (-not $cameBack -and $RelaunchExe) {
                # Launch was due and did not verifiably land: truthful result
                # for the next boot, manual state held on screen now.
                $finalMsg = "Update complete. Reopen Hermes to finish (it could not restart itself)."
                Write-Result $true 0 $finalMsg $true
                Show-ManualFinale $finalMsg
            }
            Close-ProgressWindow
        }
    }
    if ($null -ne $savedConsoleInputMode) { [HermesHandoff.ConsoleInput]::Restore($savedConsoleInputMode) }
}
exit $finalCode
