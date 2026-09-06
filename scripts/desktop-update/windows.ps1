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
#   cmd /d /s /c start "" /min %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass
#     -File scripts\desktop-update\windows.ps1
#     -InstallRoot <path>   repo checkout (HERMES_HOME\hermes-agent)
#     -Branch <ref>         branch to update against
#     -DesktopPid <pid>     the Electron main process to wait out
#     [-RelaunchAppPath <path>] app checkout passed to a dev Electron relaunch
#     [-RelaunchExe <path>] Hermes.exe to start when done (omit = no relaunch)
#     [-NoUi]               headless (tests); default shows a progress window
#     [-NoMarkerCleanup]    leave .hermes-update-in-progress in place (tests)
#
# SAFETY POSTURE: both preflight gates FAIL CLOSED. A Desktop that never
# exits, or a venv shim that never unlocks, aborts the hand-off without
# mutating the install -- a skipped update is recoverable, a half-updated
# venv is not. Every exit path (success, abort, crash) writes
# .hermes-update-result.json for the relaunched Desktop to surface, and
# relaunches the Desktop so the user is never left stranded.
#
# Marker: we claim HERMES_HOME\.hermes-update-in-progress with OUR pid as
# step 0 (the wrapper cmd.exe pid the Desktop saw is useless -- it exits
# immediately), stamped with OUR kernel creation time as the identity token
# (HERMES_UPDATE_STARTED_AT from the Desktop predates our own process and
# is kept for the log only).
# hermes_cli/update_lock.py's ancestry rule lets our
# `hermes update` child adopt the claim; electron/update-marker.ts parks a
# relaunched Desktop on it. Cleanup only removes the marker while WE still
# own it (a handoff partner that rewrote it keeps its claim).

param(
    [string]$InstallRoot,
    [string]$Branch = "main",
    [int]$DesktopPid = 0,
    [string]$RelaunchExe = "",
    [string]$RelaunchAppPath = "",
    [switch]$NoUi,
    [switch]$NoMarkerCleanup,
    [switch]$SelfTestUi,
    [switch]$SelfTestPipeDrain,
    [switch]$SelfTestMarker,
    [switch]$SelfTestLog,
    [switch]$SelfTestRelaunchCommand
)

if (-not $SelfTestUi -and -not $SelfTestPipeDrain -and -not $SelfTestLog -and -not $SelfTestRelaunchCommand -and -not $InstallRoot) {
    # Mandatory in spirit; relaxed in the signature only so the self-test
    # switches can drive the UI / the pipe drain / the log without a checkout.
    throw "-InstallRoot is required"
}

$ErrorActionPreference = "Continue"
# Foreground helpers: the script is spawned via `cmd start /min`, so its
# WinForms window comes up backgrounded unless we explicitly claim focus --
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
# Render UTF-8 glyphs (checkmarks, arrows) correctly in our own console echo
# too; the legacy conhost default OEM codepage shows them as mojibake.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}
$TempDir = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
$HermesHome = if ($InstallRoot) { Split-Path -Parent $InstallRoot } else { $TempDir }
$MarkerPath = Join-Path $HermesHome ".hermes-update-in-progress"
$LogDir = Join-Path $HermesHome "logs"
$LogPath = Join-Path $LogDir "desktop-update-handoff.log"
$ResultPath = Join-Path $HermesHome ".hermes-update-result.json"
$script:Ui = $null
$script:UiStage = "Hermes will open once done."   # until the first gate; matches ui.html
$script:UiStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$script:MarkerBody = $null

$script:HandoffLogDrops = 0
$script:HandoffLogUtf8 = [System.Text.UTF8Encoding]::new($false)
$script:HandoffLogRetries = 6
$script:HandoffLogRetryMs = 50

function Add-HandoffLogLines([string[]]$Lines) {
    # Append a batch of lines to the hand-off log through our own FileStream,
    # never Add-Content. Two properties of Add-Content lost every line of the
    # 2026-09-05 updates as soon as anything else looked at the log:
    #
    #  * It opens the file with FileShare.Write ONLY (measured: while its
    #    provider writer is held, a Read-access open is refused). Every log
    #    reader -- git-bash `tail -F` / `wc -l`, PowerShell Get-Content -Tail,
    #    any .NET reader -- holds Read access, so the two handles are mutually
    #    exclusive: while a reader has the file Add-Content's open fails with
    #    a sharing violation, and while Add-Content has it the reader fails.
    #    A `tail -F` held for minutes lost everything; a `wc -l` every 15 s
    #    lost whatever coincided (70/500 under a tight polling loop).
    #  * Its IOException is a NON-terminating error. Under this script's
    #    $ErrorActionPreference = "Continue" it printed the red "being used by
    #    another process" block and fell through to `return`, so the try/catch
    #    retry never executed and the drop counter never moved.
    #
    # FileShare.ReadWrite | Delete lets every reader coexist (0/500 under the
    # same poller, 0/300 against a Get-Content -Tail loop where Add-Content
    # lost 287). A .NET constructor exception IS terminating, so the retry
    # below really runs. The floor that remains -- a holder that opened the
    # file with FileShare.Read, i.e. no write sharing at all -- is counted and
    # announced by the next successful write, so the log says it has a gap
    # instead of ending mid-update. Nothing here may throw into the update
    # path: a lost log line is a WARNING, never a failed update.
    if (-not $Lines -or $Lines.Count -eq 0) { return }
    $payload = $Lines
    if ($script:HandoffLogDrops -gt 0) {
        $notice = "{0:yyyy-MM-ddTHH:mm:ssK} WARNING: {1} hand-off log line(s) could not be written while another process held this file; see the console output" -f (Get-Date), $script:HandoffLogDrops
        $payload = @($notice) + $Lines
    }
    $bytes = $script:HandoffLogUtf8.GetBytes(($payload -join "`r`n") + "`r`n")
    for ($attempt = 0; $attempt -lt $script:HandoffLogRetries; $attempt++) {
        try {
            $stream = [System.IO.FileStream]::new($LogPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
            try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
            $script:HandoffLogDrops = 0
            return
        } catch {
            # A sharing violation (IOException) is the retryable case. Anything
            # else (log dir gone, permissions) will not improve in 50 ms, but
            # one bounded pass costs nothing and keeps this path branch-free.
            Start-Sleep -Milliseconds $script:HandoffLogRetryMs
        }
    }
    $script:HandoffLogDrops += $Lines.Count
}

function Write-HandoffLog([string]$Message) {
    $line = "{0:yyyy-MM-ddTHH:mm:ssK} {1}" -f (Get-Date), $Message
    Add-HandoffLogLines @($line)
    Write-Host $line
}

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
    # The OS default browser, read from the UserChoice ProgId that the
    # Windows Settings app writes (https first, http as fallback). Only
    # Chromium-family browsers (ChromeHTML / MSEdgeHTM) support the
    # --app + --user-data-dir combo the shim relies on; any other
    # default browser returns $null and degrades to the WinForms card.
    $progId = $null
    foreach ($proto in @("https", "http")) {
        try {
            $progId = (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\$proto\UserChoice" -Name ProgId -ErrorAction Stop).ProgId
        } catch { continue }
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
    # profile plus any stale hermes-update-ui-* leftovers from interrupted
    # past runs. A browser that is still shutting down may hold the lock, in
    # which case the delete silently no-ops. Safe to sweep by prefix: the
    # update marker (.hermes-update-in-progress) serialises hand-offs, so no
    # other run's profile can be in active use here.
    try {
        $profileDirs = @()
        if ($script:UiServer.Profile) { $profileDirs += $script:UiServer.Profile }
        Get-ChildItem -LiteralPath $TempDir -Directory -Filter "hermes-update-ui-*" -ErrorAction SilentlyContinue |
            ForEach-Object { $profileDirs += $_.FullName }
        foreach ($dir in ($profileDirs | Select-Object -Unique)) {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    } catch {}
    $script:UiServer = $null
}

function Publish-UiEvent([string]$Status, [string]$Message) {
    # Retain a terminal event until the page acknowledges this exact receipt.
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
        # `cmd start /min` spawned us backgrounded, so the card comes up
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
    try {
        $obj = @{
            ok         = $Ok
            exit_code  = $Code
            manual     = $ManualAction
            message    = $Message
            branch     = $Branch
            finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        } | ConvertTo-Json -Compress
        [System.IO.File]::WriteAllText($ResultPath, $obj)
    } catch {}
}

function Claim-UpdateMarker([string]$Body, [string]$ExpectedBody = "") {
    # Production adopts the Desktop's exact claim while holding one exclusive
    # handle, so the marker is never absent and a foreign owner is untouched.
    # The standalone marker self-test exercises the create-only path.
    $stream = $null
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
        if ($ExpectedBody) {
            $stream = [System.IO.File]::Open(
                $MarkerPath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            $reader = New-Object System.IO.StreamReader(
                $stream,
                [System.Text.Encoding]::UTF8,
                $true,
                4096,
                $true
            )
            $current = $reader.ReadToEnd()
            $reader.Dispose()
            if ($current -cne $ExpectedBody) { throw "marker owner claim changed" }
            $stream.SetLength(0)
            $stream.Position = 0
        } else {
            $stream = New-Object System.IO.FileStream(
                $MarkerPath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )
        }
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
        $stream.Dispose()
        $stream = $null
        return [System.IO.File]::ReadAllText($MarkerPath, [System.Text.Encoding]::UTF8) -eq $Body
    } catch {
        if ($stream) { try { $stream.Dispose() } catch {} }
        Write-HandoffLog "update marker claim refused: $($_.Exception.Message)"
        return $false
    }
}

function Write-UpdateReceiptState([DateTime]$SinceUtc) {
    # `hermes update` finalizes logs\update_receipts\latest.json at its
    # command boundary. A run that leaves no receipt has lost its own
    # post-mortem (every 2026-09-05 run did), so record which case this was
    # in the log that survives the console.
    try {
        $latest = Join-Path (Join-Path $LogDir "update_receipts") "latest.json"
        if (-not (Test-Path -LiteralPath $latest -PathType Leaf)) {
            Write-HandoffLog "update receipt: none on disk ($latest)"
            return
        }
        $fi = Get-Item -LiteralPath $latest
        if ($fi.LastWriteTimeUtc -ge $SinceUtc.AddSeconds(-2)) {
            Write-HandoffLog ("update receipt: written {0} ({1} bytes)" -f $latest, $fi.Length)
        } else {
            Write-HandoffLog ("update receipt: NOT written by this run; latest.json dates from {0:u}" -f $fi.LastWriteTimeUtc)
        }
    } catch {
        Write-HandoffLog "update receipt: state unavailable ($($_.Exception.Message))"
    }
}

function Remove-MarkerIfOwned {
    if ($NoMarkerCleanup) { return }
    try {
        if (-not $script:MarkerBody -or -not (Test-Path -LiteralPath $MarkerPath -PathType Leaf)) { return }
        $tombstone = "$MarkerPath.cas-release-$PID-$([Guid]::NewGuid().ToString('N'))"
        [System.IO.File]::Move($MarkerPath, $tombstone)
        $raw = [System.IO.File]::ReadAllText($tombstone, [System.Text.Encoding]::UTF8)
        if ($raw -eq $script:MarkerBody) {
            [System.IO.File]::Delete($tombstone)
            Write-HandoffLog "removed update marker (exact owner claim)"
        } elseif (-not (Test-Path -LiteralPath $MarkerPath)) {
            [System.IO.File]::Move($tombstone, $MarkerPath)
            Write-HandoffLog "leaving update marker: claim changed during cleanup"
        } else {
            Write-HandoffLog "leaving update marker: a newer claim won cleanup"
        }
    } catch {
        try {
            if ($tombstone -and (Test-Path -LiteralPath $tombstone) -and -not (Test-Path -LiteralPath $MarkerPath)) {
                [System.IO.File]::Move($tombstone, $MarkerPath)
            }
        } catch {}
    }
}

function Get-DesktopRelaunchInvocation {
    $arguments = if ($RelaunchAppPath) { @($RelaunchAppPath) } else { @() }
    $suffix = if ($RelaunchAppPath) { ' "{0}"' -f $RelaunchAppPath } else { "" }
    return @{
        Executable = $RelaunchExe
        Arguments = $arguments
        CommandLine = ('"{0}"{1}' -f $RelaunchExe, $suffix)
    }
}

function Start-DesktopRelaunch {
    # Returns $true only when a launch VERIFIABLY happened (WMI accepted and
    # the pid exists, or the fallback spawn returned a live process). The
    # finally block downgrades the on-screen/on-disk outcome when it didn't
    # — the sibling truth contract to posix.sh's launch acceptance.
    if (-not $RelaunchExe) { return $false }
    $relaunch = Get-DesktopRelaunchInvocation
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
            CommandLine      = $relaunch.CommandLine
            CurrentDirectory = $workDir
        } -ErrorAction Stop
        if ($r -and $r.ReturnValue -eq 0) {
            Write-HandoffLog "desktop relaunched detached (pid $($r.ProcessId))"
            $spawned = $true
            # Hand our foreground rights to the new Desktop and focus its
            # main window once it exists. A WMI-spawned process starts
            # unfocused, and Windows only lets the CURRENT foreground
            # owner (us, while the progress window is up / just closed)
            # delegate that right. Poll briefly for the window: Electron
            # takes a couple seconds to create it.
            try {
                if ($script:Win32) {
                    [HermesHandoff.Win32]::AllowSetForegroundWindow([int]$r.ProcessId) | Out-Null
                    $deadline = (Get-Date).AddSeconds(20)
                    while ((Get-Date) -lt $deadline) {
                        $hwnd = [System.IntPtr]::Zero
                        try {
                            $p = Get-Process -Id $r.ProcessId -ErrorAction Stop
                            $hwnd = $p.MainWindowHandle
                        } catch {
                            # Process died before showing a window — that is a
                            # failed launch, not merely an unfocused one.
                            Write-HandoffLog "WARNING: relaunched desktop exited before its window appeared"
                            $spawned = $false
                            break
                        }
                        if ($hwnd -ne [System.IntPtr]::Zero) {
                            [HermesHandoff.Win32]::ShowWindow($hwnd, 9) | Out-Null  # SW_RESTORE
                            [HermesHandoff.Win32]::SetForegroundWindow($hwnd) | Out-Null
                            Write-HandoffLog "focused relaunched desktop window"
                            break
                        }
                        Start-Sleep -Milliseconds 400
                    }
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
            Start-Process -FilePath 'explorer.exe' -ArgumentList $relaunch.CommandLine | Out-Null
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
                        if ($script:Win32) {
                            [HermesHandoff.Win32]::AllowSetForegroundWindow([int]$fresh[0].Id) | Out-Null
                            $focusDeadline = (Get-Date).AddSeconds(20)
                            while ((Get-Date) -lt $focusDeadline) {
                                $hwnd = [System.IntPtr]::Zero
                                try { $hwnd = (Get-Process -Id $fresh[0].Id -ErrorAction Stop).MainWindowHandle } catch { break }
                                if ($hwnd -ne [System.IntPtr]::Zero) {
                                    [HermesHandoff.Win32]::ShowWindow($hwnd, 9) | Out-Null  # SW_RESTORE
                                    [HermesHandoff.Win32]::SetForegroundWindow($hwnd) | Out-Null
                                    Write-HandoffLog "focused relaunched desktop window"
                                    break
                                }
                                Start-Sleep -Milliseconds 400
                            }
                        }
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
            $p = Start-Process -FilePath $RelaunchExe -ArgumentList $relaunch.Arguments -WorkingDirectory (Split-Path -Parent $RelaunchExe) -PassThru
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

# A quiet step is not a silent one to the person watching. `hermes update`
# runs the desktop rebuild for minutes with nothing on its pipes (the build
# streams to logs/update.log), and the hand-off window used to show only the
# "running:" line for that whole stretch. Every StepHeartbeatSeconds of pipe
# silence, Invoke-HermesStep writes one "still running" line with the elapsed
# time and the update log's size, age and last line. 0 disables. Overridable
# for the self-test; not documented as a user knob.
$script:StepHeartbeatSeconds = 30
if ($env:HERMES_UPDATE_STEP_HEARTBEAT_SECONDS) {
    $parsedHeartbeat = -1
    if ([int]::TryParse($env:HERMES_UPDATE_STEP_HEARTBEAT_SECONDS, [ref]$parsedHeartbeat) -and $parsedHeartbeat -ge 0) {
        $script:StepHeartbeatSeconds = $parsedHeartbeat
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

function Get-FileTailLine([string]$Path, [int]$MaxChars = 160) {
    # Last non-empty line of a file another process is appending to. Reads at
    # most the final 4 KiB with FileShare.ReadWrite so the writer is never
    # blocked; $null when the file is absent or unreadable.
    try {
        $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            $take = [Math]::Min(4096, $fs.Length)
            if ($take -le 0) { return $null }
            $fs.Seek(-$take, [System.IO.SeekOrigin]::End) | Out-Null
            $bytes = New-Object byte[] $take
            $read = $fs.Read($bytes, 0, $take)
            $text = [System.Text.Encoding]::UTF8.GetString($bytes, 0, $read)
        } finally { $fs.Dispose() }
        $lines = @($text -split "`r?`n" | Where-Object { $_.Trim() })
        if ($lines.Count -eq 0) { return $null }
        $last = $lines[$lines.Count - 1].Trim()
        if ($last.Length -gt $MaxChars) { $last = $last.Substring(0, $MaxChars) + "..." }
        return $last
    } catch {
        return $null
    }
}

function Write-StepLines($Sink, [int]$Flushed, [string]$Prefix, [switch]$Final) {
    # Log every complete line that arrived in $Sink since index $Flushed and
    # return the new index. A trailing partial line waits for its newline
    # unless -Final. Lines are batched per call (one file append), so a chatty
    # step costs one file append per drain pass, not one per line.
    $length = $Sink.Length
    if ($length -le $Flushed) { return $Flushed }
    $text = $Sink.ToString($Flushed, $length - $Flushed)
    $segment = $text
    if (-not $Final) {
        $end = $text.LastIndexOf("`n")
        if ($end -lt 0) { return $Flushed }
        $segment = $text.Substring(0, $end + 1)
    }
    $stamp = "{0:yyyy-MM-ddTHH:mm:ssK}" -f (Get-Date)
    $lines = @()
    foreach ($ln in ($segment -split "`r?`n")) {
        if ($ln.Trim()) { $lines += ("{0} {1}{2}" -f $stamp, $Prefix, $ln) }
    }
    if ($lines.Count -gt 0) {
        Add-HandoffLogLines $lines
        foreach ($line in $lines) { Write-Host $line }
    }
    return $Flushed + $segment.Length
}

function Get-StepHeartbeatLine([string]$Tag, [int]$StepPid, [datetime]$StartedAt, [datetime]$LastProgressAt) {
    $now = Get-Date
    $elapsed = [int]($now - $StartedAt).TotalSeconds
    $idle = [int]($now - $LastProgressAt).TotalSeconds
    $logNote = "no update log yet"
    try {
        $fi = New-Object System.IO.FileInfo($script:StepProgressLogPath)
        if ($fi.Exists) {
            $age = [int]($now - $fi.LastWriteTime).TotalSeconds
            $tail = Get-FileTailLine $fi.FullName
            $logNote = "update.log {0} bytes, last written {1}s ago" -f $fi.Length, $age
            if ($tail) { $logNote += (": " + $tail) }
        }
    } catch {}
    return ("{0}| still running: {1}s elapsed, no pipe output for {2}s (pid {3}); {4}" -f $Tag, $elapsed, $idle, $StepPid, $logNote)
}

if (-not ("HermesUpdateJob" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.Win32.SafeHandles;

public static class HermesUpdateJob {
    public sealed class StartedProcess {
        public Process Process;
        public StreamReader StandardOutput;
        public StreamReader StandardError;
        public IntPtr Job;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes {
        public int Length;
        public IntPtr SecurityDescriptor;
        public bool InheritHandle;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct StartupInfo {
        public int Size;
        public string Reserved;
        public string Desktop;
        public string Title;
        public int X;
        public int Y;
        public int XSize;
        public int YSize;
        public int XCountChars;
        public int YCountChars;
        public int FillAttribute;
        public int Flags;
        public short ShowWindow;
        public short Reserved2;
        public IntPtr Reserved2Ptr;
        public IntPtr StdInput;
        public IntPtr StdOutput;
        public IntPtr StdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation {
        public IntPtr Process;
        public IntPtr Thread;
        public int ProcessId;
        public int ThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicAccountingInformation {
        public long TotalUserTime;
        public long TotalKernelTime;
        public long ThisPeriodTotalUserTime;
        public long ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount;
        public uint TotalProcesses;
        public uint ActiveProcesses;
        public uint TotalTerminatedProcesses;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CreatePipe(out IntPtr read, out IntPtr write, ref SecurityAttributes attributes, int size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetHandleInformation(IntPtr handle, int mask, int flags);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(
        string applicationName, StringBuilder commandLine,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles,
        int creationFlags, IntPtr environment, string currentDirectory,
        ref StartupInfo startupInfo, out ProcessInformation processInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, uint exitCode);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetStdHandle(int standardHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool QueryInformationJobObject(
        IntPtr job,
        int informationClass,
        out BasicAccountingInformation information,
        uint informationLength,
        IntPtr returnLength
    );

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    public static StartedProcess StartAssigned(string executable, string arguments) {
        IntPtr job = IntPtr.Zero;
        IntPtr outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
        IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero;
        ProcessInformation pi = new ProcessInformation();
        try {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new InvalidOperationException("CreateJobObject failed");
            SecurityAttributes sa = new SecurityAttributes();
            sa.Length = Marshal.SizeOf(typeof(SecurityAttributes));
            sa.InheritHandle = true;
            if (!CreatePipe(out outRead, out outWrite, ref sa, 0) ||
                !CreatePipe(out errRead, out errWrite, ref sa, 0))
                throw new InvalidOperationException("CreatePipe failed");
            if (!SetHandleInformation(outRead, 1, 0) || !SetHandleInformation(errRead, 1, 0))
                throw new InvalidOperationException("SetHandleInformation failed");

            StartupInfo si = new StartupInfo();
            si.Size = Marshal.SizeOf(typeof(StartupInfo));
            si.Flags = 0x00000100; // STARTF_USESTDHANDLES
            si.StdInput = GetStdHandle(-10);
            si.StdOutput = outWrite;
            si.StdError = errWrite;
            StringBuilder commandLine = new StringBuilder("\"" + executable + "\" " + arguments);
            if (!CreateProcess(executable, commandLine, IntPtr.Zero, IntPtr.Zero, true,
                    0x00000004 | 0x08000000, IntPtr.Zero, null, ref si, out pi))
                throw new InvalidOperationException("CreateProcess failed");
            if (!AssignProcessToJobObject(job, pi.Process)) {
                TerminateProcess(pi.Process, 1);
                throw new InvalidOperationException("AssignProcessToJobObject failed");
            }

            Process process = Process.GetProcessById(pi.ProcessId);
            // Force Process to open its own stable query handle before the raw
            // CreateProcess handle is closed; PS 5.1 otherwise reports a null
            // ExitCode after fast children have already disappeared.
            IntPtr stableProcessHandle = process.Handle;
            StreamReader stdout = new StreamReader(new FileStream(
                new SafeFileHandle(outRead, true), FileAccess.Read, 4096, false), Encoding.UTF8);
            StreamReader stderr = new StreamReader(new FileStream(
                new SafeFileHandle(errRead, true), FileAccess.Read, 4096, false), Encoding.UTF8);
            outRead = IntPtr.Zero;
            errRead = IntPtr.Zero;
            CloseHandle(outWrite); outWrite = IntPtr.Zero;
            CloseHandle(errWrite); errWrite = IntPtr.Zero;
            if (ResumeThread(pi.Thread) == 0xffffffff)
                throw new InvalidOperationException("ResumeThread failed");
            return new StartedProcess { Process = process, StandardOutput = stdout, StandardError = stderr, Job = job };
        } catch {
            if (pi.Process != IntPtr.Zero) TerminateProcess(pi.Process, 1);
            if (job != IntPtr.Zero) CloseHandle(job);
            throw;
        } finally {
            if (pi.Thread != IntPtr.Zero) CloseHandle(pi.Thread);
            if (pi.Process != IntPtr.Zero) CloseHandle(pi.Process);
            if (outRead != IntPtr.Zero) CloseHandle(outRead);
            if (outWrite != IntPtr.Zero) CloseHandle(outWrite);
            if (errRead != IntPtr.Zero) CloseHandle(errRead);
            if (errWrite != IntPtr.Zero) CloseHandle(errWrite);
        }
    }

    public static bool TerminateAndWait(IntPtr job, uint exitCode, int timeoutMs) {
        if (job == IntPtr.Zero || !TerminateJobObject(job, exitCode)) return false;
        Stopwatch clock = Stopwatch.StartNew();
        BasicAccountingInformation information;
        do {
            if (!QueryInformationJobObject(
                    job, 1, out information,
                    (uint)Marshal.SizeOf(typeof(BasicAccountingInformation)),
                    IntPtr.Zero)) return false;
            if (information.ActiveProcesses == 0) return true;
            Thread.Sleep(50);
        } while (clock.ElapsedMilliseconds < timeoutMs);
        return false;
    }

    public static void Close(IntPtr job) {
        if (job != IntPtr.Zero) CloseHandle(job);
    }
}
'@
}

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
    # Both pipes drain asynchronously (no deadlock however chatty the child)
    # while a small DoEvents loop keeps the marquee animating through long
    # silent stretches (pip installs) -- the old EndOfStream pump blocked on
    # quiet children and froze it. Complete lines are written to the hand-off
    # log as they arrive (Write-StepLines), so the console/window and
    # `hermes debug share` see the step's progress live instead of a dump after
    # it exits; a "still running" heartbeat covers pipe-silent stretches. The
    # progress window's stage stays orchestrator-owned: child output is
    # logged, never parsed into it.
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
    $stepStartedAt = $lastProgressAt
    $lastHeartbeatAt = $lastProgressAt
    $outFlushed = 0
    $errFlushed = 0
    $outPrefix = "{0}| " -f $Tag
    $errPrefix = "{0}!| " -f $Tag
    $progressLogStamp = Get-StepProgressLogStamp
    $stalled = $false
    while ($true) {
        $moved = $false
        $outDone = Step-PipeDrain $stdoutReader ([ref]$outTask) $outBuffer $outSink ([ref]$moved)
        $errDone = Step-PipeDrain $stderrReader ([ref]$errTask) $errBuffer $errSink ([ref]$moved)
        if ($moved) {
            $lastProgressAt = Get-Date
            $lastHeartbeatAt = $lastProgressAt
            $outFlushed = Write-StepLines $outSink $outFlushed $outPrefix
            $errFlushed = Write-StepLines $errSink $errFlushed $errPrefix
        } elseif ($script:StepHeartbeatSeconds -gt 0 -and -not $proc.HasExited -and ((Get-Date) - $lastHeartbeatAt).TotalSeconds -ge $script:StepHeartbeatSeconds) {
            $lastHeartbeatAt = Get-Date
            Write-HandoffLog (Get-StepHeartbeatLine $Tag $proc.Id $stepStartedAt $lastProgressAt)
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
                Write-HandoffLog ("{0}!| step stalled: no stdout/stderr for {1}s and no update.log growth while pid {2} remained alive; cancelling its process tree." -f $Tag, $script:StepIdleTimeoutSeconds, $proc.Id)
                $stalled = [HermesUpdateJob]::TerminateAndWait($job, 124, 10000)
                if (-not $stalled) {
                    Write-HandoffLog ("{0}!| process-tree cancellation could not prove quiescence; refusing the timeout retry." -f $Tag)
                    $script:TreeSafeToFinalize = $false
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
    }
    # Bounded overload deliberately: the argument-less overload also waits on
    # redirected streams, which is the very wait we just bounded. HasExited is
    # already true here, so this call only settles ExitCode.
    [void]$proc.WaitForExit(5000)
    if ($abandoned) {
        Write-HandoffLog ("{0}!| pipe drain abandoned after {1}s: '{0}' exited but a surviving descendant still holds its stdout/stderr handles. Continuing the hand-off with the output captured so far (#90455)." -f $Tag, $script:StepDrainGraceSeconds)
    }
    # Everything that arrived was logged live; only a trailing partial line
    # (no newline before the pipe closed) is still pending.
    [void](Write-StepLines $outSink $outFlushed $outPrefix -Final)
    [void](Write-StepLines $errSink $errFlushed $errPrefix -Final)
    $outText = $outSink.ToString()
    $errText = $errSink.ToString()
    $all = $outText
    if ($errText) { $all += "`n" + $errText }
    $code = if ($stalled) { 124 } else { $proc.ExitCode }
    [HermesUpdateJob]::Close($job)
    return @{ Code = $code; Output = $all; TreeQuiesced = (-not $stalled -or $proc.HasExited); StartedAfterJobAssignment = $true }
}

$finalCode = 1
$finalMsg = "update did not complete"
$script:TreeSafeToFinalize = $true

if ($SelfTestRelaunchCommand) {
    Get-DesktopRelaunchInvocation | ConvertTo-Json -Compress
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
#   livelog -- a step whose first line must reach the hand-off log while the
#            step is still running, and whose quiet stretch must produce a
#            "still running" heartbeat. The child itself watches the log for
#            its own first line and encodes the answer in its exit code, so
#            the proof is not a timing coincidence.
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
    $liveLogPs1 = Join-Path $TempDir "hermes-step-livelog-$stamp.ps1"
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
[System.IO.File]::WriteAllText($PidFile, [string]$grandchild.Id)
Write-Output "pipe-drain step output"
[Console]::Out.Flush()
exit 7
'@
    # Writes straight to the console stream, holding nothing: a step that is
    # merely loud. `hermes update` is this shape -- the Electron/vite build
    # alone is megabytes. Few large lines rather than many small ones on
    # purpose: Write-HandoffLog is one file append per line and runs inside the
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
    # Prints a line, then polls the hand-off log for that line while it is
    # still alive: exit 9 = the line was streamed before the step exited,
    # exit 10 = it was not (the pre-streaming behaviour: dump after exit).
    $liveLogSource = @'
param([int]$WaitSeconds, [string]$HandoffLog)
Write-Output "live line one"
[Console]::Out.Flush()
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$seenAt = $null
# Single-line loop on purpose: a closing brace in column 0 inside this
# here-string would end the -SelfTest block early for the source-contract tests.
while ((Get-Date) -lt $deadline) { if ($null -eq $seenAt) { try { if ((Get-Content -LiteralPath $HandoffLog -Raw -ErrorAction Stop) -match "livelog\| live line one") { $seenAt = Get-Date } } catch {} } elseif (((Get-Date) - $seenAt).TotalSeconds -ge 5) { break }; Start-Sleep -Milliseconds 250 }
Write-Output "live line two"
[Console]::Out.Flush()
if ($null -ne $seenAt) { exit 9 } else { exit 10 }
'@
    [System.IO.File]::WriteAllText($childPs1, $childSource)
    [System.IO.File]::WriteAllText($floodPs1, $floodSource)
    [System.IO.File]::WriteAllText($stallPs1, $stallSource)
    [System.IO.File]::WriteAllText($logStallPs1, $logStallSource)
    [System.IO.File]::WriteAllText($liveLogPs1, $liveLogSource)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $res = Invoke-HermesStep $powershell @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $childPs1,
        "-Hold", [string]$hold, "-PidFile", $pidFile
    ) "pipedrain"
    $sw.Stop()
    $elapsed = [Math]::Round($sw.Elapsed.TotalSeconds, 2)

    $leakPid = 0
    if (Test-Path -LiteralPath $pidFile) {
        [void][int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$leakPid)
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

    # livelog arm: a 2s heartbeat so the quiet stretch (the child only polls
    # the log) is short; the child keeps waiting 5s after it sees its line so
    # at least one heartbeat lands before it exits.
    # The child is pipe-silent while it polls, which the idle watchdog (set to
    # seconds by the pytest caller) would read as a stall: lift the ceiling
    # for exactly this arm, the way logstall swaps in its own progress log.
    $liveWait = 12
    $savedHeartbeat = $script:StepHeartbeatSeconds
    $savedIdle = $script:StepIdleTimeoutSeconds
    $script:StepHeartbeatSeconds = 2
    $script:StepIdleTimeoutSeconds = $liveWait + 30
    $liveLogSw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $livelog = Invoke-HermesStep $powershell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $liveLogPs1,
            "-WaitSeconds", [string]$liveWait, "-HandoffLog", $LogPath
        ) "livelog"
    } finally {
        $script:StepHeartbeatSeconds = $savedHeartbeat
        $script:StepIdleTimeoutSeconds = $savedIdle
    }
    $liveLogSw.Stop()
    $liveLogElapsed = [Math]::Round($liveLogSw.Elapsed.TotalSeconds, 2)
    $liveLogText = ""
    try { $liveLogText = Get-Content -LiteralPath $LogPath -Raw -ErrorAction Stop } catch {}
    $liveHeartbeatSeen = $liveLogText -match "livelog\| still running: "
    $liveSecondLineLogged = $liveLogText -match "livelog\| live line two"

    Remove-Item -LiteralPath $childPs1, $floodPs1, $stallPs1, $logStallPs1, $liveLogPs1, $pidFile, $stallPidFile, $stallGrandchildPidFile, $logStallProgress -Force -ErrorAction SilentlyContinue

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
    $liveLogBudget = $liveWait + 30
    if ($livelog.Code -ne 9) { $problems += "livelog arm exit code $($livelog.Code), expected 9 -- the step's first line did not reach the hand-off log while the step was still running (output is being dumped after exit again)" }
    if (-not $liveHeartbeatSeen) { $problems += "livelog arm produced no 'still running' heartbeat during its quiet stretch" }
    if (-not $liveSecondLineLogged) { $problems += "livelog arm's final line was not logged" }
    if ($livelog.Output -notmatch "live line one" -or $livelog.Output -notmatch "live line two") { $problems += "livelog arm step output was lost from the returned Output" }
    if ($liveLogElapsed -ge $liveLogBudget) { $problems += "livelog arm returned in ${liveLogElapsed}s, over the ${liveLogBudget}s budget" }

    $detail = "leak: elapsed=${elapsed}s budget=${budget}s code=$($res.Code) grandchildAlive=$leakAlive | flood: ${floodKb}KB in ${floodElapsed}s budget=${floodBudget}s bytes=$floodBytes code=$($flood.Code) | stall: elapsed=${stallElapsed}s budget=${stallBudget}s code=$($stall.Code) childAlive=$stallAlive descendantAlive=$stallGrandchildAlive quiesced=$($stall.TreeQuiesced) | logstall: elapsed=${logStallElapsed}s budget=${logStallBudget}s code=$($logstall.Code) | livelog: elapsed=${liveLogElapsed}s budget=${liveLogBudget}s code=$($livelog.Code) heartbeat=$liveHeartbeatSeen"
    if ($problems.Count -gt 0) {
        Write-Host "PIPE-DRAIN SELF-TEST: FAIL $detail -- $($problems -join '; ')"
        exit 1
    }
    Write-Host "PIPE-DRAIN SELF-TEST: PASS $detail"
    exit 0
}

# -SelfTestLog: prove the hand-off log survives a concurrent reader ---------
# The 2026-09-05 log loss needs no update to reproduce, only a second handle
# on desktop-update-handoff.log. Two arms, both through the real
# Write-HandoffLog / Add-HandoffLogLines:
#
#   shared  -- a reader holds the file for the whole arm with Read access and
#              FileShare.ReadWrite, the handle every log viewer holds
#              (git-bash tail -F / wc -l, Get-Content -Tail, .NET readers).
#              Every line must land and the drop counter must stay 0. This is
#              the arm Add-Content failed: its FileShare.Write open is refused
#              while any Read-access handle exists.
#   blocked -- a holder that shares only Read, so no writer can open the file
#              at all: the floor. The lines must be COUNTED, nothing may throw
#              out of the logging path, and the first write after the holder
#              lets go must announce the gap with a WARNING line.
# Exits before any marker/desktop machinery, same as -SelfTestUi; touches
# nothing but the log under $TempDir (no -InstallRoot => $HermesHome = TEMP).
if ($SelfTestLog) {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    $logExisted = Test-Path -LiteralPath $LogPath
    $problems = @()
    $sharedLines = 50
    Write-HandoffLog "SELF-TEST: log sharing (no update will run)"

    $reader = $null
    try {
        $reader = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        for ($i = 1; $i -le $sharedLines; $i++) { Write-HandoffLog "SELF-TEST: shared-reader line $i" }
        $stamp = "{0:yyyy-MM-ddTHH:mm:ssK}" -f (Get-Date)
        Add-HandoffLogLines @("$stamp SELF-TEST: shared-reader batch a", "$stamp SELF-TEST: shared-reader batch b")
    } catch {
        $problems += "shared arm threw out of the logging path: $($_.Exception.Message)"
    } finally {
        if ($reader) { $reader.Dispose() }
    }
    $sharedDrops = $script:HandoffLogDrops

    $blockedLines = 3
    $holder = $null
    try {
        $holder = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
        for ($i = 1; $i -le $blockedLines; $i++) { Write-HandoffLog "SELF-TEST: blocked line $i" }
    } catch {
        $problems += "blocked arm threw out of the logging path: $($_.Exception.Message)"
    } finally {
        if ($holder) { $holder.Dispose() }
    }
    $blockedDrops = $script:HandoffLogDrops
    Write-HandoffLog "SELF-TEST: after release"
    $finalDrops = $script:HandoffLogDrops

    $text = ""
    try { $text = [System.IO.File]::ReadAllText($LogPath) } catch { $problems += "could not read the log back: $($_.Exception.Message)" }
    $sharedSeen = ([regex]::Matches($text, "SELF-TEST: shared-reader line \d+\r\n")).Count
    $batchSeen = ([regex]::Matches($text, "SELF-TEST: shared-reader batch [ab]\r\n")).Count
    $blockedSeen = ([regex]::Matches($text, "SELF-TEST: blocked line \d+")).Count
    $warningSeen = $text -match "WARNING: $blockedLines hand-off log line\(s\) could not be written"
    $afterSeen = $text -match "SELF-TEST: after release"
    $stampOk = $text -match "(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} SELF-TEST: shared-reader line 1\r$"
    $bom = $false
    if (-not $logExisted) {
        try { $head = [System.IO.File]::ReadAllBytes($LogPath); $bom = ($head.Length -ge 3 -and $head[0] -eq 0xEF -and $head[1] -eq 0xBB -and $head[2] -eq 0xBF) } catch {}
    }

    if ($sharedSeen -ne $sharedLines) { $problems += "shared arm: $sharedSeen of $sharedLines lines reached the log while a Read/ReadWrite reader held it (an Add-Content FileShare.Write open loses all of them)" }
    if ($batchSeen -ne 2) { $problems += "shared arm: batched Add-HandoffLogLines wrote $batchSeen of 2 lines" }
    if ($sharedDrops -ne 0) { $problems += "shared arm: drop counter is $sharedDrops, expected 0" }
    if (-not $stampOk) { $problems += "line format changed: expected '{yyyy-MM-ddTHH:mm:ssK} message' with CRLF" }
    if ($blockedSeen -ne 0) { $problems += "blocked arm: $blockedSeen line(s) landed while a FileShare.Read holder had the file (the holder did not block writers; fixture invalid)" }
    if ($blockedDrops -ne $blockedLines) { $problems += "blocked arm: drop counter is $blockedDrops, expected $blockedLines (drops are not being counted -- the retry catch is not running)" }
    if (-not $warningSeen) { $problems += "blocked arm: the first write after release did not announce the $blockedLines dropped line(s) with a WARNING" }
    if (-not $afterSeen) { $problems += "blocked arm: the write after release was lost" }
    if ($finalDrops -ne 0) { $problems += "blocked arm: drop counter is $finalDrops after the announcing write, expected 0" }
    if ($bom) { $problems += "a fresh log was created with a UTF-8 BOM" }

    $detail = "shared: lines=$sharedSeen/$sharedLines batch=$batchSeen/2 drops=$sharedDrops | blocked: landed=$blockedSeen counted=$blockedDrops warning=$warningSeen afterRelease=$afterSeen finalDrops=$finalDrops | bom=$bom"
    if ($problems.Count -gt 0) {
        Write-Host "LOG SELF-TEST: FAIL $detail -- $($problems -join '; ')"
        exit 1
    }
    Write-Host "LOG SELF-TEST: PASS $detail"
    exit 0
}

try {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    Remove-Item -LiteralPath $ResultPath -Force -ErrorAction SilentlyContinue
    Show-ProgressWindow
    Write-HandoffLog "hand-off start: root=$InstallRoot branch=$Branch desktopPid=$DesktopPid pid=$PID"

    # -- 0. Claim the update marker with OUR pid ---------------------------
    try {
        # The marker contract (Rust/TS/Python readers) is "<pid>\n<ts>\n".
        # <ts> is an IDENTITY TOKEN, not a clock to compare: readers prove
        # <pid> by fetching its kernel creation time and requiring it to be
        # no later than <ts> (update-marker.ts probePidIdentity evaluates
        # exactly the expression below for the pid it reads). Writing our own
        # creation time makes that check exact for the life of this process
        # and false for any later reuse of our pid. The Desktop's
        # HERMES_UPDATE_STARTED_AT is stamped before it spawns us, so it can
        # only predate our creation; on 2026-09-06 a spawn across a
        # one-second boundary (release :06.977Z, process born in :07) made
        # the Desktop judge its own updater's claim stale and abort with
        # "did not acknowledge the protected handoff". Wall clock is only
        # the fallback when the creation time cannot be read.
        $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        try {
            $startedAt = [int64][DateTimeOffset]::new((Get-Process -Id $PID -ErrorAction Stop).StartTime.ToUniversalTime()).ToUnixTimeSeconds()
        } catch {
            $startedAt = 0L
        }
        if ($startedAt -le 0 -or $startedAt -gt ($epoch + 5)) { $startedAt = $epoch }
        $desktopStartedAt = 0L
        if (-not [int64]::TryParse($env:HERMES_UPDATE_STARTED_AT, [ref]$desktopStartedAt) -or $desktopStartedAt -le 0) {
            $desktopStartedAt = 0L
        }
        $markerBody = "$PID`n$startedAt`n"
        $createSelfTestMarker = $SelfTestMarker -and $DesktopPid -le 0
        $expectedMarkerBody = if ($createSelfTestMarker) { "" } else { "$DesktopPid`n$desktopStartedAt`n" }
        if (-not $createSelfTestMarker -and ($DesktopPid -le 0 -or $desktopStartedAt -le 0)) {
            throw "desktop marker identity is missing"
        }
        if (-not (Claim-UpdateMarker $markerBody $expectedMarkerBody)) {
            throw "desktop update marker was absent, changed, or could not be adopted"
        }
        $script:MarkerBody = $markerBody
        Write-HandoffLog "claimed update marker (pid $PID, created $startedAt; desktop hand-off started at $desktopStartedAt)"
    } catch {
        $finalCode = 8
        $finalMsg = "Update aborted: could not claim the authenticated update marker. Nothing was changed."
        Write-HandoffLog "$finalMsg $($_.Exception.Message)"
        exit $finalCode
    }

    if ($SelfTestMarker) {
        $finalCode = 0
        $finalMsg = "marker self-test complete"
        exit 0
    }

    # Check only the interpreter here: dependency recovery belongs to update.
    $pythonExe = Join-Path $InstallRoot "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        $finalCode = 3
        $finalMsg = "Update aborted: $pythonExe is missing. Repair the installation and review antivirus quarantine before retrying."
        Write-HandoffLog $finalMsg
        exit $finalCode
    }

    # -- 1. Wait for the Desktop to exit (FAIL CLOSED) ----------------------
    Publish-UiProgress "Waiting for Hermes to close"
    if ($DesktopPid -gt 0) {
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $proc = Get-Process -Id $DesktopPid -ErrorAction SilentlyContinue
            if (-not $proc) { break }
            Start-Sleep -Milliseconds 300
            if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
        }
        if (Get-Process -Id $DesktopPid -ErrorAction SilentlyContinue) {
            # A live Desktop means a live backend re-locking the venv at any
            # moment. Updating under it is how installs brick. Abort.
            $finalCode = 4
            $finalMsg = "Update aborted: the Hermes window (pid $DesktopPid) did not exit within 30s. Nothing was changed. Close Hermes fully and try again."
            Write-HandoffLog $finalMsg
            exit $finalCode
        }
        Write-HandoffLog "desktop exited"
    }

    # -- 2. Wait for the venv shim to unlock (FAIL CLOSED) ------------------
    Publish-UiProgress "Preparing Hermes files"
    $shim = Join-Path $InstallRoot "venv\Scripts\hermes.exe"
    if (Test-Path -LiteralPath $shim) {
        $unlocked = $false
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline) {
            try {
                $fs = [System.IO.File]::Open($shim, 'Open', 'ReadWrite', 'None')
                $fs.Close()
                $unlocked = $true
                break
            } catch {
                Start-Sleep -Milliseconds 400
                if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
            }
        }
        if (-not $unlocked) {
            # Something still maps the venv. --force-ing past it guarantees a
            # half-updated venv (the exact 2026-08-09 Access-denied brick).
            $finalCode = 5
            $finalMsg = "Update aborted: another process is still holding the Hermes install open (venv\Scripts\hermes.exe locked after 20s). Nothing was changed. Close other Hermes windows/terminals and try again."
            Write-HandoffLog $finalMsg
            exit $finalCode
        }
        Write-HandoffLog "venv shim unlocked"
    }

    # -- 3. Run the update from the CURRENT checkout ------------------------
    # --force skips only the hermes.exe shim guard, which step 2 just PROVED
    # is unlocked; the venv-python holder guard (orphan reap included) stays
    # active. Our marker claim is adopted by the child via update_lock.py's
    # process-ancestry rule.
    #
    # DRIVE THE UPDATE THROUGH venv\Scripts\python.exe, NOT venv\Scripts\hermes.exe.
    # `uv pip install -e .` has to replace the console-script shims, so
    # _quarantine_running_hermes_exe must first rename the running hermes.exe
    # out of the way. On Windows that rename fails whenever ANY child process
    # spawned from that hermes.exe is still alive: a child inherits a handle on
    # the parent image, and the resulting sharing violation is indistinguishable
    # from a user leaving a second Hermes window open. It is the inherited
    # handle, not the trampoline itself, that pins the file -- killing the child
    # makes the same rename succeed immediately, and the shim flavour (uv
    # trampoline vs distlib launcher) makes no difference.
    #
    # The updater reliably spawns such children itself (npx cache warm, memory
    # provider refresh -- hindsight-api runs as a daemon with --idle-timeout
    # 300 and outlives the step that started it), so this is a race, not a
    # deterministic failure: the same hand-off succeeds on one run and dies on
    # the next. Step 2's preflight cannot catch it, because the shim genuinely
    # IS unlocked at that moment.
    #
    # When the rename loses that race there is no recovery: `uv pip install -e .`
    # exits 2 and the ZIP fallback repeats the identical sequence, so the desktop
    # build stage is never reached and apps/desktop/release is left missing -- an
    # install whose Start Menu shortcut points at a Hermes.exe that no longer
    # exists. (A reboot-deferred rename was the old last resort here; it needed
    # elevation a Desktop-driven update does not have, and freed nothing for the
    # install already in flight.)
    #
    # Running the same code as `python.exe -m hermes_cli.main update` puts the
    # inherited handles on python.exe, which uv never has to replace.
    #
    # posix.sh is deliberately left alone: unlinking a running executable is
    # legal there, so the equivalent call is harmless.
    $pythonExe = Join-Path $InstallRoot "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $finalCode = 3
        $finalMsg = "Update aborted: $pythonExe is missing. The install needs repair (run the Hermes installer or `hermes doctor`)."
        Write-HandoffLog $finalMsg
        exit $finalCode
    }
    $updateArgs = @("-m", "hermes_cli.main", "update", "--yes", "--gateway", "--force", "--branch", $Branch)
    # --keep-stash: never re-apply local source edits after the update (they
    # stay parked in git stash). Probe --help first: the flag ships with newer
    # backends and an unknown flag would abort argparse with exit 2, which
    # collides with the "close all Hermes windows" sentinel.
    try {
        $updateHelp = & $pythonExe -m hermes_cli.main update --help 2>$null | Out-String
        if ($updateHelp -match "--keep-stash") {
            $updateArgs += "--keep-stash"
        } else {
            Write-HandoffLog "installed hermes predates --keep-stash; running without it"
        }
    } catch {
        Write-HandoffLog "could not probe update --help; running without --keep-stash"
    }
    Write-HandoffLog ("running: python " + ($updateArgs -join " "))
    Publish-UiProgress "Updating code and dependencies"
    $updateStepStartedUtc = [DateTime]::UtcNow
    $res = Invoke-HermesStep $pythonExe $updateArgs "update"
    Write-HandoffLog "hermes update exit code: $($res.Code)"

    $retryPolicyPath = Join-Path $PSScriptRoot "retry-policy.ps1"
    if (Test-Path -LiteralPath $retryPolicyPath) {
        . $retryPolicyPath
        $shouldRetry = Test-HermesUpdateShouldRetry -ExitCode $res.Code -InstallRoot $InstallRoot
    } else {
        # The child may have swapped to a checkout without the companion policy
        # while this older script is still running in memory. Preserve the
        # previous fail-closed behavior instead of calling an undefined function.
        Write-HandoffLog "retry policy is unavailable after checkout swap; using legacy retry rules"
        $shouldRetry = $res.Code -ne 0 -and $res.Code -ne 2
    }
    if ($shouldRetry) {
        # One retry for update-boundary failures. Most exit-2 safety refusals
        # remain terminal, but self-lock deferral also uses exit 2 and writes
        # .update-incomplete after the code swap. That marker is only a retry
        # signal here: the fresh process's early-recovery pass finishes core
        # dependency sync before native modules load, then `update` continues
        # the remaining Desktop/skills stages of the full pipeline.
        Write-HandoffLog "first attempt left retryable update state; retrying once in a fresh process"
        Publish-UiProgress "Retrying update"
        $res = Invoke-HermesStep $pythonExe $updateArgs "update"
        Write-HandoffLog "retry exit code: $($res.Code)"
    }
    Write-UpdateReceiptState $updateStepStartedUtc

    # -- 4. Truthful completion: don't trust exit 0 -------------------------
    # `hermes update` treats a Desktop GUI build failure as NON-fatal (prints
    # a one-line warning, exits 0). For a Desktop-DRIVEN update that warning
    # is fatal: we would relaunch the old exe and call it success. Detect it,
    # retry the build once, and propagate honestly.
    $desktopBuildFailed = $false
    if ($res.Code -eq 0 -and $res.Output -match "Desktop build failed") {
        Write-HandoffLog "hermes update reported a desktop build failure (non-fatal there, fatal here); retrying build"
        Publish-UiProgress "Rebuilding Desktop"
        $rebuild = Invoke-HermesStep $pythonExe @("-m", "hermes_cli.main", "desktop", "--force-build", "--build-only") "rebuild"
        Write-HandoffLog "desktop rebuild exit code: $($rebuild.Code)"
        if ($rebuild.Code -ne 0) { $desktopBuildFailed = $true }
    }

    # A zero-exit update is not proof that the runtime survived the update.
    if ($res.Code -eq 0 -and -not $desktopBuildFailed) {
        $verifyCode = "import hermes_cli.main; from pathlib import Path; from hermes_cli.desktop_update_verify import verify_windows_desktop_update; verify_windows_desktop_update(Path.cwd())"
        $verify = Invoke-HermesStep $pythonExe @("-c", $verifyCode) "verify"
        if ($verify.Code -ne 0) {
            $finalCode = 8
            $finalMsg = "The updated Hermes runtime or Desktop build failed verification. Repair the installation and review antivirus quarantine before retrying."
            Write-HandoffLog $finalMsg
            exit $finalCode
        }
    }

    if ($res.Code -eq 0 -and -not $desktopBuildFailed) {
        $finalCode = 0
        $finalMsg = "Update complete."
    } elseif ($desktopBuildFailed) {
        $finalCode = 6
        $finalMsg = "Code and dependencies updated, but the Desktop app REBUILD FAILED - you are running the previous build. Run `hermes desktop --force-build` from a terminal to retry."
    } else {
        $finalCode = $res.Code
        $finalMsg = "Update failed (exit $($res.Code)). Run `hermes debug share` in a terminal to send a report."
    }
    exit $finalCode
} finally {
    # Truth ordering (sibling contract to posix.sh finish()):
    #   1. durable result + marker removal (the relaunched Desktop consumes
    #      the result on boot and must not park on our marker);
    #   2. attempt the relaunch and require ACCEPTANCE;
    #   3. only then the terminal UI state — done means "Hermes is back",
    #      manual means "it is not, reopen it", error is error (and still
    #      tries to bring the app back after showing itself).
    if (-not $script:TreeSafeToFinalize) {
        # A failed job termination means a mutating descendant may still own
        # checkout/install files. Preserve the marker and do not relaunch into
        # that unknown state. This is intentionally fail-closed; the marker's
        # dead-owner recovery remains the next-start escape hatch.
        $finalCode = 7
        $finalMsg = "Update recovery could not stop every updater process. Hermes was not restarted to avoid overlapping the active install. Wait for it to finish or restart Windows, then reopen Hermes."
        Write-Result $false $finalCode $finalMsg
        Write-HandoffLog $finalMsg
        Show-ErrorFinale $finalMsg
        Close-ProgressWindow
    } else {
        Write-Result ($finalCode -eq 0) $finalCode $finalMsg
        Remove-MarkerIfOwned
        if ($finalCode -ne 0) {
            Show-ErrorFinale $finalMsg
            Close-ProgressWindow
            [void](Start-DesktopRelaunch)
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
}
