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
#   cmd /d /s /c start "" /min powershell -NoProfile -ExecutionPolicy Bypass
#     -File scripts\desktop-update\windows.ps1
#     -InstallRoot <path>   repo checkout (HERMES_HOME\hermes-agent)
#     -Branch <ref>         branch to update against
#     -DesktopPid <pid>     the Electron main process to wait out
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
# immediately), retaining HERMES_UPDATE_STARTED_AT from the Desktop hand-off.
# hermes_cli/update_lock.py's ancestry rule lets our
# `hermes update` child adopt the claim; electron/update-marker.ts parks a
# relaunched Desktop on it. Cleanup only removes the marker while WE still
# own it (a handoff partner that rewrote it keeps its claim).

param(
    [string]$InstallRoot,
    [string]$Branch = "main",
    [int]$DesktopPid = 0,
    [string]$RelaunchExe = "",
    [switch]$NoUi,
    [switch]$NoMarkerCleanup,
    [switch]$SelfTestUi,
    [switch]$SelfTestPipeDrain,
    [switch]$SelfTestMarker
)

if (-not $SelfTestUi -and -not $SelfTestPipeDrain -and -not $InstallRoot) {
    # Mandatory in spirit; relaxed in the signature only so the self-test
    # switches can drive the UI / the pipe drain without a checkout.
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

function Write-HandoffLog([string]$Message) {
    $line = "{0:yyyy-MM-ddTHH:mm:ssK} {1}" -f (Get-Date), $Message
    try { Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 } catch {}
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
                    if ($request -match "^GET /progress") {
                        $elapsed = [Math]::Floor($State.clock.Elapsed.TotalSeconds)
                        $snapshot = @{
                            status          = $State.status
                            message         = $State.message
                            elapsed_seconds = $elapsed
                        } | ConvertTo-Json -Compress
                        Send-Response $stream "200 OK" "application/json; charset=utf-8" ([System.Text.Encoding]::UTF8.GetBytes($snapshot))
                    } elseif ($request -match "^GET / ") {
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
    # The event the shim listens for. One beat of poll latency (400ms) before
    # teardown so the page actually renders the terminal state.
    $script:UiState.message = $Message
    $script:UiState.status = $Status
    if ($script:UiServer) { Start-Sleep -Milliseconds 900 }
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

function Remove-MarkerIfOwned {
    if ($NoMarkerCleanup) { return }
    try {
        if (Test-Path -LiteralPath $MarkerPath) {
            $firstLine = (Get-Content -LiteralPath $MarkerPath -TotalCount 1 -ErrorAction SilentlyContinue)
            if ("$firstLine".Trim() -eq "$PID") {
                Remove-Item -LiteralPath $MarkerPath -Force -ErrorAction SilentlyContinue
                Write-HandoffLog "removed update marker (owned)"
            } else {
                Write-HandoffLog "leaving update marker: owned by pid '$firstLine', not us ($PID)"
            }
        }
    } catch {}
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
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    # .Arguments string (PS 5.1 / .NET Framework has no ArgumentList).
    # Args here are fixed flags + a branch ref; quote each defensively.
    $psi.Arguments = ($HermesArgs | ForEach-Object { '"{0}"' -f ($_ -replace '"', '\"') }) -join ' '
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    # hermes update prints UTF-8 (checkmarks, arrows, box glyphs). PS 5.1
    # defaults these readers to the OEM codepage, which mangles every
    # multi-byte glyph into mojibake in the log.
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    # And ask the child to actually EMIT UTF-8: Python decides its stdio
    # encoding from the console codepage when attached to one.
    $psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"
    $psi.EnvironmentVariables["PYTHONUTF8"] = "1"
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $outSink = New-Object System.Text.StringBuilder
    $errSink = New-Object System.Text.StringBuilder
    $outBuffer = New-Object char[] 16384
    $errBuffer = New-Object char[] 16384
    $outTask = $proc.StandardOutput.ReadAsync($outBuffer, 0, $outBuffer.Length)
    $errTask = $proc.StandardError.ReadAsync($errBuffer, 0, $errBuffer.Length)
    $abandonAt = $null
    $abandoned = $false
    while ($true) {
        $moved = $false
        $outDone = Step-PipeDrain $proc.StandardOutput ([ref]$outTask) $outBuffer $outSink ([ref]$moved)
        $errDone = Step-PipeDrain $proc.StandardError ([ref]$errTask) $errBuffer $errSink ([ref]$moved)
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
    return @{ Code = $proc.ExitCode; Output = $all }
}

$finalCode = 1
$finalMsg = "update did not complete"

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
# Two arms, because the bound and the drain rate fail in opposite directions:
#
#   leak  -- a step whose grandchild outlives it. Guards the #90455 deadlock:
#            the drain must abandon rather than wait out the descendant.
#   flood -- a chatty step that leaks nothing. Guards the other cliff: a drain
#            that idles after every chunk it reads is metered at one buffer per
#            tick, which backpressures the running step. Waiting for EOF and
#            trickling toward it are both ways to make a fast step slow.
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
    [System.IO.File]::WriteAllText($childPs1, $childSource)
    [System.IO.File]::WriteAllText($floodPs1, $floodSource)
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

    Remove-Item -LiteralPath $childPs1, $floodPs1, $pidFile -Force -ErrorAction SilentlyContinue

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

    $detail = "leak: elapsed=${elapsed}s budget=${budget}s code=$($res.Code) grandchildAlive=$leakAlive | flood: ${floodKb}KB in ${floodElapsed}s budget=${floodBudget}s bytes=$floodBytes code=$($flood.Code)"
    if ($problems.Count -gt 0) {
        Write-Host "PIPE-DRAIN SELF-TEST: FAIL $detail -- $($problems -join '; ')"
        exit 1
    }
    Write-Host "PIPE-DRAIN SELF-TEST: PASS $detail"
    exit 0
}

try {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    Remove-Item -LiteralPath $ResultPath -Force -ErrorAction SilentlyContinue
    Show-ProgressWindow
    Write-HandoffLog "hand-off start: root=$InstallRoot branch=$Branch desktopPid=$DesktopPid pid=$PID"

    # -- 0. Claim the update marker with OUR pid ---------------------------
    try {
        $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $startedAt = 0L
        $hasStartedAt = [int64]::TryParse($env:HERMES_UPDATE_STARTED_AT, [ref]$startedAt)
        if (-not $hasStartedAt -or $startedAt -gt $epoch -or ($epoch - $startedAt) -gt 1200) {
            $startedAt = $epoch
        }
        # WriteAllText for byte-exact LF framing: Set-Content emits CRLF and
        # the marker contract (Rust/TS/Python readers) is "<pid>\n<ts>\n".
        [System.IO.File]::WriteAllText($MarkerPath, "$PID`n$startedAt`n")
        Write-HandoffLog "claimed update marker (pid $PID)"
    } catch {
        Write-HandoffLog "WARNING: could not write update marker: $($_.Exception.Message)"
    }

    if ($SelfTestMarker) {
        $finalCode = 0
        $finalMsg = "marker self-test complete"
        exit 0
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
    $res = Invoke-HermesStep $pythonExe $updateArgs "update"
    Write-HandoffLog "hermes update exit code: $($res.Code)"

    if ($res.Code -ne 0 -and $res.Code -ne 2) {
        # One retry for the update-boundary class (fresh code on disk, stale
        # code in memory). Exit 2 ("close all Hermes windows") is not retryable.
        Write-HandoffLog "first attempt failed; retrying once (freshly pulled fix loads on the second run)"
        Publish-UiProgress "Retrying update"
        $res = Invoke-HermesStep $pythonExe $updateArgs "update"
        Write-HandoffLog "retry exit code: $($res.Code)"
    }

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
