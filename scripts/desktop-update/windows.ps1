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
# immediately). hermes_cli/update_lock.py's ancestry rule lets our
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
    [switch]$SelfTestUi
)

if (-not $SelfTestUi -and -not $InstallRoot) {
    # Mandatory in spirit; relaxed in the signature only so -SelfTestUi can
    # drive the UI without a checkout.
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

function Write-HandoffLog([string]$Message) {
    $line = "{0:yyyy-MM-ddTHH:mm:ssK} {1}" -f (Get-Date), $Message
    try { Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 } catch {}
    Write-Host $line
}

# ── The shim: repo-owned HTML in a chromeless Edge app window ──────────────
# The window is a veneer, not a participant: the update runs identically with
# or without it (Edge missing/failed degrades to the WinForms card below,
# then log-only). It streams nothing and knows nothing — it polls /progress
# for one of two events, `done` or `error`, and reacts. The loopback listener
# is not a web server in any meaningful sense; it exists because file:// pages
# cannot receive events from a detached process. Salvaged from the web-shell
# spike (Co-authored-by: teknium1), reshaped to the quiet update-surface
# contract (#75895/#83634): loader, one title, one line, no dashboard.
$script:UiState = [hashtable]::Synchronized(@{
    status  = "running"      # running | done | error
    message = ""
})
$script:UiServer = $null     # @{ Listener; Runspace; PowerShell; Port; EdgeProc }

function Get-UiHtmlPath {
    # Lives next to this script in the checkout. Missing file = fall back to
    # WinForms (old checkouts mid-update, partial syncs).
    $p = Join-Path $PSScriptRoot "ui.html"
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Find-EdgeExe {
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
        if (-not $root) { continue }
        $p = Join-Path $root "Microsoft\Edge\Application\msedge.exe"
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
                        $snapshot = @{
                            status  = $State.status
                            message = $State.message
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

        return @{ Listener = $listener; Runspace = $rs; PowerShell = $ps; Port = $port; EdgeProc = $null }
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
            if ($script:UiServer.EdgeProc -and -not $script:UiServer.EdgeProc.HasExited) {
                $script:UiServer.EdgeProc.CloseMainWindow() | Out-Null
            }
        } catch {}
    }
    $script:UiServer = $null
}

function Publish-UiEvent([string]$Status, [string]$Message) {
    # The event the shim listens for. One beat of poll latency (400ms) before
    # teardown so the page actually renders the terminal state.
    $script:UiState.message = $Message
    $script:UiState.status = $Status
    if ($script:UiServer) { Start-Sleep -Milliseconds 900 }
}

# ── Fallback card (no Edge / no HTML): same shape in WinForms ──────────────
# Matches the shim pixel-for-pixel in spirit -- loader, one title, one static
# line, OS light/dark -- so degrading is invisible to the user.
function Get-AppsUseLightTheme {
    try {
        $v = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" -Name AppsUseLightTheme -ErrorAction Stop
        return [int]$v.AppsUseLightTheme -ne 0
    } catch { return $true }
}

function Show-ProgressWindow {
    if ($NoUi) { return }

    # ── Primary: the HTML shim in a chromeless Edge app window ─────────────
    # Same footprint as the card (280x320), spawned as a normal window: it
    # claims attention once by appearing, then competes with nothing.
    $htmlPath = Get-UiHtmlPath
    $edge = Find-EdgeExe
    if ($htmlPath -and $edge) {
        $server = Start-UiServer $htmlPath
        if ($server) {
            try {
                # Dedicated tiny profile dir: guarantees a NEW WINDOW + process
                # we own (a default-profile launch delegates to an existing
                # Edge and returns instantly, leaving nothing to close), and
                # avoids touching the user's real browser profile.
                $edgeProfile = Join-Path $TempDir ("hermes-update-ui-{0}" -f $PID)
                $edgeArgs = @(
                    "--app=http://127.0.0.1:$($server.Port)/",
                    "--user-data-dir=$edgeProfile",
                    "--no-first-run", "--no-default-browser-check",
                    "--disable-features=msImplicitSignin",
                    "--window-size=280,320"
                )
                $server.EdgeProc = Start-Process -FilePath $edge -ArgumentList $edgeArgs -PassThru
                $script:UiServer = $server
                Write-HandoffLog "shim: Edge app window on 127.0.0.1:$($server.Port)"
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
        $sub.Text = "Hermes will open once done."
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
        $script:Ui = [pscustomobject]@{ Form = $form; Bar = $bar; Title = $title; Sub = $sub }
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
        try { $script:Ui.Form.Close() } catch {}
        $script:Ui = $null
    }
}

function Write-Result([bool]$Ok, [int]$Code, [string]$Message) {
    # Consumed (read + deleted) by the relaunched Desktop on boot so the
    # user actually SEES how a detached update ended.
    try {
        $obj = @{
            ok         = $Ok
            exit_code  = $Code
            message    = $Message
            branch     = $Branch
            finished_at = [int][double]::Parse((Get-Date -UFormat %s), [System.Globalization.CultureInfo]::InvariantCulture)
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
    if (-not ($RelaunchExe -and (Test-Path -LiteralPath $RelaunchExe))) { return $false }
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

function Invoke-HermesStep([string]$Exe, [string[]]$HermesArgs, [string]$Tag) {
    # The window shows nothing live, so no line-pump: both pipes drain
    # asynchronously (no deadlock however chatty the child) while a small
    # DoEvents loop keeps the marquee animating through long silent
    # stretches (pip installs) -- the old EndOfStream pump blocked on quiet
    # children and froze it. Full output still lands in the hand-off log
    # afterwards, where `hermes debug share` picks it up.
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
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 150
        if ($script:Ui) { [System.Windows.Forms.Application]::DoEvents() }
    }
    $proc.WaitForExit()
    $outText = $outTask.Result
    $errText = $errTask.Result
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
    Start-Sleep -Seconds $hold
    if ($env:HERMES_SELFTEST_FAIL) {
        Show-ErrorFinale "self-test error state"
    } else {
        Close-ProgressWindow
    }
    exit 0
}

try {
    New-Item -ItemType Directory -Path $LogDir -Force -ErrorAction SilentlyContinue | Out-Null
    Remove-Item -LiteralPath $ResultPath -Force -ErrorAction SilentlyContinue
    Show-ProgressWindow
    Write-HandoffLog "hand-off start: root=$InstallRoot branch=$Branch desktopPid=$DesktopPid pid=$PID"

    # -- 0. Claim the update marker with OUR pid ---------------------------
    try {
        $epoch = [int][double]::Parse((Get-Date -UFormat %s), [System.Globalization.CultureInfo]::InvariantCulture)
        # WriteAllText for byte-exact LF framing: Set-Content emits CRLF and
        # the marker contract (Rust/TS/Python readers) is "<pid>\n<ts>\n".
        [System.IO.File]::WriteAllText($MarkerPath, "$PID`n$epoch`n")
        Write-HandoffLog "claimed update marker (pid $PID)"
    } catch {
        Write-HandoffLog "WARNING: could not write update marker: $($_.Exception.Message)"
    }

    # -- 1. Wait for the Desktop to exit (FAIL CLOSED) ----------------------
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
    $hermesExe = Join-Path $InstallRoot "venv\Scripts\hermes.exe"
    if (-not (Test-Path -LiteralPath $hermesExe)) {
        $finalCode = 3
        $finalMsg = "Update aborted: $hermesExe is missing. The install needs repair (run the Hermes installer or `hermes doctor`)."
        Write-HandoffLog $finalMsg
        exit $finalCode
    }
    $updateArgs = @("update", "--yes", "--gateway", "--force", "--branch", $Branch)
    Write-HandoffLog ("running: hermes " + ($updateArgs -join " "))
    $res = Invoke-HermesStep $hermesExe $updateArgs "update"
    Write-HandoffLog "hermes update exit code: $($res.Code)"

    if ($res.Code -ne 0 -and $res.Code -ne 2) {
        # One retry for the update-boundary class (fresh code on disk, stale
        # code in memory). Exit 2 ("close all Hermes windows") is not retryable.
        Write-HandoffLog "first attempt failed; retrying once (freshly pulled fix loads on the second run)"
        $res = Invoke-HermesStep $hermesExe $updateArgs "update"
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
        $rebuild = Invoke-HermesStep $hermesExe @("desktop", "--force-build", "--build-only") "rebuild"
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
        $cameBack = Start-DesktopRelaunch
        if (-not $cameBack -and $RelaunchExe) {
            # Launch was due and did not verifiably land: truthful result
            # for the next boot, manual state held on screen now.
            $finalMsg = "Update complete. Reopen Hermes to finish (it could not restart itself)."
            Write-Result $true 0 $finalMsg
            Show-ManualFinale $finalMsg
        }
        Close-ProgressWindow
    }
}
