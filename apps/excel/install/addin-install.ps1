<#
.SYNOPSIS
  Standalone, idempotent, per-box installer for the Hermes for Excel add-in.

.DESCRIPTION
  requires elevation? NO. Everything lives under %LOCALAPPDATA% and HKCU.

  Steps (in order):
    (a) Ensure Node.js LTS (winget install OpenJS.NodeJS.LTS if node missing).
    (b) Copy the apps/excel payload to the install dir (excludes uploads/exports/*.log).
    (c) Create the HERMES_EXCEL_DATA_DIR.
    (d) Mark-of-the-web unblock (Unblock-File) on the whole install tree.
    (e) Generate + persist a per-install HERMES_EXCEL_BRIDGE_TOKEN.
    (f) Emit run-bridge.cmd + run-bridge.vbs from templates (absolute paths,
        env baked in).
    (g) Register one tracked autostart supervisor via Scheduled Task
        (register-task.ps1), removing the legacy Startup shortcut if present.
        Any previously-running bridge from this install dir is stopped first.
    (h) Register the developer sideload (register-sideload.ps1).
    (i) Health-check: node ... --check-hermes, start the bridge, GET /api/health.
        Fail loudly (non-zero exit) if the bridge does not come up.

  Re-runnable: existence checks + -Force everywhere; never hard-fails on re-run.

.PARAMETER Port
  Bridge port. Use 0 (the default) to select the first available port starting
  at 8787. An explicitly selected occupied port fails instead of silently
  accepting an unrelated service.

.PARAMETER SkipSideload
  Skip the developer sideload registration (component 4).

.PARAMETER SourceDir
  Where the add-in payload is copied FROM. Defaults to the apps/excel tree that
  contains this script (this script lives in <SourceDir>\install).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install\addin-install.ps1
  powershell -ExecutionPolicy Bypass -File install\addin-install.ps1 -Port 8790
#>
[CmdletBinding()]
param(
  [ValidateRange(0, 65535)]
  [int]    $Port         = 0,
  [switch] $SkipSideload,
  [string] $SourceDir
)

$ErrorActionPreference = 'Stop'

# ============================================================================
#  Config block -- a deployer can retarget paths/port/modes here.
# ============================================================================
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $SourceDir) { $SourceDir = Split-Path -Parent $ScriptDir }   # apps/excel
$InstallDir  = Join-Path $env:LOCALAPPDATA 'hermes\excel-addin'
$DataDir     = Join-Path $InstallDir 'data'                            # HERMES_EXCEL_DATA_DIR
$TokenFile   = Join-Path $DataDir   '.bridge-token'                    # 0600-ish secret
$DoclingMode = 'wsl'                                                   # wsl | native | docker (default wsl on Windows)
$WslDistro   = 'Ubuntu-24.04'                                          # HERMES_EXCEL_WSL_DISTRO
$NodeWingetId = 'OpenJS.NodeJS.LTS'
$HealthTimeoutSec = 25
$PreferredPort = 8787
# Items copied; uploads/exports/logs are deliberately excluded.
$ExcludeDirs  = @('uploads', 'exports', 'node_modules', '.git')
$ExcludeFiles = @('*.log')
# ============================================================================

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "    $msg" }
function Write-Warn2($msg) { Write-Warning $msg }

function Test-TcpPortAvailable([int]$Candidate) {
  $listener = $null
  try {
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Candidate)
    $listener.Start()
    return $true
  } catch {
    return $false
  } finally {
    if ($listener) { try { $listener.Stop() } catch {} }
  }
}

function Resolve-BridgePort {
  if ($Port -ne 0) {
    if (-not (Test-TcpPortAvailable $Port)) {
      throw "Requested port $Port is already in use. Stop the conflicting service or rerun without -Port to auto-select."
    }
    return $Port
  }

  foreach ($candidate in $PreferredPort..($PreferredPort + 100)) {
    if (Test-TcpPortAvailable $candidate) {
      if ($candidate -ne $PreferredPort) {
        Write-Warn2 "Port $PreferredPort is occupied; selected $candidate for the Excel bridge."
      }
      return $candidate
    }
  }
  throw "No available bridge port found in $PreferredPort-$($PreferredPort + 100)."
}

# ---------------------------------------------------------------------------
# (a) Ensure Node.js LTS
# ---------------------------------------------------------------------------
function Ensure-Node {
  Write-Step '(a) Ensuring Node.js is available'
  $node = Get-Command node -ErrorAction SilentlyContinue
  if ($node) {
    $ver = (& node --version) 2>$null
    Write-Info "Node present: $ver ($($node.Source))"
    return
  }
  Write-Info 'node not on PATH; attempting install via winget.'
  $winget = Get-Command winget -ErrorAction SilentlyContinue
  if (-not $winget) {
    throw @"
Node.js is not installed and winget is not available.
Install Node.js LTS manually from https://nodejs.org/en/download (or
'App Installer' from the Microsoft Store to get winget), then re-run this script.
"@
  }
  Write-Info "Running: winget install $NodeWingetId"
  & winget install --id $NodeWingetId --silent --accept-source-agreements --accept-package-agreements
  # winget may have updated PATH only for new processes; probe common location.
  $node = Get-Command node -ErrorAction SilentlyContinue
  if (-not $node) {
    $candidate = Join-Path $env:ProgramFiles 'nodejs\node.exe'
    if (Test-Path -LiteralPath $candidate) {
      $env:PATH = "$($env:ProgramFiles)\nodejs;$env:PATH"
      $node = Get-Command node -ErrorAction SilentlyContinue
    }
  }
  if (-not $node) {
    throw "Node was installed by winget but is still not on PATH for this session. Open a new shell and re-run this installer."
  }
  Write-Info "Node now available: $(& node --version)"
}

# ---------------------------------------------------------------------------
# (b) Copy payload
# ---------------------------------------------------------------------------
function Copy-Payload {
  Write-Step "(b) Copying add-in payload to '$InstallDir'"
  if (-not (Test-Path -LiteralPath (Join-Path $SourceDir 'broker\server.mjs'))) {
    throw "Source payload looks wrong: '$SourceDir\broker\server.mjs' not found."
  }
  if (-not (Test-Path -LiteralPath $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
  }

  # robocopy is the most reliable mirror on Windows; /XD/XF exclude dirs/files.
  # Note: we do NOT /MIR so we never delete the live data dir on re-runs.
  $robo = Get-Command robocopy -ErrorAction SilentlyContinue
  if ($robo) {
    $args = @($SourceDir, $InstallDir, '/E', '/NJH', '/NJS', '/NDL', '/NP', '/R:1', '/W:1')
    foreach ($d in $ExcludeDirs)  { $args += @('/XD', (Join-Path $SourceDir $d)) }
    foreach ($f in $ExcludeFiles) { $args += @('/XF', $f) }
    & robocopy @args | Out-Null
    # robocopy exit codes 0-7 are success; 8+ is a real error.
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed copying payload (exit $LASTEXITCODE)." }
  } else {
    # Fallback: manual recursive copy honoring excludes.
    Get-ChildItem -LiteralPath $SourceDir -Recurse -Force | ForEach-Object {
      $rel = $_.FullName.Substring($SourceDir.Length).TrimStart('\')
      $top = ($rel -split '\\')[0]
      if ($ExcludeDirs -contains $top) { return }
      if ($_.PSIsContainer) { return }
      foreach ($pat in $ExcludeFiles) { if ($_.Name -like $pat) { return } }
      $dest = Join-Path $InstallDir $rel
      $destDir = Split-Path -Parent $dest
      if (-not (Test-Path -LiteralPath $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
      Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
  }
  Write-Info 'Payload copied (uploads/exports/*.log excluded).'
}

# ---------------------------------------------------------------------------
# (c) Data dir
# ---------------------------------------------------------------------------
function Ensure-DataDir {
  Write-Step "(c) Ensuring data dir '$DataDir'"
  if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
  }
  Write-Info 'Data dir ready (uploads/exports/logs live here, outside the web root).'
}

# ---------------------------------------------------------------------------
# (d) Mark-of-the-web unblock
# ---------------------------------------------------------------------------
function Unblock-Tree {
  Write-Step '(d) Unblocking files (mark-of-the-web)'
  Get-ChildItem -LiteralPath $InstallDir -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
  Write-Info 'Unblock-File applied across install tree.'
}

# ---------------------------------------------------------------------------
# (e) Bridge token
# ---------------------------------------------------------------------------
function Ensure-BridgeToken {
  Write-Step '(e) Ensuring per-install bridge token'
  if (Test-Path -LiteralPath $TokenFile) {
    $tok = (Get-Content -LiteralPath $TokenFile -Raw).Trim()
    if ($tok) { Write-Info 'Existing bridge token reused.'; return $tok }
  }
  # Two GUIDs concatenated (hex, no braces) -> 64 hex chars of entropy.
  $tok = ([guid]::NewGuid().ToString('N')) + ([guid]::NewGuid().ToString('N'))
  Set-Content -LiteralPath $TokenFile -Value $tok -Encoding ASCII -Force -NoNewline

  # 0600-ish: restrict the secret file to the current user only.
  try {
    $acl = Get-Acl -LiteralPath $TokenFile
    $acl.SetAccessRuleProtection($true, $false)   # disable inheritance, drop inherited
    $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
      $me, 'FullControl', 'Allow')
    $acl.SetAccessRule($rule)
    Set-Acl -LiteralPath $TokenFile -AclObject $acl
  } catch {
    Write-Warn2 "Could not tighten ACL on token file: $($_.Exception.Message)"
  }

  # Also expose as a user env var so other tooling/tests can read it.
  [Environment]::SetEnvironmentVariable('HERMES_EXCEL_BRIDGE_TOKEN', $tok, 'User')
  Write-Info 'Generated new bridge token (persisted to data dir + User env).'
  return $tok
}

# ---------------------------------------------------------------------------
# (f) Emit run-bridge.cmd + run-bridge.vbs from templates
# ---------------------------------------------------------------------------
function Emit-Launchers([string]$BridgeToken) {
  Write-Step '(f) Generating run-bridge.cmd and run-bridge.vbs'
  $cmdTpl = Join-Path $InstallDir 'install\run-bridge.cmd.template'
  $vbsTpl = Join-Path $InstallDir 'install\run-bridge.vbs.template'
  if (-not (Test-Path -LiteralPath $cmdTpl)) { throw "Missing template '$cmdTpl'." }
  if (-not (Test-Path -LiteralPath $vbsTpl)) { throw "Missing template '$vbsTpl'." }

  # InstallDir without trailing backslash for clean substitution.
  $idir = $InstallDir.TrimEnd('\')

  $cmd = Get-Content -LiteralPath $cmdTpl -Raw
  $cmd = $cmd.Replace('__INSTALL_DIR__',   $idir)
  $cmd = $cmd.Replace('__PORT__',          "$Port")
  $cmd = $cmd.Replace('__DATA_DIR__',      $DataDir)
  $cmd = $cmd.Replace('__BRIDGE_TOKEN__',  $BridgeToken)
  $cmd = $cmd.Replace('__DOCLING_MODE__',  $DoclingMode)
  $cmd = $cmd.Replace('__WSL_DISTRO__',    $WslDistro)
  Set-Content -LiteralPath (Join-Path $InstallDir 'run-bridge.cmd') -Value $cmd -Encoding ASCII -Force

  $vbs = Get-Content -LiteralPath $vbsTpl -Raw
  $vbs = $vbs.Replace('__INSTALL_DIR__', $idir)
  Set-Content -LiteralPath (Join-Path $InstallDir 'run-bridge.vbs') -Value $vbs -Encoding ASCII -Force

  Write-Info 'Launchers written with absolute paths and env baked in.'
}

# ---------------------------------------------------------------------------
# Stop any bridge previously launched from this install dir.
# ---------------------------------------------------------------------------
function Stop-ExistingBridge {
  Write-Info 'Stopping any previously-running bridge from this install dir...'
  $idir = $InstallDir.TrimEnd('\')
  try {
    Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$idir*") } |
      ForEach-Object {
        Write-Info "  killing stale bridge PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      }
  } catch {
    Write-Warn2 "Could not enumerate node processes: $($_.Exception.Message)"
  }
}

# ---------------------------------------------------------------------------
# (g) Autostart: one tracked scheduled task
# ---------------------------------------------------------------------------
function Register-Autostart {
  Write-Step '(g) Registering the tracked Scheduled Task supervisor'
  Stop-ExistingBridge

  # Remove the legacy Startup shortcut. Running both it and the Scheduled Task
  # creates competing supervisors after logon and can cause restart churn.
  $startup = [Environment]::GetFolderPath('Startup')
  $lnkPath = Join-Path $startup 'Hermes Excel Bridge.lnk'
  if (Test-Path -LiteralPath $lnkPath) {
    Remove-Item -LiteralPath $lnkPath -Force -ErrorAction SilentlyContinue
    Write-Info "Removed legacy Startup shortcut '$lnkPath'."
  }

  # Scheduled Task (logon-triggered supervisor).
  $regTask = Join-Path $InstallDir 'install\register-task.ps1'
  if (Test-Path -LiteralPath $regTask) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $regTask `
      -InstallDir $InstallDir -DataDir $DataDir -Port $Port
  } else {
    Write-Warn2 "register-task.ps1 not found at '$regTask'; scheduled task skipped."
  }
}

# ---------------------------------------------------------------------------
# (h) Developer sideload
# ---------------------------------------------------------------------------
function Register-Sideload {
  if ($SkipSideload) { Write-Step '(h) Skipping sideload (-SkipSideload)'; return }
  Write-Step '(h) Registering developer sideload'
  $regSide = Join-Path $InstallDir 'install\register-sideload.ps1'
  if (Test-Path -LiteralPath $regSide) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $regSide `
      -InstallDir $InstallDir -Port $Port
  } else {
    Write-Warn2 "register-sideload.ps1 not found at '$regSide'; sideload skipped."
  }
}

# ---------------------------------------------------------------------------
# (i) Health check
# ---------------------------------------------------------------------------
function Invoke-HealthCheck([string]$BridgeToken) {
  Write-Step '(i) Health check'
  $server = Join-Path $InstallDir 'broker\server.mjs'

  # --check-hermes is allowed to fail (the Hermes gateway may simply be down);
  # we report but do not abort on it.
  Write-Info 'Running: node broker\server.mjs --check-hermes'
  try {
    & node $server --check-hermes
    if ($LASTEXITCODE -eq 0) { Write-Info 'Hermes endpoint: OK' }
    else { Write-Warn2 'Hermes endpoint check failed (gateway may be down). Continuing.' }
  } catch {
    Write-Warn2 "Hermes endpoint check error: $($_.Exception.Message). Continuing."
  }

  # Start the bridge via the generated launcher (hidden), then poll /api/health.
  Write-Info "Starting bridge and probing http://localhost:$Port/api/health ..."
  $vbs = Join-Path $InstallDir 'run-bridge.vbs'
  & (Join-Path $env:SystemRoot 'System32\wscript.exe') "$vbs"

  $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
  $health = $null
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 800
    try {
      $resp = Invoke-RestMethod -Uri "http://localhost:$Port/api/health" -TimeoutSec 3 -Method Get
      if ($resp -and $resp.service -eq 'hermes-excel-bridge' -and [int]$resp.port -eq $Port) {
        $health = $resp
        break
      }
    } catch { }
  }

  if (-not $health) {
    throw "Bridge did NOT respond on http://localhost:$Port/api/health within $HealthTimeoutSec s. Install FAILED."
  }

  $hermesOk  = if ($health.hermes  -and $health.hermes.ok)  { 'OK' } else { 'DOWN' }
  $doclingOk = if ($health.docling -and $health.docling.ok) { 'OK' } else { 'DOWN' }
  Write-Host ''
  Write-Host "    Bridge:  UP  (port $Port)" -ForegroundColor Green
  Write-Host "    Hermes:  $hermesOk"
  Write-Host "    Docling: $doclingOk"
}

# ============================================================================
#  Main
# ============================================================================
try {
  # A re-run may own the requested port already. Stop only bridges launched
  # from this install directory before deciding whether the port is available.
  Stop-ExistingBridge
  $Port = Resolve-BridgePort
  Write-Host ''
  Write-Host "Hermes for Excel installer  (port $Port)" -ForegroundColor White
  Write-Host "Source : $SourceDir"
  Write-Host "Install: $InstallDir"
  Write-Host ''

  Ensure-Node
  Copy-Payload
  Ensure-DataDir
  Unblock-Tree
  $token = Ensure-BridgeToken
  Emit-Launchers $token
  Register-Autostart
  Register-Sideload
  Invoke-HealthCheck $token

  Write-Host ''
  Write-Host 'Hermes for Excel installed successfully.' -ForegroundColor Green
  Write-Host "  - Restart Excel; open the pane from the Hermes ribbon group."
  Write-Host "  - Uninstall with: install\rollback.ps1"
  Write-Host ''
  exit 0
}
catch {
  Write-Host ''
  Write-Error "INSTALL FAILED: $($_.Exception.Message)"
  exit 1
}
