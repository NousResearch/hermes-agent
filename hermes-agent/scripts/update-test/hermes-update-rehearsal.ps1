<#
.SYNOPSIS
  hermes-update-rehearsal.ps1 -- for an EXISTING Hermes install (Windows).

.DESCRIPTION
  Two steps:

    pre   take a `hermes backup` of your data, snapshot the disk that holds
          HERMES_HOME and the desktop app's Electron userData (Volume Shadow Copy),
          then point the install's update source at a fork so `hermes update`
          pulls that fork's main. Prints what to do next.
    post  put both trees back exactly as they were at the snapshot.

  Plus `status`, which only prints. This script never judges your install: it
  reports what it did and stops. Whether the update worked is yours to see.

  The snapshot is copy-on-write, so pre copies nothing and takes seconds. post
  mirrors the snapshot back over both trees: only files the update changed are
  copied, and files it added are deleted. HERMES_HOME\cache is left as it is.

  If Windows drops the snapshot (it can when the disk's shadow storage fills),
  your files are untouched but post cannot roll back; the `hermes backup` zip
  still holds your config, keys, sessions, memories and skills. To make that
  unlikely, pre raises the shadow-storage cap to 128 GB (a ceiling, not a
  reservation) and post puts the original cap back.

  pre and post need an elevated (Run as Administrator) PowerShell.

.PARAMETER Command
  pre | post | status

.PARAMETER Source
  Repo to pull the update from; updates follow its main (default: the
  rehearsal fork).

.PARAMETER BackupRoot
  Where the backup lives (default: $HOME\hermes-update-rehearsal).

.PARAMETER Yes
  post: skip the confirmation.

.EXAMPLE
  ./hermes-update-rehearsal.ps1 pre -Source <git-url>
  # ... run `hermes update`, use Hermes, test ...
  ./hermes-update-rehearsal.ps1 post

.NOTES
  `pre` and `hermes update` need network access to -Source, and a usable git on PATH.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('pre', 'post', 'status', 'help')]
  [string]$Command = 'help',

  [string]$Source = 'https://github.com/ethernet8023/hermes-agent.git',
  [string]$BackupRoot,
  [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$OfficialHttps = 'https://github.com/NousResearch/hermes-agent.git'
$OfficialSsh = 'git@github.com:NousResearch/hermes-agent.git'
# Diff-area cap pre sets while the snapshot is alive (post restores the original).
$ShadowStorageMax = [UInt64]128GB

$script:Snap = ''

function Say  { param([string]$m) Write-Host $m }
function Ok   { param([string]$m) Write-Host "  OK $m" }
function Warn { param([string]$m) Write-Warning "  $m" }
function Step { param([string]$m) Write-Host "`n=== $m ===" }
function Fail { param([string]$m) throw "ERROR: $m" }

function Assert-Elevated {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $admin = (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $admin) {
    Fail 'the disk snapshot needs admin rights -- open PowerShell with "Run as Administrator" and run this again (nothing was changed)'
  }
}

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
function Get-ResolvedPaths {
  $suffix = if ($env:HERMES_DATA_DIR_SUFFIX) { $env:HERMES_DATA_DIR_SUFFIX } else { '' }
  $userProfile = $env:USERPROFILE
  if ($env:HERMES_HOME) {
    $home_ = $env:HERMES_HOME
    try { $home_ = (Resolve-Path -LiteralPath $home_ -ErrorAction Stop).Path }
    catch { $home_ = [IO.Path]::GetFullPath($home_) }
  }
  else {
    $base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $userProfile 'AppData\Local' }
    $home_ = "$(Join-Path $base 'hermes')$suffix"
  }
  $install = Join-Path $home_ 'hermes-agent'
  if ($env:HERMES_DESKTOP_USER_DATA_DIR) {
    $ud = $env:HERMES_DESKTOP_USER_DATA_DIR
    try { $ud = (Resolve-Path -LiteralPath $ud -ErrorAction Stop).Path }
    catch { $ud = [IO.Path]::GetFullPath($ud) }
    $userData = $ud
    $userDataSource = 'env'
  }
  else {
    $appData = if ($env:APPDATA) { $env:APPDATA } else { Join-Path $userProfile 'AppData\Roaming' }
    $userData = Join-Path $appData "Hermes$suffix"
    $userDataSource = 'default'
  }
  return [pscustomobject]@{
    Home = $home_; Install = $install
    UserData = $userData; UserDataOrigin = $userDataSource
  }
}

function Get-LatestSnapshot {
  if (-not (Test-Path -LiteralPath $BackupRoot)) { return $null }
  $dirs = Get-ChildItem -LiteralPath $BackupRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name
  if (-not $dirs) { return $null }
  return $dirs[-1].FullName
}

function Load-Snapshot {
  $snap = Get-LatestSnapshot
  if (-not $snap) { Fail "no backup found under $BackupRoot -- run 'pre' first" }
  $script:Snap = $snap
  $recordedPath = Join-Path $snap 'hermes-home.txt'
  if (-not (Test-Path -LiteralPath $recordedPath)) { Fail "$snap is not a rehearsal backup (no hermes-home.txt)" }
  if (-not (Test-Path -LiteralPath (Join-Path $snap 'shadows.txt'))) { Fail "$snap has no disk snapshot (made by an older version of this script)" }
  $P = Get-ResolvedPaths
  $recorded = ((Get-Content -LiteralPath $recordedPath -Raw) -replace "`r", '').Trim()
  if ($recorded -ne $P.Home) {
    Fail "that backup belongs to HERMES_HOME=$recorded, not $($P.Home); pass -BackupRoot to pick the right one"
  }
}

# ---------------------------------------------------------------------------
# git helpers (stdout only: merging stderr folds git warnings into the value)
# ---------------------------------------------------------------------------

function Invoke-Git {
  param($P, [string[]]$GitArgs)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = & git -C $P.Install @GitArgs 2>$null }
  finally { $ErrorActionPreference = $prev }
  return (($out | Out-String) -replace "`r", '').Trim()
}

function Invoke-GitCmd {
  param([string[]]$GitArgs)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & git @GitArgs 2>$null } finally { $ErrorActionPreference = $prev }
}

# ---------------------------------------------------------------------------
# Volume Shadow Copy
# ---------------------------------------------------------------------------
# One snapshot per volume. shadows.txt holds one "volume<TAB>id<TAB>device" line
# each; post looks every id up again and refuses to touch anything if one is gone.

function Get-VolumeRoot {
  param([string]$Path)
  $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path))
  if ($root -notmatch '^[A-Za-z]:\\$') { Fail "$Path is not on a local drive letter; a disk snapshot cannot cover it" }
  return $root.ToUpperInvariant()
}

function Get-ShadowStorageMax {
  # Exact bytes (UInt64::MaxValue = UNBOUNDED) of the diff-area cap for snapshots
  # of $Volume, or $null when the volume has no shadow-storage association yet.
  param([string]$Volume)
  $vol = Get-CimInstance Win32_Volume | Where-Object { $_.Name -eq $Volume }
  if (-not $vol) { return $null }
  $st = Get-CimInstance Win32_ShadowStorage | Where-Object { $_.Volume.DeviceID -eq $vol.DeviceID } | Select-Object -First 1
  if (-not $st) { return $null }
  return [UInt64]$st.MaxSpace
}

function Set-ShadowStorageMax {
  param([string]$Volume, [UInt64]$Bytes)
  $size = if ($Bytes -eq [UInt64]::MaxValue) { 'UNBOUNDED' } else { "$Bytes" }
  $d = $Volume.TrimEnd('\')
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = & vssadmin resize shadowstorage "/for=$d" "/on=$d" "/maxsize=$size" 2>&1; $code = $LASTEXITCODE }
  finally { $ErrorActionPreference = $prev }
  if ($code -ne 0) { $out | Where-Object { "$_".Trim() } | ForEach-Object { Write-Host "    $_" } }
  return ($code -eq 0)
}

function Format-Bytes {
  param([UInt64]$Bytes)
  if ($Bytes -eq [UInt64]::MaxValue) { return 'unbounded' }
  return '{0:N1} GB' -f ($Bytes / 1GB)
}

# post: put every diff-area cap pre raised back to what it was.
function Restore-ShadowStorage {
  $file = Join-Path $script:Snap 'shadowstorage.txt'
  if (-not (Test-Path -LiteralPath $file)) { return }
  foreach ($line in @(Get-Content -LiteralPath $file | Where-Object { $_.Trim() })) {
    $f = $line -split "`t"
    if (Set-ShadowStorageMax $f[0] ([UInt64]$f[1])) { Ok "snapshot room on $($f[0]) back to $(Format-Bytes ([UInt64]$f[1]))" }
    else { Warn "could not put the snapshot room on $($f[0]) back to $(Format-Bytes ([UInt64]$f[1])); run: vssadmin resize shadowstorage /for=$($f[0].TrimEnd('\')) /on=$($f[0].TrimEnd('\')) /maxsize=$($f[1])" }
  }
}

# CIM, never Get-WmiObject: pwsh has no WMI v1 cmdlets and proxies them through a
# Windows PowerShell compat session, whose objects come back deserialized with
# their methods (Create, Delete) stripped.
function New-Shadow {
  param([string]$Volume)
  $r = Invoke-CimMethod -ClassName Win32_ShadowCopy -MethodName Create -Arguments @{ Volume = $Volume; Context = 'ClientAccessible' }
  if ($r.ReturnValue -ne 0) { Fail "could not snapshot $Volume (Win32_ShadowCopy.Create returned $($r.ReturnValue))" }
  $sc = Get-CimInstance Win32_ShadowCopy | Where-Object { $_.ID -eq $r.ShadowID }
  return [pscustomobject]@{ Volume = $Volume; Id = $sc.ID; Device = $sc.DeviceObject }
}

function Read-Shadows {
  return @(Get-Content -LiteralPath (Join-Path $script:Snap 'shadows.txt') | Where-Object { $_.Trim() } | ForEach-Object {
    $f = $_ -split "`t"
    [pscustomobject]@{ Volume = $f[0]; Id = $f[1]; Device = $f[2] }
  })
}

function Test-ShadowAlive {
  param($Shadow)
  return [bool](Get-CimInstance Win32_ShadowCopy | Where-Object { $_.ID -eq $Shadow.Id })
}

function Remove-Shadow {
  param($Shadow)
  Get-CimInstance Win32_ShadowCopy | Where-Object { $_.ID -eq $Shadow.Id } | Remove-CimInstance
}

function Get-MountPath {
  param($Shadow)
  return Join-Path $script:Snap ('vss-' + $Shadow.Volume.Substring(0, 1))
}

function Dismount-Shadow {
  param([string]$Link)
  # rmdir WITHOUT /s: removes the link, never what it points at.
  if (Test-Path -LiteralPath $Link) { cmd /c rmdir "$Link" | Out-Null }
}

function Mount-Shadow {
  param($Shadow)
  $link = Get-MountPath $Shadow
  Dismount-Shadow $link
  $null = cmd /c mklink /d "$link" "$($Shadow.Device)\"
  if ($LASTEXITCODE -ne 0) { Fail "could not mount the snapshot of $($Shadow.Volume) at $link" }
  return $link
}

# Mirror one tree from the snapshot over the live tree: copies what differs,
# deletes what the snapshot does not have. Returns robocopy's summary lines.
function Invoke-Mirror {
  param([string]$From, [string]$To, [string[]]$ExcludeDirs = @())
  $roboArgs = @($From, $To, '/MIR', '/COPY:DAT', '/DCOPY:DAT', '/SL', '/MT:16', '/R:2', '/W:1', '/NP', '/NFL', '/NDL', '/NJH')
  if ($ExcludeDirs.Count) { $roboArgs += '/XD'; $roboArgs += $ExcludeDirs }
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = & robocopy @roboArgs 2>&1; $code = $LASTEXITCODE }
  finally { $ErrorActionPreference = $prev }
  # robocopy: 0-7 = success (bit flags for copied/extra/mismatched), 8+ = failures.
  if ($code -ge 8) {
    $out | ForEach-Object { Write-Host "    $_" }
    Fail "restoring $To failed (robocopy exit $code) -- the snapshot is kept; fix the cause and run post again"
  }
  return @($out | Where-Object { $_ -match '^\s*(Total\s+Copied|Dirs\s*:|Files\s*:|Bytes\s*:)' })
}

# ---------------------------------------------------------------------------
# pre
# ---------------------------------------------------------------------------

function Resolve-HermesExe {
  param($P)
  foreach ($c in @(
      (Join-Path $P.Install 'venv\Scripts\hermes.exe'),
      (Join-Path $P.Home 'bin\hermes.exe'),
      (Join-Path $P.Home 'bin\hermes.cmd'))) {
    if (Test-Path -LiteralPath $c) { return $c }
  }
  return $null
}

function Invoke-Pre {
  $P = Get-ResolvedPaths
  Step 'your install'
  Say "HERMES_HOME   $($P.Home)"
  Say "install       $($P.Install)"
  Say "desktop data  $($P.UserData) ($($P.UserDataOrigin))"
  Say "backup to     $BackupRoot"
  if (-not (Test-Path -LiteralPath $P.Home)) { Fail "no HERMES_HOME at $($P.Home)" }
  if (-not (Test-Path -LiteralPath (Join-Path $P.Install '.git'))) { Fail "no git checkout at $($P.Install) -- this tool covers source installs" }
  Assert-Elevated
  $hermesExe = Resolve-HermesExe $P
  if (-not $hermesExe) { Fail "no hermes executable found under $($P.Install)\venv\Scripts or $($P.Home)\bin" }

  $volumes = @(Get-VolumeRoot $P.Home)
  $userDataExists = Test-Path -LiteralPath $P.UserData
  if ($userDataExists) { $volumes += Get-VolumeRoot $P.UserData }
  $volumes = @($volumes | Select-Object -Unique)

  Step 'before we start (nothing here is pass/fail, just read it)'
  $procs = @(Get-Process -Name 'Hermes', 'hermes' -ErrorAction SilentlyContinue)
  if ($procs.Count) {
    Warn 'Hermes looks like it is running -- close the desktop app and the gateway'
    Warn "before you run 'hermes update', or the dependency sync may fail:"
    $procs | ForEach-Object { Write-Host "    $($_.ProcessName) (pid $($_.Id))" }
  }
  else { Ok 'no Hermes processes running' }
  $n = @(Invoke-GitCmd @('config', '--global', '--get-regexp', '^url\.')).Count + @(Invoke-GitCmd @('-C', $P.Install, 'config', '--local', '--get-regexp', '^url\.')).Count
  if ($n -eq 0) { Ok 'global git config has no URL rewrites' }
  else { Warn "$n existing url.* insteadOf entr(y/ies) in your git config; we add more and remove only ours" }

  # Before anything is written: a bad -Source or no network should abort with
  # nothing done, not after the backup and snapshot.
  $targetSha = ((@(Invoke-GitCmd @('ls-remote', $Source, 'refs/heads/main')) | Out-String) -split '\s+')[0]
  if ($targetSha -notmatch '^[0-9a-f]{40}$') { Fail "could not read main from $Source (network? permissions? bad -Source?) -- nothing was done" }
  Ok "$Source main is at $targetSha"

  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
  $script:Snap = Join-Path $BackupRoot $stamp
  if (Test-Path -LiteralPath $script:Snap) { Fail "backup dir already exists: $($script:Snap)" }
  New-Item -ItemType Directory -Force -Path $script:Snap | Out-Null

  Step 'backing up your data (hermes backup)'
  # Insurance for the case where Windows drops the snapshot: config, keys,
  # sessions, memories, skills, with SQLite copied consistently. Code, venvs and
  # models are left out -- those are what a reinstall brings back.
  $zip = Join-Path $script:Snap 'hermes-backup.zip'
  $started = Get-Date
  $prevHome = $env:HERMES_HOME
  $env:HERMES_HOME = $P.Home
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = & $hermesExe backup -o $zip 2>&1; $code = $LASTEXITCODE }
  finally { $ErrorActionPreference = $prev; $env:HERMES_HOME = $prevHome }
  $elapsed = [int]((Get-Date) - $started).TotalSeconds
  if ($code -eq 1 -and (Test-Path -LiteralPath $zip)) {
    $out | ForEach-Object { Write-Host "    $_" }
    Warn "hermes backup finished INCOMPLETE (${elapsed}s): the files listed above are not in $zip"
  }
  elseif ($code -ne 0 -or -not (Test-Path -LiteralPath $zip)) {
    $out | ForEach-Object { Write-Host "    $_" }
    Fail "hermes backup failed (exit $code) -- nothing else was done"
  }
  else { Ok "hermes-backup.zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB, ${elapsed}s)" }

  # --- snapshot: the rollback point. Everything after this is undone by post.
  Step 'snapshotting the disk'
  $shadowLines = @()
  foreach ($v in $volumes) {
    $t0 = Get-Date
    $s = New-Shadow $v
    $shadowLines += "$($s.Volume)`t$($s.Id)`t$($s.Device)"
    Ok "snapshot of $v in $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1))s ($($s.Id))"
  }
  Set-Content -LiteralPath (Join-Path $script:Snap 'shadows.txt') -Encoding utf8 -Value $shadowLines

  # The snapshot keeps the OLD copy of every block rewritten anywhere on the
  # disk; past the cap Windows deletes the snapshot. The cap is a ceiling, not
  # a reservation. Raised AFTER the snapshot: on client Windows the storage
  # association may only exist once a shadow does.
  # Each original cap is recorded BEFORE it is raised, so a pre that dies
  # mid-way still leaves post and status the record to put it back from;
  # restoring a cap that never got raised is a no-op.
  $storageFile = Join-Path $script:Snap 'shadowstorage.txt'
  Set-Content -LiteralPath $storageFile -Encoding utf8 -Value @()
  foreach ($v in $volumes) {
    $orig = Get-ShadowStorageMax $v
    if ($null -eq $orig) { Warn "could not read the snapshot room on $v; leaving it as it is"; continue }
    if ($orig -ge $ShadowStorageMax) { Ok "snapshot room on $v is $(Format-Bytes $orig)"; continue }
    Add-Content -LiteralPath $storageFile -Encoding utf8 -Value "$v`t$orig"
    if (Set-ShadowStorageMax $v $ShadowStorageMax) {
      Ok "snapshot room on $v raised from $(Format-Bytes $orig) to $(Format-Bytes $ShadowStorageMax) (post puts it back)"
    }
    else { Warn "could not raise the snapshot room on $v; it stays $(Format-Bytes $orig) -- writing more than that to the disk before post drops the snapshot" }
  }

  # Plain text, not JSON: post compares this string byte-for-byte to decide
  # whether the backup belongs to the home it is about to restore.
  Set-Content -LiteralPath (Join-Path $script:Snap 'hermes-home.txt') -Encoding utf8 -Value @($P.Home)
  $manifest = [ordered]@{
    schema              = 4
    created             = (Get-Date).ToUniversalTime().ToString('o')
    hermes_home         = $P.Home
    install_dir         = $P.Install
    userdata_dir        = $P.UserData
    userdata_dir_source = $P.UserDataOrigin
    userdata_existed    = [bool]$userDataExists
    rehearsal_source    = $Source
  }
  $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:Snap 'manifest.json') -Encoding utf8

  Step "pointing your install at $Source"
  # insteadOf is a TRANSPORT rewrite. Your checkout's origin keeps the official
  # URL, which matters: `hermes update` resolves its channel from the archive and
  # validates it against `git config --get remote.origin.url`. Repointing origin
  # at a fork would make the update fail before any git work.
  # Straight at the fork: the updater follows main, so the fork's main is what
  # lands (force-push it to the branch under test).
  # REPO-LOCAL: the checkout's config lives inside the snapshotted home, so
  # post's restore removes the redirect for free.
  foreach ($url in @($OfficialHttps, $OfficialSsh)) {
    # --add: the key is multi-valued; a plain set would drop the first URL.
    $null = Invoke-GitCmd @('-C', $P.Install, 'config', '--local', '--add', "url.$Source.insteadOf", $url)
    if ($LASTEXITCODE -ne 0) { Fail "could not write the URL redirect into $($P.Install)\.git\config" }
  }
  Set-Content -LiteralPath (Join-Path $P.Home '.skip_upstream_prompt') -Encoding utf8 -Value @()
  Set-Content -LiteralPath (Join-Path $script:Snap 'target-sha') -Encoding utf8 -Value @($targetSha)
  Ok "official repo URL now resolves to $Source"
  Ok "created $($P.Home)\.skip_upstream_prompt (stops the 'add upstream remote?' prompt)"

  Step 'ready'
  Say 'your install is unchanged so far -- nothing has been updated yet.'
  Say ''
  Say 'continue with the instructions provided'
  Say "your backup is at $($script:Snap) -- keep it until post has run."
}

# ---------------------------------------------------------------------------
# status (read-only)
# ---------------------------------------------------------------------------

function Invoke-Status {
  $P = Get-ResolvedPaths
  Step 'your install'
  Say "HERMES_HOME   $($P.Home)"
  Say "install       $($P.Install)"
  Say "desktop data  $($P.UserData) ($($P.UserDataOrigin))"
  Say "backup root   $BackupRoot"
  Step 'backup'
  $snap = Get-LatestSnapshot
  if (-not $snap) { Say "none -- nothing has been set up yet (run 'pre')"; return }
  $script:Snap = $snap
  Say "latest        $snap"
  $ts = Join-Path $snap 'target-sha'
  if (Test-Path -LiteralPath $ts) {
    Say "prepared for  $((Get-Content -LiteralPath $ts -Raw).Trim()) (main at pre time)"
    $manifest = Get-Content -LiteralPath (Join-Path $snap 'manifest.json') -Raw | ConvertFrom-Json
    Say "source        $($manifest.rehearsal_source)"
  }
  else { Say 'prepared      no' }
  if (Test-Path -LiteralPath (Join-Path $snap 'shadows.txt')) {
    foreach ($s in Read-Shadows) {
      Say "snapshot      $($s.Volume) $(if (Test-ShadowAlive $s) { 'present' } else { 'GONE -- post cannot roll back' })"
    }
  }
  # The raised cap outlives this backup if post never runs; say how to undo it.
  $storageFile = Join-Path $snap 'shadowstorage.txt'
  if (Test-Path -LiteralPath $storageFile) {
    foreach ($line in @(Get-Content -LiteralPath $storageFile | Where-Object { $_.Trim() })) {
      $f = $line -split "`t"
      $d = $f[0].TrimEnd('\')
      Say "snapshot room $($f[0]) was $(Format-Bytes ([UInt64]$f[1])) before pre (post puts it back; without post: vssadmin resize shadowstorage /for=$d /on=$d /maxsize=$($f[1]))"
    }
  }
  Say "data backup   $(if (Test-Path -LiteralPath (Join-Path $snap 'hermes-backup.zip')) { 'hermes-backup.zip' } else { 'none' })"
  Say "marker        $(if (Test-Path -LiteralPath (Join-Path $P.Home '.skip_upstream_prompt')) { 'present' } else { 'absent' })"
  $n = @(Invoke-GitCmd @('config', '--global', '--get-regexp', '^url\.')).Count + @(Invoke-GitCmd @('-C', $P.Install, 'config', '--local', '--get-regexp', '^url\.')).Count
  Say "git rewrites  $n insteadOf entr(y/ies)"
  if (Test-Path -LiteralPath (Join-Path $P.Install '.git')) {
    Say "checkout now  $(Invoke-Git $P @('rev-parse', '--short', 'HEAD')) ($(Invoke-Git $P @('branch', '--show-current')))"
  }
}

# ---------------------------------------------------------------------------
# post
# ---------------------------------------------------------------------------

function Confirm-Action {
  param([string]$Prompt)
  if ($Yes) { return }
  $reply = Read-Host "$Prompt [y/N]"
  if ($reply -notmatch '^(y|yes)$') { Fail 'aborted -- nothing was changed' }
}

function Invoke-Post {
  Load-Snapshot
  Assert-Elevated
  $P = Get-ResolvedPaths
  $manifest = Get-Content -LiteralPath (Join-Path $script:Snap 'manifest.json') -Raw | ConvertFrom-Json
  $shadows = Read-Shadows

  # Check EVERY snapshot before touching anything: half a rollback is worse than none.
  $gone = @($shadows | Where-Object { -not (Test-ShadowAlive $_) })
  if ($gone.Count) {
    $zip = Join-Path $script:Snap 'hermes-backup.zip'
    $gone | ForEach-Object { Warn "the snapshot of $($_.Volume) is gone (Windows deletes snapshots when the disk's shadow storage fills)" }
    Say ''
    Say 'Nothing was changed. An exact rollback is no longer possible, but your data is'
    Say "backed up in $zip. To put your config, keys, sessions, memories and"
    Say 'skills back, either:'
    Say "  hermes import `"$zip`""
    Say 'or, if hermes itself no longer starts, extract the zip over your home:'
    Say "  tar -xf `"$zip`" -C `"$($P.Home)`""
    Say 'then reinstall Hermes to get the program back.'
    Restore-ShadowStorage
    Fail 'the disk snapshot is gone'
  }

  Step 'this will restore:'
  Say "  $($P.Home)  (everything except cache\, including the checkout)"
  Say "  $($P.UserData)"
  Say "  to how they were at $($script:Snap)"
  Confirm-Action "Put everything back from $($script:Snap)?"

  Step 'stopping Hermes'
  foreach ($name in @('Hermes', 'hermes')) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
  Ok 'asked Hermes to stop (if anything was running)'

  $mounts = @{}
  try {
    foreach ($s in $shadows) { $mounts[$s.Volume] = Mount-Shadow $s }
    $trees = @([pscustomobject]@{ Label = 'HERMES_HOME'; Live = $P.Home; IsHome = $true })
    $trees += [pscustomobject]@{ Label = "the desktop app's data"; Live = $P.UserData; IsHome = $false }
    foreach ($t in $trees) {
      Step "restoring $($t.Label)"
      $vol = Get-VolumeRoot $t.Live
      if (-not $t.IsHome -and -not $manifest.userdata_existed) {
        # There was no desktop data at the snapshot: exact means none now either.
        if (Test-Path -LiteralPath $t.Live) { Remove-Item -LiteralPath $t.Live -Recurse -Force; Ok "removed $($t.Live) (it did not exist before)" }
        else { Ok 'nothing to restore (there was no desktop data before)' }
        continue
      }
      if (-not $mounts.ContainsKey($vol)) { Fail "no snapshot recorded for $vol (where $($t.Live) lives)" }
      $from = Join-Path $mounts[$vol] $t.Live.Substring(3)
      if (-not (Test-Path -LiteralPath $from)) { Fail "$($t.Live) is not in the snapshot of $vol" }
      $exclude = @()
      if ($t.IsHome) { $exclude = @((Join-Path $from 'cache'), (Join-Path $t.Live 'cache')) }
      $started = Get-Date
      $summary = Invoke-Mirror -From $from -To $t.Live -ExcludeDirs $exclude
      Ok "restored in $([int]((Get-Date) - $started).TotalSeconds)s"
      $summary | ForEach-Object { Write-Host "    $($_.Trim())" }
    }
  }
  finally {
    foreach ($link in $mounts.Values) { Dismount-Shadow $link }
  }

  Step 'removing the disk snapshot'
  foreach ($s in $shadows) { Remove-Shadow $s; Ok "deleted the snapshot of $($s.Volume)" }
  Restore-ShadowStorage

  Step 'done'
  Say "Your HERMES_HOME and the desktop app's data are back exactly as they were."
  Say "Open the desktop app once and run 'hermes doctor' to confirm."
  Say "Nothing was judged or changed by this script; the backup at $($script:Snap)"
  Say 'is yours to keep or delete.'
}

# ---------------------------------------------------------------------------

if (-not $BackupRoot) { $BackupRoot = Join-Path $env:USERPROFILE 'hermes-update-rehearsal' }

switch ($Command) {
  'pre' { Invoke-Pre }
  'post' { Invoke-Post }
  'status' { Invoke-Status }
  default {
    # $PSCommandPath is empty when the script was piped in rather than run
    # from a file, so fall back to a self-contained summary.
    if ($PSCommandPath) {
      Get-Help $PSCommandPath -Detailed | Out-String | Write-Host
    }
    else {
      Write-Host 'hermes-update-rehearsal.ps1 -- run against an EXISTING Hermes install.'
      Write-Host ''
      Write-Host '  pre     back up your data, snapshot the disk, point the update source at a fork'
      Write-Host '  post    restore both trees exactly as they were at the snapshot'
      Write-Host '  status  print what is prepared (read-only; nothing is touched)'
      Write-Host ''
      Write-Host 'pre and post need an elevated (Run as Administrator) PowerShell.'
      Write-Host ''
      Write-Host 'Options:'
      Write-Host '  -Source URL        repo to pull the update from; updates follow its main'
      Write-Host '  -BackupRoot DIR    where the backup lives'
      Write-Host '  -Yes               post: skip the confirmation'
      Write-Host ''
      Write-Host "Run 'pre' first: it reports what it did and prints the next commands."
    }
  }
}
