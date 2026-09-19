<#
.SYNOPSIS
  hermes-update-rehearsal.ps1 -- for an EXISTING Hermes install (Windows).

.DESCRIPTION
  Two steps:

    pre   back up your ENTIRE HERMES_HOME and the desktop app's Electron userData,
          then point the install's update source at a custom repo + ref so
          `hermes update` pulls it. Prints what to do next.
    post  wipe both trees and put the backup back exactly as it was.

  Plus `status`, which only prints. This script never judges your install: it
  reports what it did and stops. Whether the update worked is yours to see.

  The backup is one plain tar per tree with nothing filtered out, and `post`
  restores those tars over empty directories, so every file comes back as it was.

.PARAMETER Command
  pre | post | status

.PARAMETER Source
  Repo to pull the update from (default: the rehearsal fork).

.PARAMETER Ref
  Branch or tag in that repo (default: main).

.PARAMETER BackupRoot
  Where the backup lives (default: $HOME\hermes-update-rehearsal).

.PARAMETER Yes
  post: skip the confirmation.

.EXAMPLE
  ./hermes-update-rehearsal.ps1 pre --source <git-url> --ref <branch>
  # ... run `hermes update`, use Hermes, test ...
  ./hermes-update-rehearsal.ps1 post

.NOTES
  `pre` needs network access to -Source and a usable git on PATH.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('pre', 'post', 'status', 'help')]
  [string]$Command = 'help',

  [string]$Source = 'https://github.com/ethernet8023/hermes-agent.git',
  [string]$Ref = 'main',
  [string]$BackupRoot,
  [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$OfficialHttps = 'https://github.com/NousResearch/hermes-agent.git'
$OfficialSsh = 'git@github.com:NousResearch/hermes-agent.git'

$script:Snap = ''
$script:Tar = $null

function Say  { param([string]$m) Write-Host $m }
function Ok   { param([string]$m) Write-Host "  OK $m" }
function Warn { param([string]$m) Write-Warning "  $m" }
function Step { param([string]$m) Write-Host "`n=== $m ===" }
function Fail { param([string]$m) throw "ERROR: $m" }

function Resolve-Tar {
  # A PATH pointing into a WindowsApps payload (the bundled app's toolchain)
  # yields a tar.exe PowerShell cannot launch, so prefer the real one.
  $cands = @()
  if ($env:SystemRoot) {
    $cands += (Join-Path $env:SystemRoot 'System32\tar.exe')
    $cands += (Join-Path $env:SystemRoot 'Sysnative\tar.exe')
  }
  $onPath = Get-Command tar.exe -ErrorAction SilentlyContinue
  if ($onPath) { $cands += $onPath.Source }
  foreach ($c in $cands) {
    if (-not $c) { continue }
    if ($c -like '*\Microsoft\WindowsApps\*') { continue }
    if (-not (Test-Path -LiteralPath $c)) { continue }
    try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { return $c } }
    catch { }
  }
  return $null
}

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
function Get-ResolvedPaths {
  $suffix = if ($env:HERMES_DATA_DIR_SUFFIX) { $env:HERMES_DATA_DIR_SUFFIX } else { '' }
  $userProfile = $env:USERPROFILE
  if ($env:HERMES_HOME) {
    # Must tolerate a MISSING home: post resolves paths in order to recreate them.
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
# pre
# ---------------------------------------------------------------------------

function Invoke-BackupTar {
  param([string]$TarExe, [string[]]$TarArgs, [string]$Root, [string]$Label)


  $old = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  $output = & $TarExe @TarArgs 2>&1
  $ErrorActionPreference = $old

  if ($LASTEXITCODE -eq 0) { return }

  $denied = @($output | Where-Object {
    $_ -match 'Permission denied|Access is denied'
  })
  if ($denied) {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    Warn "tar could not read $($denied.Count) path(s) under $Root`:" 
    $denied | ForEach-Object { Write-Host "    $_" }
    Say ''
    Say 'run this in an elevated (Run as Administrator) PowerShell, then re-run pre:'
    Say "  icacls `"$Root`" /grant `"$user`:(OI)(CI)F`" /t /c"
    Fail "permission errors backing up $Label -- fix with the command above, then re-run"
  }

  # non-permission failure: show what tar said and fail generically
  $output | ForEach-Object { Write-Host "    $_" }
  Fail "backup failed (tar) for $Label"
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

  Step 'before we start (nothing here is pass/fail, just read it)'
  $procs = @(Get-Process -Name 'Hermes', 'hermes' -ErrorAction SilentlyContinue)
  if ($procs.Count) {
    Warn 'Hermes looks like it is running -- close the desktop app and the gateway'
    Warn "before you run 'hermes update', or the dependency sync may fail:"
    $procs | ForEach-Object { Write-Host "    $($_.ProcessName) (pid $($_.Id))" }
  }
  else { Ok 'no Hermes processes running' }
  $n = @(Invoke-GitCmd @('config', '--global', '--get-regexp', '^url\.')).Count
  if ($n -eq 0) { Ok 'global git config has no URL rewrites' }
  else { Warn "$n existing url.* insteadOf entr(y/ies) in your git config; we add more and remove only ours" }

  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
  $script:Snap = Join-Path $BackupRoot $stamp
  if (Test-Path -LiteralPath $script:Snap) { Fail "backup dir already exists: $($script:Snap)" }
  New-Item -ItemType Directory -Force -Path $script:Snap | Out-Null

  Step 'backing up your entire HERMES_HOME'
  $started = Get-Date
  $homeTar = Join-Path $script:Snap 'hermes-home.tar'
  # No excludes: checkout, venv, PM store and node_modules come too, so post is
  # a true rollback rather than a re-download. SQLite sidecars travel WITH their
  # db on purpose (a raw copy of db+wal+shm is consistent). No -z: the backup
  # root is typically the same internal disk, so gzip costs ~5x the wall time
  # for nothing (measured on an M1).
  Invoke-BackupTar -TarExe $script:Tar -TarArgs @('-cf', $homeTar, '-C', $P.Home, '.') -Root $P.Home -Label 'HERMES_HOME'
  $elapsed = [int]((Get-Date) - $started).TotalSeconds
  Ok "hermes-home.tar ($([math]::Round((Get-Item $homeTar).Length / 1MB, 1)) MB, ${elapsed}s)"

  Step "backing up the desktop app's data"
  if (Test-Path -LiteralPath $P.UserData) {
    $udTar = Join-Path $script:Snap 'electron-userdata.tar'
    Invoke-BackupTar -TarExe $script:Tar -TarArgs @('-cf', $udTar, '-C', $P.UserData, '.') -Root $P.UserData -Label 'Electron userData'
    Ok "electron-userdata.tar ($([math]::Round((Get-Item $udTar).Length / 1KB, 1)) KB)"
  }
  else { Warn "no Electron userData at $($P.UserData) (desktop app not installed?)" }

  Step 'recording what this backup is'
  # Plain text, not JSON: post compares this string byte-for-byte to decide
  # whether the backup belongs to the home it is about to wipe.
  Set-Content -LiteralPath (Join-Path $script:Snap 'hermes-home.txt') -Encoding utf8 -Value @($P.Home)
  $manifest = [ordered]@{
    schema              = 3
    created             = (Get-Date).ToUniversalTime().ToString('o')
    hermes_home         = $P.Home
    install_dir         = $P.Install
    userdata_dir        = $P.UserData
    userdata_dir_source = $P.UserDataOrigin
    rehearsal_source    = $Source
    rehearsal_ref       = $Ref
  }
  $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:Snap 'manifest.json') -Encoding utf8
  Ok "manifest.json (backup of $($P.Home))"

  # --- point the install at the rehearsal source ---------------------------
  Step 'fetching the rehearsal source'
  Say "source        $Source"
  Say "ref           $Ref"
  $serve = Join-Path $script:Snap 'serve.git'
  if (Test-Path -LiteralPath $serve) { Remove-Item -LiteralPath $serve -Recurse -Force }
  $null = Invoke-GitCmd @('clone', '--quiet', '--bare', '--branch', $Ref, '--single-branch', $Source, $serve)
  if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath $serve) { Remove-Item -LiteralPath $serve -Recurse -Force }
    $null = Invoke-GitCmd @('clone', '--quiet', '--bare', $Source, $serve)
    if ($LASTEXITCODE -ne 0) { Fail "could not clone $Source (network? permissions? bad -Source?)" }
    Ok "cloned the whole repo (-Ref '$Ref' is not a branch/tag name)"
  }
  else { Ok "cloned $Ref" }
  $targetSha = (@(& git -C $serve rev-parse --verify "$Ref^{commit}" 2>$null | Out-String) -replace "`r", '').Trim()
  # Shape-check: a warning or error line folded into the capture would otherwise
  # be handed to update-ref as a bogus revision.
  if ($targetSha -notmatch '^[0-9a-f]{40}$') { Fail "-Ref '$Ref' was not found in $Source" }
  $null = Invoke-GitCmd @('-C', $serve, 'update-ref', 'refs/heads/main', $targetSha)
  $null = Invoke-GitCmd @('-C', $serve, 'symbolic-ref', 'HEAD', 'refs/heads/main')
  $null = Invoke-GitCmd @('-C', $serve, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
  Ok "the update will land on $targetSha"

  Step 'pointing your install at it'
  # insteadOf is a TRANSPORT rewrite. Your checkout's origin keeps the official
  # URL, which matters: `hermes update` resolves its channel from the archive and
  # validates it against `git config --get remote.origin.url`. Repointing origin
  # at a fork would make the update fail before any git work.
  $fileUrl = 'file:///' + ($serve -replace '\\', '/')
  # REPO-LOCAL, like the POSIX script: the checkout's own config lives inside the
  # home this kit backs up, so post's wipe+restore removes it for free and no
  # writable GLOBAL git config is needed.
  foreach ($url in @($OfficialHttps, $OfficialSsh)) {
    # --add: the key is multi-valued; a plain set would drop the first URL.
    $null = Invoke-GitCmd @('-C', $P.Install, 'config', '--local', '--add', "url.$fileUrl.insteadOf", $url)
    if ($LASTEXITCODE -ne 0) { Fail "could not write the URL redirect into $($P.Install)\.git\config" }
  }
  New-Item -ItemType Directory -Force -Path $P.Home | Out-Null
  Set-Content -LiteralPath (Join-Path $P.Home '.skip_upstream_prompt') -Encoding utf8 -Value @()
  Set-Content -LiteralPath (Join-Path $script:Snap 'target-sha') -Encoding utf8 -Value @($targetSha)
  Ok 'official repo URL now resolves to the rehearsal copy'
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
  Say "latest        $snap"
  $ts = Join-Path $snap 'target-sha'
  if (Test-Path -LiteralPath $ts) {
    Say "prepared for  $((Get-Content -LiteralPath $ts -Raw).Trim())"
    $manifest = Get-Content -LiteralPath (Join-Path $snap 'manifest.json') -Raw | ConvertFrom-Json
    Say "source        $($manifest.rehearsal_source) @ $($manifest.rehearsal_ref)"
  }
  else { Say 'prepared      no' }
  Say "marker        $(if (Test-Path -LiteralPath (Join-Path $P.Home '.skip_upstream_prompt')) { 'present' } else { 'absent' })"
  $n = @(Invoke-GitCmd @('config', '--global', '--get-regexp', '^url\.')).Count
  Say "git rewrites  $n global insteadOf entr(y/ies)"
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

function Remove-StaleGlobalRedirect {
  # An earlier version of this kit wrote the insteadOf redirect into the GLOBAL
  # git config. Those entries name THIS snapshot's serve.git, which post leaves
  # behind, so they would keep hijacking `hermes update` forever. Remove only
  # the entries that point at our own rehearsal copy.
  $serve = Join-Path $script:Snap 'serve.git'
  $prefix = 'file:///' + ($serve -replace '\\', '/')
  $removed = 0
  foreach ($url in @($OfficialHttps, $OfficialSsh)) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      $out = & git config --global --get "url.$prefix.insteadOf" 2>$null
      if ($LASTEXITCODE -eq 0 -and $out) {
        $null = Invoke-GitCmd @('config', '--global', '--unset-all', "url.$prefix.insteadOf")
        $removed++
      }
    }
    finally { $ErrorActionPreference = $prev }
  }
  if ($removed) { Ok "removed $removed stale global URL redirect(s) from an older run of this kit" }
}

function Invoke-Post {
  Load-Snapshot
  $P = Get-ResolvedPaths
  Step 'this will delete and restore:'
  Say "  $($P.Home)  (all of it, including the checkout)"
  Say "  $($P.UserData)"
  Say "  from $($script:Snap)"
  Confirm-Action "Put everything back from $($script:Snap)?"

  Step 'stopping Hermes'
  foreach ($name in @('Hermes', 'hermes')) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
  Ok 'asked Hermes to stop (if anything was running)'

  Step 'clearing both trees'
  if (Test-Path -LiteralPath $P.Home) { Remove-Item -LiteralPath $P.Home -Recurse -Force; Ok "removed $($P.Home)" }
  if (Test-Path -LiteralPath $P.UserData) { Remove-Item -LiteralPath $P.UserData -Recurse -Force; Ok "removed $($P.UserData)" }

  Step 'restoring your HERMES_HOME'
  New-Item -ItemType Directory -Force -Path $P.Home | Out-Null
  & $script:Tar -xf (Join-Path $script:Snap 'hermes-home.tar') -C $P.Home
  if ($LASTEXITCODE -ne 0) { Fail "restore failed -- your backup is intact at $($script:Snap)" }
  Ok 'restored'

  Step "restoring the desktop app's data"
  $udTar = Join-Path $script:Snap 'electron-userdata.tar'
  if (Test-Path -LiteralPath $udTar) {
    New-Item -ItemType Directory -Force -Path $P.UserData | Out-Null
    & $script:Tar -xf $udTar -C $P.UserData
    if ($LASTEXITCODE -ne 0) { Fail "userData restore failed (backup intact at $($script:Snap))" }
    Ok 'restored'
  }
  else { Warn 'there was no desktop app data to restore' }

  Remove-StaleGlobalRedirect

  Step 'done'
  Say "Your HERMES_HOME and the desktop app's data are back exactly as they were."
  Say "Open the desktop app once and run 'hermes doctor' to confirm."
  Say "Nothing was judged or changed by this script; the backup at $($script:Snap)"
  Say 'is yours to keep or delete.'
}

# ---------------------------------------------------------------------------

if (-not $BackupRoot) { $BackupRoot = Join-Path $env:USERPROFILE 'hermes-update-rehearsal' }

if ($Command -ne 'help') {
  $script:Tar = Resolve-Tar
  if (-not $script:Tar) { Fail 'no usable tar.exe found; install the Windows tar or put it on PATH' }
}

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
      Write-Host '  pre     back up everything, then point the update source at a custom repo+ref'
      Write-Host '  post    wipe both trees and restore the backup exactly as it was'
      Write-Host '  status  print what is prepared (read-only; nothing is touched)'
      Write-Host ''
      Write-Host 'Options:'
      Write-Host '  -Source URL        repo to pull the update from'
      Write-Host '  -Ref REV           branch or tag in that repo'
      Write-Host '  -BackupRoot DIR    where the backup lives'
      Write-Host '  -Yes               post: skip the confirmation'
      Write-Host ''
      Write-Host "Run 'pre' first: it reports what it did and prints the next commands."
    }
  }
}
