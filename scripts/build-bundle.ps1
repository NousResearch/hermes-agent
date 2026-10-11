<#
.SYNOPSIS
  Build the desktop MSIX for the commit checked out here, on this machine.

.DESCRIPTION
  scripts\build-bundle.ps1             Sideload MSIX (bundled variant) for this machine's architecture.
  scripts\build-bundle.ps1 -Store      Microsoft Store MSIX (Store identity).

  The commit does not need to be pushed. The checkout must be clean: the build
  packages HEAD. Output goes to apps\desktop\release\. The script removes the
  AZURE_SIGN_* variables from the build, so the packages are unsigned even when
  your shell has release signing set.

  -Store needs a stable release tag. This script makes a local claim tag
  (rc.<N>-vX.Y.Z) for the next patch version, builds with it, and deletes it
  when the build ends. It never pushes the tag. Each architecture builds on its
  own native host. The script prints the claim tag's timestamp as
  HERMES_RELEASE_EPOCH. To combine the x64 and arm64 Store packages, copy both
  Store-*.msix files into one apps\desktop\release, set the same
  HERMES_RELEASE_EPOCH on that host, and run
  node scripts\bundle-store-msixbundle.mjs --tag vX.Y.Z.
  Both hosts must use the epoch from ONE claim, or the package versions differ.

  Use a short checkout path such as C:\hsb. On Windows ARM64, a long path makes
  the cryptography build fail with LNK1104.

.PARAMETER Store
  Build the Store package instead of the sideload package.

.PARAMETER Python
  Python 3.11+ to run the driver. Default: the first suitable python on PATH.

.PARAMETER Remote
  Remote that lists published release tags and attempt refs. Default: origin.
#>
[CmdletBinding()]
param(
  [switch]$Store,
  [string]$Python = '',
  [string]$Remote = 'origin'
)

# 'Stop' turns the native stderr progress lines of the build into terminating errors on Windows PowerShell 5.
$ErrorActionPreference = 'Continue'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
# Attempt numbers from here up cannot collide with a real release attempt.
$LocalAttemptFloor = 900

function Fail([string]$Message) {
  [Console]::Error.WriteLine("build-bundle: $Message")
  exit 1
}

function Find-Python {
  if ($Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { Fail "-Python '$Python' is not a file." }
    return $Python
  }
  foreach ($name in 'python', 'python3') {
    foreach ($command in @(Get-Command $name -CommandType Application -ErrorAction SilentlyContinue)) {
      # The WindowsApps python.exe is a Store stub. Over ssh it fails with "Access is denied".
      if ($command.Source -like '*\WindowsApps\*') { continue }
      & $command.Source -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>$null
      if ($LASTEXITCODE -eq 0) { return $command.Source }
    }
  }
  Fail 'needs Python 3.11+ on PATH (or pass -Python).'
}

function Git-Lines {
  $lines = & git @args
  if ($LASTEXITCODE) { Fail "git $($args -join ' ') failed" }
  return $lines | Where-Object { $_ }
}

# Next patch after the newest published stable tag (vX.Y.Z, not CalVer or canary).
function Get-ClaimVersion {
  $latest = $null
  foreach ($line in @(Git-Lines ls-remote --tags $Remote 'refs/tags/v*')) {
    if ($line -notmatch 'refs/tags/v((?:0|[1-9]\d{0,2})\.\d+\.\d+)$') { continue }
    $version = [version]$Matches[1]
    if ($null -eq $latest -or $version -gt $latest) { $latest = $version }
  }
  if ($null -eq $latest) { Fail "no stable release tag found on remote '$Remote' ($(git remote get-url $Remote)). Pass -Remote with the remote that points at NousResearch/hermes-agent." }
  return "$($latest.Major).$($latest.Minor).$($latest.Build + 1)"
}

# One past the highest attempt for this version, counting local and remote refs.
function Get-ClaimAttempt([string]$Version) {
  $refs = @(Git-Lines tag --list "rc.*-v$Version") + @(Git-Lines ls-remote --tags $Remote "refs/tags/rc.*-v$Version")
  $highest = $LocalAttemptFloor - 1
  foreach ($ref in $refs) {
    if ($ref -match "rc\.([1-9]\d*)-v$([regex]::Escape($Version))$" -and [int]$Matches[1] -gt $highest) {
      $highest = [int]$Matches[1]
    }
  }
  return $highest + 1
}

Set-Location $Repo
$py = Find-Python

if (git status --porcelain --untracked-files=all) {
  [Console]::Error.WriteLine('build-bundle: the checkout has uncommitted changes. Commit them (the build packages HEAD) or stash them.')
  git status --short --untracked-files=all
  exit 1
}
$sha = @(Git-Lines rev-parse HEAD)[0]

if ($Repo.Length -gt 40) {
  Write-Warning "checkout path is $($Repo.Length) characters. On Windows ARM64 a long path can fail the cryptography build (LNK1104). Prefer C:\hsb."
}

# Release settings and signing credentials exist only for this build. Save the caller's
# values and put them back, so a later direct build command sees an untouched session.
$buildVariables = @('PYTHONUTF8', 'RELEASE_CLAIM_TAG', 'RELEASE_CLAIM_OBJECT', 'HERMES_PAYLOAD_TAG',
  'HERMES_PAYLOAD_VERSION', 'HERMES_DESKTOP_VARIANT') +
  @(Get-ChildItem Env: | Where-Object { $_.Name -like 'AZURE_*' } | ForEach-Object { $_.Name })
$savedEnvironment = @{}
foreach ($name in $buildVariables) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
foreach ($name in $buildVariables | Where-Object { $_ -like 'AZURE_*' }) { Remove-Item "Env:$name" }
$env:PYTHONUTF8 = '1'
$claim = $null
try {
  if ($Store) {
    $version = Get-ClaimVersion
    $claim = "rc.$(Get-ClaimAttempt $version)-v$version"
    # Pass -c for identity and signing so a machine without a git identity, or with signed tags on, still works.
    & git -c user.name=hermes-local-build -c user.email=local-build@invalid -c tag.gpgSign=false `
      tag -a $claim -m 'local build claim' $sha
    if ($LASTEXITCODE) { Fail "could not create claim tag $claim" }
    $claimObject = @(Git-Lines rev-parse "refs/tags/$claim")[0]
    # Every architecture's Store package derives its 4-part version from this one timestamp.
    $taggerLine = @(@(Git-Lines cat-file -p $claimObject) | Where-Object { $_ -match '^tagger ' })[0]
    if ($taggerLine -notmatch ' (\d+) [+-]\d{4}$') { Fail "claim tag $claim has no tagger timestamp" }
    $claimEpoch = [int64]$Matches[1]
    $env:RELEASE_CLAIM_TAG = $claim
    $env:RELEASE_CLAIM_OBJECT = $claimObject
    $env:HERMES_PAYLOAD_TAG = "v$version"
    $env:HERMES_PAYLOAD_VERSION = $version
    $env:HERMES_DESKTOP_VARIANT = 'bundled'
    Write-Host "build-bundle: building $sha as Store v$version (local claim $claim)"
    # Prepare as bundled, then package as store from that preparation, as CI does.
    $work = Join-Path $Repo '.build\desktop-job'
    & $py scripts/bundles/desktop.py --tag "v$version" --release-commit $sha --variant bundled --prepare-only `
      --work $work --cache (Join-Path $Repo '.cache\desktop-inputs') --clean
    if ($LASTEXITCODE) { Fail "preparation failed (exit $LASTEXITCODE)" }
    & $py scripts/bundles/desktop.py --prepared (Join-Path $work 'prepared.json') --variant store
    if ($LASTEXITCODE) { Fail "Store packaging failed (exit $LASTEXITCODE)" }
  } else {
    Write-Host "build-bundle: building $sha (sideload)"
    & $py scripts/bundles/desktop.py --commit $sha --variant bundled --clean
    if ($LASTEXITCODE) { Fail "build failed (exit $LASTEXITCODE)" }
  }
} finally {
  foreach ($name in $buildVariables) {
    [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
  }
  if ($claim) {
    # Claim tags are local only. Delete this one even when the build fails.
    & git tag -d $claim | Out-Null
  }
}

Write-Host "build-bundle: done. Unsigned packages in $Repo\apps\desktop\release:"
if ($Store) {
  Write-Host "build-bundle: claim timestamp for combining architectures: HERMES_RELEASE_EPOCH=$claimEpoch (version v$version)"
}
Get-ChildItem (Join-Path $Repo 'apps\desktop\release') -Filter '*.msix*' |
  ForEach-Object { '  {0}  {1:N0} MB' -f $_.Name, ($_.Length / 1MB) }
