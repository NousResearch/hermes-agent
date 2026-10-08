<#
.SYNOPSIS
  Build the desktop MSIX for the commit checked out here, on this machine.

.DESCRIPTION
  scripts\build-bundle.ps1             Sideload MSIX (bundled variant) for this machine's architecture.
  scripts\build-bundle.ps1 -Store      Microsoft Store MSIX (Store identity).

  The commit does not need to be pushed. The checkout must be clean: the build
  packages HEAD. Output goes to apps\desktop\release\. The packages are unsigned.

  -Store needs a stable release tag. This script makes a local claim tag
  (rc.<N>-vX.Y.Z) for the next patch version, builds with it, and deletes it
  when the build ends. It never pushes the tag. Each architecture builds on its
  own native host. To combine x64 and arm64 Store packages into one bundle, copy
  both Store-*.msix files into one apps\desktop\release and run
  scripts\bundle-store-msixbundle.mjs.

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
  if ($Python) { return $Python }
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

# Previous builds leave read-only payload files, which a plain delete refuses.
function Remove-Output([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return }
  Write-Host "build-bundle: removing previous output $Path"
  & attrib -r "$Path\*" /s /d | Out-Null
  Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
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

# .cache stays: it holds downloaded tools and is safe to reuse.
foreach ($stale in '.build\desktop-job', 'apps\desktop\build', 'apps\desktop\dist', 'apps\desktop\release') {
  Remove-Output (Join-Path $Repo $stale)
}

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
    $env:RELEASE_CLAIM_TAG = $claim
    $env:RELEASE_CLAIM_OBJECT = $claimObject
    $env:HERMES_PAYLOAD_TAG = "v$version"
    $env:HERMES_PAYLOAD_VERSION = $version
    $env:HERMES_DESKTOP_VARIANT = 'bundled'
    Write-Host "build-bundle: building $sha as Store v$version (local claim $claim)"
    # Prepare as bundled, then package as store from that preparation, as CI does.
    $work = Join-Path $Repo '.build\desktop-job'
    & $py scripts/bundles/desktop.py --tag "v$version" --release-commit $sha --variant bundled --prepare-only `
      --work $work --cache (Join-Path $Repo '.cache\desktop-inputs')
    if ($LASTEXITCODE) { Fail "preparation failed (exit $LASTEXITCODE)" }
    & $py scripts/bundles/desktop.py --prepared (Join-Path $work 'prepared.json') --variant store
    if ($LASTEXITCODE) { Fail "Store packaging failed (exit $LASTEXITCODE)" }
  } else {
    Write-Host "build-bundle: building $sha (sideload)"
    & $py scripts/bundles/desktop.py --commit $sha --variant bundled
    if ($LASTEXITCODE) { Fail "build failed (exit $LASTEXITCODE)" }
  }
} finally {
  if ($claim) {
    # Claim tags are local only. Delete this one even when the build fails.
    & git tag -d $claim | Out-Null
  }
}

Write-Host "build-bundle: done. Unsigned packages in $Repo\apps\desktop\release:"
Get-ChildItem (Join-Path $Repo 'apps\desktop\release') -Filter '*.msix*' |
  ForEach-Object { '  {0}  {1:N0} MB' -f $_.Name, ($_.Length / 1MB) }
