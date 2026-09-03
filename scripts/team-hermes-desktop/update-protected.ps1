[CmdletBinding()]
param(
  [string]$UpstreamRemote = 'origin',
  [string]$UpstreamBranch = 'main',
  [switch]$Apply,
  [switch]$SkipFetch,
  [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

  & git @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
  }
}

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Action
  )

  Write-Host "`n==> $Label"
  & $Action
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE"
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'apps\desktop\package.json') -PathType Leaf)) {
  throw "This command must run from a Hermes Agent source tree: $repoRoot"
}

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or -not $branch) {
  throw 'Unable to resolve the current Git branch.'
}
if ($branch -notlike 'bkash/team-hermes*') {
  throw "Refusing to update unprotected branch '$branch'. Use a bkash/team-hermes* branch."
}

$dirty = @(& git status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
  throw 'Unable to inspect the working tree.'
}
if ($dirty.Count -gt 0) {
  throw "Refusing to update a dirty working tree. Preserve or commit these files first:`n$($dirty -join "`n")"
}

$contractFiles = @(
  'apps/desktop/electron/protected-edition.ts',
  'apps/desktop/electron/protected-edition.test.ts',
  'apps/desktop/electron/team-hermes-edition.ts',
  'apps/desktop/electron/team-hermes-edition.test.ts',
  'apps/desktop/src/plugins/hermes-bots/roster-pane.presentation.test.ts',
  'apps/desktop/src/components/assistant-ui/thread/agent-exchange-card.test.tsx',
  'apps/desktop/src/components/assistant-ui/thread/process-notification-note.test.tsx',
  'docs/TEAM_HERMES_DESKTOP_MAINTENANCE.md'
)
$missing = @($contractFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $repoRoot $_) -PathType Leaf) })
if ($missing.Count -gt 0) {
  throw "Team Hermes contract files are missing:`n$($missing -join "`n")"
}

$desktopBuildRoot = Join-Path $env:LOCALAPPDATA 'hermes\desktop-builds'
$currentPointer = Join-Path $desktopBuildRoot 'current.json'
$pointerHashBefore = if (Test-Path -LiteralPath $currentPointer -PathType Leaf) {
  (Get-FileHash -Algorithm SHA256 -LiteralPath $currentPointer).Hash
} else {
  $null
}

if (-not $SkipFetch) {
  Invoke-Step "Fetch $UpstreamRemote/$UpstreamBranch" { Invoke-Git fetch --prune $UpstreamRemote $UpstreamBranch }
}

$targetRef = "$UpstreamRemote/$UpstreamBranch"
& git rev-parse --verify "$targetRef^{commit}" *> $null
if ($LASTEXITCODE -ne 0) {
  throw "Unable to resolve upstream target $targetRef."
}

$headBefore = (& git rev-parse HEAD).Trim()
$upstreamHead = (& git rev-parse "$targetRef^{commit}").Trim()
Invoke-Git merge-base --is-ancestor $headBefore HEAD

Write-Host "Team Hermes branch : $branch"
Write-Host "Current commit     : $headBefore"
Write-Host "Upstream target    : $upstreamHead ($targetRef)"
Write-Host "Protected pointer  : $currentPointer"

if (-not $Apply) {
  Write-Host "`nPreflight passed. No merge, build, package, launcher, or current.json change was made."
  Write-Host "Run again with -Apply to create a rollback ref, merge upstream, and verify the custom UI."
  exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$rollbackRef = "backup/team-hermes-pre-upstream-$stamp"
Invoke-Step "Create rollback ref $rollbackRef" { Invoke-Git branch $rollbackRef $headBefore }

try {
  Invoke-Step "Merge $targetRef without replacing the custom branch" {
    Invoke-Git merge --no-ff --no-edit $targetRef
  }
} catch {
  Write-Error "Upstream merge stopped. Resolve conflicts without deleting Team Hermes contracts, then rerun this command with -SkipFetch. Rollback ref: $rollbackRef"
  throw
}

Invoke-Step 'Check patch whitespace' { Invoke-Git diff --check "$headBefore..HEAD" }

Push-Location -LiteralPath (Join-Path $repoRoot 'apps\desktop')
try {
  Invoke-Step 'Run Team Hermes Electron identity/protection tests' {
    & npm exec vitest -- run --project electron electron/protected-edition.test.ts electron/team-hermes-edition.test.ts
  }
  Invoke-Step 'Run Team, Groups, cards, and process-flashcard UI tests' {
    & npm exec vitest -- run --project ui src/plugins/hermes-bots/roster-pane.presentation.test.ts src/plugins/hermes-bots/group-chat-view.test.ts src/plugins/hermes-bots/group-activity.test.ts src/components/assistant-ui/thread/agent-exchange-card.test.tsx src/components/assistant-ui/thread/assistant-message.test.tsx src/components/assistant-ui/thread/process-notification-note.test.tsx
  }
  Invoke-Step 'Type-check Desktop renderer, Electron, and E2E projects' { & npm run typecheck }
  if (-not $SkipBuild) {
    Invoke-Step 'Build Team Hermes Desktop' { & npm run build }
  }
} finally {
  Pop-Location
}

$pointerHashAfter = if (Test-Path -LiteralPath $currentPointer -PathType Leaf) {
  (Get-FileHash -Algorithm SHA256 -LiteralPath $currentPointer).Hash
} else {
  $null
}
if ($pointerHashBefore -ne $pointerHashAfter) {
  throw 'current.json changed during source update verification. The protected runtime pointer must remain untouched.'
}

Write-Host "`nVerified upstream merge while preserving the active protected build."
Write-Host "Rollback ref       : $rollbackRef"
Write-Host "Updated source HEAD : $((& git rev-parse HEAD).Trim())"
Write-Host 'Next gate: package into a new versioned desktop-builds directory, write its manifest, then live-test before switching current.json.'
