# Bootstrap AI employee profiles + kanban board for Hermes.
# Usage: .\setup-ai-employees.ps1 [-Clone] [-BoardSlug ai-company]

param(
    [switch]$Clone,
    [string]$BoardSlug = "ai-company"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TemplateDir = Join-Path (Split-Path $ScriptDir -Parent) "templates"

$Profiles = @(
    @{
        Name = "secretary"
        Description = "Orchestrator: triage, decompose, schedule, human handoff."
        Soul = "SOUL-secretary.md"
    },
    @{
        Name = "job-recruiter"
        Description = "Creates and publishes job postings; tracks applicants."
        Soul = "SOUL-job-recruiter.md"
    },
    @{
        Name = "job-seeker"
        Description = "Finds roles, drafts applications, tracks pipeline."
        Soul = "SOUL-job-seeker.md"
    },
    @{
        Name = "self-improver"
        Description = "Reviews skills, memory, failures; proposes improvements."
        Soul = "SOUL-self-improver.md"
    },
    @{
        Name = "delivery-worker"
        Description = "Executes contracted deliverables end-to-end."
        Soul = "SOUL-delivery-worker.md"
    }
)

function Invoke-HermesCli {
    param([string[]]$HermesArgs)
    & hermes @HermesArgs
    if ($LASTEXITCODE -ne 0) {
        throw "hermes failed: hermes $($HermesArgs -join ' ')"
    }
}

Write-Host "== AI Employee Org setup ==" -ForegroundColor Cyan

foreach ($p in $Profiles) {
  $exists = $false
  try {
    Invoke-HermesCli @("profile", "show", $p.Name) 2>$null | Out-Null
    $exists = $true
  } catch {
    $exists = $false
  }

  if (-not $exists) {
    $createArgs = @("profile", "create", $p.Name, "--description", $p.Description)
    if ($Clone) { $createArgs += "--clone" }
    Write-Host "Creating profile $($p.Name) ..."
    Invoke-HermesCli $createArgs
  } else {
    Write-Host "Profile $($p.Name) already exists — skipping create."
  }

  $homeRoot = Join-Path $env:USERPROFILE ".hermes\profiles\$($p.Name)"
  $soulDest = Join-Path $homeRoot "SOUL.md"
  $soulSrc = Join-Path $TemplateDir $p.Soul
  if (Test-Path $soulSrc) {
    Copy-Item -Path $soulSrc -Destination $soulDest -Force
    Write-Host "  Updated SOUL.md"
  }
}

Write-Host "Initializing kanban board '$BoardSlug' ..."
try {
  Invoke-HermesCli @("kanban", "boards", "create", $BoardSlug, "--name", "AI Company", "--switch")
} catch {
  Write-Host "  Board may already exist — continuing."
}

Invoke-HermesCli @("kanban", "init")

Write-Host @"

Next steps:
  1. Set kanban.orchestrator_profile: secretary in ~/.hermes/config.yaml
  2. hermes gateway run  (dispatcher owner)
  3. hermes skills install official/autonomous-ai-agents/ai-employee-org
  4. hermes kanban create "Smoke test" --assignee secretary

"@ -ForegroundColor Green
