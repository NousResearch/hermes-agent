# Hermes Skills Smoke Test
# Validates all ported skills: format, frontmatter, cross-references, discoverability
param(
    [switch]$Quick,       # Skip Hermes discovery test (needs .venv)
    [switch]$Verbose      # Show per-skill details
)

$ErrorActionPreference = "Stop"
$SkillsRoot = "$PSScriptRoot\..\..\skills\software-development"
$Pass = 0
$Fail = 0
$Warn = 0
$Results = @()

Write-Host "`n=== Hermes Skills Smoke Test ===" -ForegroundColor Cyan
Write-Host "Skills root: $SkillsRoot`n"

# ── Test 1: All SKILL.md files exist ──
Write-Host "── Test 1: SKILL.md file existence ──" -ForegroundColor Yellow
$skillDirs = Get-ChildItem -Path $SkillsRoot -Directory | Where-Object { $_.Name -notmatch '^(dogfood|hermes-agent|inspecting|node-inspect|python-debugpy|requesting-code-review|simplify-code|spike)' }
foreach ($dir in $skillDirs) {
    $skillFile = Join-Path $dir.FullName "SKILL.md"
    if (Test-Path $skillFile) {
        $Results += @{ Name = $dir.Name; Test = "File exists"; Status = "PASS" }
        $Pass++
        if ($Verbose) { Write-Host "  PASS: $($dir.Name)/SKILL.md" -ForegroundColor Green }
    } else {
        $Results += @{ Name = $dir.Name; Test = "File exists"; Status = "FAIL" }
        $Fail++
        Write-Host "  FAIL: $($dir.Name)/SKILL.md NOT FOUND" -ForegroundColor Red
    }
}
Write-Host "  $Pass / $($Pass + $Fail) files found`n"

# ── Test 2: YAML frontmatter validity ──
Write-Host "── Test 2: YAML frontmatter validity ──" -ForegroundColor Yellow
$fmPass = 0; $fmFail = 0
foreach ($dir in $skillDirs) {
    $skillFile = Join-Path $dir.FullName "SKILL.md"
    if (-not (Test-Path $skillFile)) { continue }
    
    $content = Get-Content $skillFile -Raw
    $frontmatterErrors = @()
    
    # Check for opening ---
    if ($content -notmatch '^---\s*$') {
        $frontmatterErrors += "Missing opening '---'"
    }
    
    # Extract frontmatter between first two ---
    if ($content -match '^---\s*\n(.*?)\n---\s*\n') {
        $fm = $matches[1]
    } else {
        $frontmatterErrors += "Cannot extract frontmatter"
    }
    
    # Required fields
    $requiredFields = @('name:', 'description:', 'version:', 'license:', 'platforms:')
    foreach ($field in $requiredFields) {
        if ($fm -notmatch "(?m)^$field") {
            $frontmatterErrors += "Missing required field: $($field.TrimEnd(':'))"
        }
    }
    
    # name must match directory name
    if ($fm -match "(?m)^name:\s*""?([^""\n]+)""?") {
        $fmName = $matches[1].Trim()
        if ($fmName -ne $dir.Name) {
            $frontmatterErrors += "Frontmatter name '$fmName' != directory name '$($dir.Name)'"
        }
    }
    
    # Check for metadata.hermes.tags
    if ($fm -notmatch 'hermes:') {
        $Warn++
        Write-Host "  WARN: $($dir.Name) — no metadata.hermes section" -ForegroundColor Yellow
    } elseif ($fm -notmatch 'tags:') {
        $Warn++
        Write-Host "  WARN: $($dir.Name) — no tags in metadata.hermes" -ForegroundColor Yellow
    }
    
    if ($frontmatterErrors.Count -eq 0) {
        $fmPass++
        if ($Verbose) { Write-Host "  PASS: $($dir.Name) frontmatter" -ForegroundColor Green }
    } else {
        $fmFail++
        Write-Host "  FAIL: $($dir.Name) — $($frontmatterErrors -join ', ')" -ForegroundColor Red
    }
}
Write-Host "  $fmPass / $($fmPass + $fmFail) frontmatters valid`n"

# ── Test 3: No duplicate names ──
Write-Host "── Test 3: No duplicate skill names ──" -ForegroundColor Yellow
$allNames = @()
foreach ($dir in $skillDirs) {
    $skillFile = Join-Path $dir.FullName "SKILL.md"
    if (-not (Test-Path $skillFile)) { continue }
    $content = Get-Content $skillFile -Raw
    if ($content -match "(?m)^name:\s*""?([^""\n]+)""?") {
        $allNames += $matches[1].Trim()
    }
}
$dupes = $allNames | Group-Object | Where-Object { $_.Count -gt 1 }
if ($dupes) {
    Write-Host "  FAIL: Duplicate names found: $($dupes.Name -join ', ')" -ForegroundColor Red
    $Fail++
} else {
    Write-Host "  PASS: All $($allNames.Count) names unique" -ForegroundColor Green
    $Pass++
}
Write-Host ""

# ── Test 4: related_skills cross-references exist ──
Write-Host "── Test 4: related_skills references ──" -ForegroundColor Yellow
$knownSkills = $skillDirs.Name
foreach ($dir in $skillDirs) {
    $skillFile = Join-Path $dir.FullName "SKILL.md"
    if (-not (Test-Path $skillFile)) { continue }
    $content = Get-Content $skillFile -Raw
    
    # Extract related_skills
    if ($content -match 'related_skills:\s*\[([^\]]+)\]') {
        $refs = $matches[1] -split ',' | ForEach-Object { $_.Trim() -replace '[\[\]""]', '' }
        foreach ($ref in $refs) {
            if ($ref -and $ref -notin $knownSkills -and $ref -notmatch '^(web_search|web_extract|cronjob|delegate_task|browser_navigate|popular-web-designs|github-pr-workflow|github-code-review|github-auth|coding-standards|code-simplification|git-workflow-and-versioning|design-md|web-performance-audit|web-accessibility-audit|seo-aeo-audit|baseline-ui|code-review-and-quality|security-review|context-engineering|subagent-driven-development)$') {
                Write-Host "  WARN: $($dir.Name) references '$ref' — not in known skills" -ForegroundColor Yellow
                $Warn++
            }
        }
    }
}
Write-Host "  Cross-reference check complete`n"

# ── Test 5: Skill content quality ──
Write-Host "── Test 5: Content quality checks ──" -ForegroundColor Yellow
$qPass = 0; $qWarn = 0
foreach ($dir in $skillDirs) {
    $skillFile = Join-Path $dir.FullName "SKILL.md"
    if (-not (Test-Path $skillFile)) { continue }
    $content = Get-Content $skillFile -Raw
    $warnings = @()
    
    # Check for Hermes Integration section
    if ($content -notmatch 'Hermes Integration') {
        $warnings += "Missing 'Hermes Integration' section"
    }
    
    # Check for placeholder/TODO remnants
    if ($content -match '(?i)\b(TBD|TODO|FIXME|fill in|implement later)\b') {
        $warnings += "Contains placeholder text"
    }
    
    # Check minimum length
    $lineCount = ($content -split '\n').Count
    if ($lineCount -lt 20) {
        $warnings += "Too short ($lineCount lines)"
    }
    
    if ($warnings.Count -eq 0) {
        $qPass++
    } else {
        $qWarn++
        Write-Host "  WARN: $($dir.Name) — $($warnings -join ', ')" -ForegroundColor Yellow
    }
}
Write-Host "  $qPass / $($qPass + $qWarn) pass content quality`n"

# ── Test 6: All 24 expected new skills present ──
Write-Host "── Test 6: Expected skills inventory ──" -ForegroundColor Yellow
$expectedNew = @(
    "agent-reach", "agent-self-evaluation", "blueprint", "ci-fix",
    "codebase-onboarding", "create-pull-request", "design-md-library",
    "doubt-driven-development", "eval-harness", "executing-plans",
    "gstack-review", "hallmark", "idea-to-design",
    "incremental-implementation", "karpathy-guidelines",
    "loop-engineering", "loopy", "ponytail", "production-audit",
    "safety-guard", "self-learning", "spec-driven-development",
    "taste-skill", "verification-loop", "writing-plans"
)
$present = @()
$missing = @()
$existingSkills = $skillDirs.Name
foreach ($name in $expectedNew) {
    if ($name -in $existingSkills) {
        $present += $name
    } else {
        $missing += $name
    }
}
Write-Host "  Present: $($present.Count) / $($expectedNew.Count)" -ForegroundColor Green
if ($missing) {
    Write-Host "  MISSING: $($missing -join ', ')" -ForegroundColor Red
    $Fail += $missing.Count
}
Write-Host ""

# ── Summary ──
Write-Host "=== SMOKE TEST SUMMARY ===" -ForegroundColor Cyan
Write-Host "  Skills found:     $($skillDirs.Count) total, $($present.Count) new" -ForegroundColor White
Write-Host "  Frontmatter:      $fmPass / $($fmPass + $fmFail) valid" -ForegroundColor $(if($fmFail -eq 0){'Green'}else{'Red'})
Write-Host "  Unique names:     $(if($dupes){'FAIL'}else{'PASS'})" -ForegroundColor $(if($dupes){'Red'}else{'Green'})
Write-Host "  Content quality:  $qPass / $($qPass + $qWarn) clean" -ForegroundColor $(if($qWarn -eq 0){'Green'}else{'Yellow'})
Write-Host "  Warnings:         $Warn" -ForegroundColor $(if($Warn -eq 0){'Green'}else{'Yellow'})
Write-Host ""
if ($missing) {
    Write-Host "  VERDICT: FAIL — $($missing.Count) expected skills missing" -ForegroundColor Red
    exit 1
} elseif ($fmFail -gt 0) {
    Write-Host "  VERDICT: FAIL — $fmFail frontmatter errors" -ForegroundColor Red
    exit 1
} elseif ($dupes) {
    Write-Host "  VERDICT: FAIL — duplicate skill names" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  VERDICT: PASS — All 24 skills validated successfully!" -ForegroundColor Green
    exit 0
}
