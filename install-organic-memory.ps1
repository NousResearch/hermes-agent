# Organic Memory Architecture — One-Line Installer for Windows
# Usage: iex (irm https://raw.githubusercontent.com/20231118185SSPU/hermes-agent/feat/organic-memory-architecture/install-organic-memory.ps1)

$ErrorActionPreference = "Stop"

Write-Host "`n🧬 Organic Memory Architecture Installer" -ForegroundColor Cyan
Write-Host ""

# Find Hermes installation
$hermesPaths = @(
    "$env:LOCALAPPDATA\hermes\hermes-agent",
    "$env:USERPROFILE\.hermes",
    "$env:USERPROFILE\hermes-agent"
)

$hermesDir = $null
foreach ($p in $hermesPaths) {
    if (Test-Path "$p\agent\memory_manager.py") {
        $hermesDir = $p
        break
    }
}

if (-not $hermesDir) {
    Write-Host "❌ Could not find Hermes installation." -ForegroundColor Red
    Write-Host "   Expected locations:" -ForegroundColor Yellow
    foreach ($p in $hermesPaths) { Write-Host "   - $p" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "   Install Hermes first: iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Found Hermes at: $hermesDir" -ForegroundColor Green

# Backup current state
Write-Host "📦 Creating backup..." -ForegroundColor Cyan
Push-Location $hermesDir
git stash push -m "organic-memory-installer-backup" 2>$null
Pop-Location

# Download files
Write-Host "📥 Downloading organic memory files..." -ForegroundColor Cyan

$baseUrl = "https://raw.githubusercontent.com/20231118185SSPU/hermes-agent/feat/organic-memory-architecture"
$files = @(
    @{src="agent/memory_pipeline.py"; dst="agent\memory_pipeline.py"},
    @{src="plugins/memory/holographic/episodic.py"; dst="plugins\memory\holographic\episodic.py"},
    @{src="plugins/memory/holographic/dreaming.py"; dst="plugins\memory\holographic\dreaming.py"},
    @{src="ORGANIC_MEMORY.md"; dst="ORGANIC_MEMORY.md"}
)

foreach ($f in $files) {
    $dstPath = Join-Path $hermesDir $f.dst
    $dstDir = Split-Path $dstPath -Parent
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    try {
        Invoke-WebRequest -Uri "$baseUrl/$($f.src)" -OutFile $dstPath -UseBasicParsing
        Write-Host "  ✅ $($f.src)" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ $($f.src): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Patch memory_manager.py
Write-Host "🔧 Patching memory_manager.py..." -ForegroundColor Cyan
$managerPath = Join-Path $hermesDir "agent\memory_manager.py"
$managerContent = Get-Content $managerPath -Raw

if ($managerContent -notmatch "from agent\.memory_pipeline import") {
    # Add import after existing memory_provider import
    $managerContent = $managerContent -replace "(from agent\.memory_provider import MemoryProvider)", "`$1`nfrom agent.memory_pipeline import MemoryPipeline, _load_pipeline_config"

    # Add _pipeline field to __init__
    $managerContent = $managerContent -replace "(self\._has_external: bool = False)", "`$1`n        self._pipeline: MemoryPipeline | None = None"

    Set-Content $managerPath $managerContent -Encoding UTF8
    Write-Host "  ✅ memory_manager.py patched (import + field)" -ForegroundColor Green
    Write-Host "  ⚠️  Note: You still need to wire pipeline into lifecycle methods." -ForegroundColor Yellow
    Write-Host "     See: https://github.com/20231118185SSPU/hermes-agent/blob/feat/organic-memory-architecture/ORGANIC_MEMORY.md" -ForegroundColor Yellow
} else {
    Write-Host "  ⚠️  memory_manager.py already has pipeline import" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To enable, add to your config.yaml:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  memory:" -ForegroundColor White
Write-Host "    provider: holographic" -ForegroundColor White
Write-Host "    pipeline:" -ForegroundColor White
Write-Host "      enabled: true" -ForegroundColor White
Write-Host "      episodic:" -ForegroundColor White
Write-Host "        enabled: true" -ForegroundColor White
Write-Host "      dreaming:" -ForegroundColor White
Write-Host "        enabled: true" -ForegroundColor White
Write-Host ""
Write-Host "Then restart: hermes" -ForegroundColor Cyan
Write-Host ""
Write-Host "Full merge (recommended):" -ForegroundColor Yellow
Write-Host "  cd $hermesDir" -ForegroundColor White
Write-Host "  git remote add fork https://github.com/20231118185SSPU/hermes-agent.git" -ForegroundColor White
Write-Host "  git fetch fork feat/organic-memory-architecture" -ForegroundColor White
Write-Host "  git merge fork/feat/organic-memory-architecture --no-edit" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
