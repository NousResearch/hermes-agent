#Requires -Version 5.1
<#
  全自动：安装 Ollama（若缺失）→ 启动本机服务 → 拉取 gemma4:26b → 试跑一条对话。
  需要管理员权限时仅 winget 安装步骤可能弹出 UAC（视策略而定）。
  日志：默认追加到 $env:TEMP\hermes-gemma4-auto.log
#>
$ErrorActionPreference = "Stop"
$LogFile = Join-Path $env:TEMP "hermes-gemma4-auto.log"
function Write-Log([string]$m) {
    $line = "$(Get-Date -Format o)  $m"
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

function Find-OllamaExe {
    foreach ($p in @(
            "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
            "$env:ProgramFiles\Ollama\ollama.exe",
            "C:\Program Files\Ollama\ollama.exe"
        )) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    return $null
}

function Wait-OllamaApi {
    param([int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        }
        catch {}
        Start-Sleep -Seconds 2
    }
    return $false
}

function Ensure-OllamaInstalled {
    $exe = Find-OllamaExe
    if ($exe) {
        Write-Log "Found Ollama: $exe"
        return $exe
    }
    Write-Log "Ollama not found. Installing via winget (large download, please wait)..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget not available. Install Ollama manually from https://ollama.com/download then re-run this script."
    }
    & winget.exe install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "winget install Ollama failed (exit $LASTEXITCODE). Install from https://ollama.com/download then re-run."
    }
    # 刷新 PATH（当前进程）
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
    Start-Sleep -Seconds 3
    $exe = Find-OllamaExe
    if (-not $exe) {
        throw "Ollama installed but ollama.exe not found. Restart the terminal or PC, then re-run."
    }
    Write-Log "Installed Ollama: $exe"
    return $exe
}

function Ensure-OllamaServe([string]$OllamaExe) {
    if (Wait-OllamaApi -TimeoutSec 5) {
        Write-Log "Ollama API already up on http://127.0.0.1:11434"
        return
    }
    Write-Log "Starting: `"$OllamaExe`" serve"
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    if (-not (Wait-OllamaApi -TimeoutSec 180)) {
        throw "Ollama did not respond on port 11434. Try starting Ollama from the Start menu, then re-run."
    }
    Write-Log "Ollama API is ready."
}

# --- main ---
Write-Log "=== autoinstall-run-gemma4-26b start ==="
try {
    $ollama = Ensure-OllamaInstalled
    Ensure-OllamaServe -OllamaExe $ollama

    Write-Log "Pulling gemma4:26b (~18GB)..."
    & $ollama pull gemma4:26b
    if ($LASTEXITCODE -ne 0) {
        throw "ollama pull failed (exit $LASTEXITCODE)"
    }
    Write-Log "Pull complete."

    Write-Log "Smoke test: ollama run gemma4:26b ..."
    $msg = "Reply with exactly one word: OK"
    & $ollama run gemma4:26b $msg
    Write-Log "Smoke test finished."

    Write-Log "Done. Use: ollama run gemma4:26b"
    Write-Log "Hermes 已可配合: base_url http://127.0.0.1:11434/v1 , model gemma4:26b"
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
Write-Log "=== autoinstall-run-gemma4-26b end (success) ==="
exit 0
