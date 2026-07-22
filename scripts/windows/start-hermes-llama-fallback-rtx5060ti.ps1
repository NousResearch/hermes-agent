param(
    [int]$Port = 8081,
    [int]$ContextSize = 65536,
    [string]$ModelPath = 'H:\models\InternScience\Agents-A1-4B-F16-GGUF\Agents-A1-4B-F16.gguf',
    [string]$ServerExe = "$env:LOCALAPPDATA\Programs\llama-turboquant\bin\llama-server.exe"
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ServerExe)) { throw "llama-server not found: $ServerExe" }
if (-not (Test-Path -LiteralPath $ModelPath)) { throw "model not found: $ModelPath" }

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing) {
    Write-Output "llama-server already listening on $Port (pid=$($existing.OwningProcess))"
    exit 0
}

$logDir = Join-Path $env:USERPROFILE '.hermes\logs\llama-fallback'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outLog = Join-Path $logDir "llama-rtx5060ti-$stamp.out.log"
$errLog = Join-Path $logDir "llama-rtx5060ti-$stamp.err.log"

# RTX 5060 Ti 16GB + installed llama-turboquant v10264.
# The installed Agents-A1-4B F16 model uses ~10.3GB VRAM with this profile.
$args = @(
    '--model', $ModelPath,
    '--host', '127.0.0.1', '--port', [string]$Port,
    '--ctx-size', [string]$ContextSize,
    '--n-gpu-layers', '99',
    '--flash-attn', 'on',
    '--cache-type-k', 'turbo4', '--cache-type-v', 'turbo4',
    '--parallel', '1', '--batch-size', '1024', '--ubatch-size', '256',
    '--reasoning', 'off', '--reasoning-budget', '0',
    '--jinja', '--cont-batching',
    '--spec-type', 'ngram-mod',
    '--spec-ngram-mod-n-match', '24',
    '--spec-ngram-mod-n-min', '48',
    '--spec-ngram-mod-n-max', '64'
)

$p = Start-Process -FilePath $ServerExe -ArgumentList $args -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
    if ($p.HasExited) { throw "llama-server exited with $($p.ExitCode). See $errLog" }
    try {
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
        if ($models.error.type -ne 'unavailable_error') {
            Write-Output "ready http://127.0.0.1:$Port/v1"
            Write-Output "pid=$($p.Id) model=$ModelPath ctx=$ContextSize kv=turbo4/turbo4 batch=1024/256"
            exit 0
        }
    } catch {}
    Start-Sleep -Seconds 2
}
throw "llama-server did not become ready. See $errLog"
