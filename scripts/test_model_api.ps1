# test_model_api.ps1 - Test model API connectivity
# Usage: powershell -ExecutionPolicy Bypass -File scripts/test_model_api.ps1

$apiBase = "https://ai-pool.evebattery.com/v1/chat/completions"
$apiKey  = "sk-dooFBpzVWgrvf32YLPFfq5r63dEYHELlUjMT84KrEH5wG0zN"
$model   = "Qwen3-235B-A22B-w8a8"

Write-Host "[Test] POST $apiBase"
Write-Host "[Test] Model: $model"

$body = '{"model":"Qwen3-235B-A22B-w8a8","messages":[{"role":"user","content":"hello, introduce yourself in one sentence"}],"max_tokens":200}'

$headers = @{
    "Content-Type"  = "application/json"
    "Authorization" = $apiKey
}

try {
    [System.Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}
    $resp = Invoke-RestMethod -Uri $apiBase -Method Post -Headers $headers -Body $body -TimeoutSec 60
    Write-Host "[OK] HTTP 200"
    Write-Host "[Reply] $($resp.choices[0].message.content)"
    Write-Host "[Usage] prompt=$($resp.usage.prompt_tokens) completion=$($resp.usage.completion_tokens) total=$($resp.usage.total_tokens)"
    Write-Host ""
    Write-Host "API test PASSED" -ForegroundColor Green
}
catch {
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
