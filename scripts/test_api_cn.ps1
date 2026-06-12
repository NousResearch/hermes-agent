[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$headers = @{
    "Content-Type" = "application/json; charset=utf-8"
    "Authorization" = "Bearer sk-dooFBpzVWgrvf32YLPFfq5r63dEYHELlUjMT84KrEH5wG0zN"
}
$body = '{"model":"Qwen3-235B-A22B-w8a8","messages":[{"role":"user","content":"你好，请回复一个字"}],"max_tokens":10}'
$response = Invoke-WebRequest -Uri "https://ai-pool.evebattery.com/v1/chat/completions" -Method POST -Headers $headers -Body $body -TimeoutSec 30
Write-Output $response.Content
