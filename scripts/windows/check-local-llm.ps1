# Smoke-test local llama.cpp OpenAI-compatible endpoint and emit JSON summary.

param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [int]$MinContext = 64000,
    [switch]$UsePython
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonScript = Join-Path $repoRoot "agent\local_secretary\llama_contract.py"

if ($UsePython -or (Test-Path -LiteralPath $pythonScript)) {
    $code = @"
import json, sys
from agent.local_secretary.llama_contract import run_llama_contract_checks
payload = run_llama_contract_checks(sys.argv[1], min_context=int(sys.argv[2]))
print(json.dumps(payload, ensure_ascii=False, indent=2))
sys.exit(0 if payload.get('ok') else 1)
"@
    Push-Location $repoRoot
    try {
        $json = py -3 -c $code $BaseUrl $MinContext
        Write-Output $json
        $parsed = $json | ConvertFrom-Json
        if (-not $parsed.ok) { exit 1 }
        exit 0
    } finally {
        Pop-Location
    }
}

$base = $BaseUrl.TrimEnd('/')
$result = [ordered]@{
    base_url = $base
    min_context = $MinContext
    ok = $true
    checks = [ordered]@{}
    summary = "ok"
}

function Set-CheckFailure {
    param([string]$Name, [string]$Message)
    $result.checks[$Name] = @{ ok = $false; error = $Message }
    $result.ok = $false
    $result.summary = "failed"
}

try {
    $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
    $ids = @($models.data | ForEach-Object { $_.id })
    $result.checks.models = @{ ok = $true; model_ids = $ids }
    $modelId = if ($ids.Count -gt 0) { $ids[0] } else { "unknown" }
} catch {
    Set-CheckFailure "models" $_.Exception.Message
    $result | ConvertTo-Json -Depth 8
    exit 1
}

try {
    $props = Invoke-RestMethod -Uri "$base/props" -TimeoutSec 10
    $ctx = $null
    if ($props.default_generation_settings.n_ctx) {
        $ctx = [int]$props.default_generation_settings.n_ctx
    }
    if ($null -eq $ctx -and $props.n_ctx) { $ctx = [int]$props.n_ctx }
    if ($null -eq $ctx -or $ctx -lt $MinContext) {
        Set-CheckFailure "context_size" "context $ctx below minimum $MinContext"
    } else {
        $result.checks.context_size = @{ ok = $true; n_ctx = $ctx }
    }
} catch {
    Set-CheckFailure "props" $_.Exception.Message
}

$chatBody = @{
    model = $modelId
    messages = @(@{ role = "user"; content = "Reply with the single word: pong" })
    max_tokens = 16
    temperature = 0
} | ConvertTo-Json -Depth 6

try {
    $chat = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -Body $chatBody -ContentType "application/json; charset=utf-8" -TimeoutSec 120
    $content = $chat.choices[0].message.content
    $result.checks.chat_completion = @{ ok = $true; content_preview = [string]$content.Substring(0, [Math]::Min(120, [string]$content.Length)) }
} catch {
    Set-CheckFailure "chat_completion" $_.Exception.Message
}

$toolBody = @{
    model = $modelId
    messages = @(@{ role = "user"; content = "What is the weather in Tokyo?" })
    tools = @(@{
        type = "function"
        function = @{
            name = "get_weather"
            description = "Get weather for a city"
            parameters = @{
                type = "object"
                properties = @{ city = @{ type = "string" } }
                required = @("city")
            }
        }
    })
    tool_choice = "auto"
    max_tokens = 128
    temperature = 0
} | ConvertTo-Json -Depth 10

try {
    $tool = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -Body $toolBody -ContentType "application/json; charset=utf-8" -TimeoutSec 120
    $message = $tool.choices[0].message
    if ($message.tool_calls -and $message.tool_calls.Count -gt 0) {
        $result.checks.tool_calling = @{ ok = $true; tool_calls = $message.tool_calls.Count }
    } elseif ($message.content -match '(?i)tool_call|function_call|<tool_call>|Action:') {
        Set-CheckFailure "tool_calling" "tool call returned as plain text — start llama-server with --jinja"
    } else {
        Set-CheckFailure "tool_calling" "no structured tool_calls in response"
    }
} catch {
    Set-CheckFailure "tool_calling" $_.Exception.Message
}

$result | ConvertTo-Json -Depth 8
if (-not $result.ok) { exit 1 }
