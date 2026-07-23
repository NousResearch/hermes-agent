# Launcher for Qwen3.6-35B IQ3_M model on RTX 5060 Ti 16GB with GPU + CPU offloading
# Path: C:\Users\downl\Desktop\SO8T\gguf_models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf

param(
    [int]$GpuLayers = 28,
    [int]$WaitSeconds = 300
)

$ErrorActionPreference = "Stop"

$ModelPath = "C:\Users\downl\Desktop\SO8T\gguf_models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model file not found at: $ModelPath"
}

$env:HERMES_LLAMA_GGUF_PATH = $ModelPath
$env:HERMES_LLAMA_MODEL = "Qwen3.6-35B-A3B-Uncensored-IQ3_M"
$env:HERMES_LLAMA_ALIAS = "Qwen3.6-35B-A3B-Uncensored-IQ3_M"
$env:HERMES_LLAMA_GPU_LAYERS = [string]$GpuLayers
$env:HERMES_LLAMA_THREADS = "8"
$env:HERMES_LLAMA_CACHE_TYPE_K = "q4_0"
$env:HERMES_LLAMA_CACHE_TYPE_V = "q4_0"

Write-Host "[Qwen3.6-35B Launcher] Offloading $GpuLayers/40 layers to GPU VRAM (RTX 5060 Ti 16GB), remaining to CPU RAM..."
$script = Join-Path $PSScriptRoot "start-llama-secretary.ps1"
& $script -WaitSeconds $WaitSeconds
