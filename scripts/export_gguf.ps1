param(
    [Parameter(Mandatory = $true)]
    [string]$MergedModelDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputGguf,

    [string]$LlamaCppRoot = "",
    [string]$ConvertScript = "",
    [string]$QuantizeExe = "",
    [string]$PythonExe = "python",
    [switch]$NoMtp,
    [string]$Quantization = "Q8_0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $MergedModelDir)) {
    throw "MergedModelDir not found: $MergedModelDir"
}

if (-not $LlamaCppRoot) {
    $fromEnv = [Environment]::GetEnvironmentVariable("LLAMA_CPP_ROOT")
    if ($fromEnv -and $fromEnv.Trim()) {
        $LlamaCppRoot = $fromEnv.Trim()
    }
}

if (-not $LlamaCppRoot -and -not $ConvertScript) {
    throw "Set -LlamaCppRoot, LLAMA_CPP_ROOT, or -ConvertScript to a llama.cpp convert_hf_to_gguf.py path."
}

if ($LlamaCppRoot -and -not (Test-Path -LiteralPath $LlamaCppRoot)) {
    throw "LlamaCppRoot not found: $LlamaCppRoot"
}

if ($ConvertScript) {
    $convert = $ConvertScript
} else {
    $convert = Join-Path $LlamaCppRoot "convert_hf_to_gguf.py"
}

if ($QuantizeExe) {
    $quantize = $QuantizeExe
} else {
    $quantize = Join-Path $LlamaCppRoot "build\bin\Release\llama-quantize.exe"
}

if (-not (Test-Path -LiteralPath $convert)) {
    throw "convert_hf_to_gguf.py not found: $convert"
}
if (-not (Test-Path -LiteralPath $quantize)) {
    if ($LlamaCppRoot) {
        $quantize = Join-Path $LlamaCppRoot "llama-quantize.exe"
    }
}
if (-not (Test-Path -LiteralPath $quantize)) {
    throw "llama-quantize.exe not found. Build llama.cpp first or pass -QuantizeExe."
}

$outPath = [IO.Path]::GetFullPath($OutputGguf)
$outDir = Split-Path -Parent $outPath
if ($outDir) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$f16Path = [IO.Path]::ChangeExtension($outPath, ".f16.gguf")
$convertArgs = @($convert, "--outfile", $f16Path, "--outtype", "f16")
if ($NoMtp) {
    $convertArgs += "--no-mtp"
}
$convertArgs += $MergedModelDir
& $PythonExe @convertArgs
& $quantize $f16Path $outPath $Quantization

Write-Host "Wrote GGUF: $outPath"
