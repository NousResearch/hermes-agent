param(
    [string]$ServerExe = "",
    [string]$ModelPath = "",
    [string]$HfRepo = "",
    [int]$Port = 8081,
    [int]$ContextSize = 65536,
    [ValidateSet("f16v_turbo4", "f16v_q4_0", "bf16v_q4_0", "bf16v_turbo3", "triality_vector_v_turbo3", "triality_plus_v_turbo3", "triality_minus_v_turbo3", "turbo4", "q4_0")]
    [string]$KvProfile = "triality_vector_v_turbo3",
    [ValidateSet("ngram-mod", "mtp", "none")]
    [string]$SpecType = "ngram-mod",
    [int]$SpecNgramMatch = 24,
    [int]$SpecNgramMin = 48,
    [int]$SpecNgramMax = 64,
    [int]$WaitSeconds = 240
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SharedScript = Join-Path $ScriptDir "start-hermes-llama-fallback.ps1"

& $SharedScript `
    -GpuProfile "rtx5060ti" `
    -ServerExe $ServerExe `
    -ModelPath $ModelPath `
    -HfRepo $HfRepo `
    -Port $Port `
    -ContextSize $ContextSize `
    -KvProfile $KvProfile `
    -SpecType $SpecType `
    -SpecNgramMatch $SpecNgramMatch `
    -SpecNgramMin $SpecNgramMin `
    -SpecNgramMax $SpecNgramMax `
    -WaitSeconds $WaitSeconds
