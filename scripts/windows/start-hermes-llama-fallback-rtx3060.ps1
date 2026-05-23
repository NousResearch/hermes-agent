param(
    [string]$ServerExe = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$ContextSize = 65536,
    [ValidateSet("f16v_turbo4", "f16v_q4_0", "turbo4", "q4_0")]
    [string]$KvProfile = "f16v_turbo4",
    [ValidateSet("ngram-mod", "mtp", "none")]
    [string]$SpecType = "ngram-mod",
    [int]$SpecNgramMatch = 24,
    [int]$SpecNgramMin = 48,
    [int]$SpecNgramMax = 64,
    [int]$WaitSeconds = 180
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SharedScript = Join-Path $ScriptDir "start-hermes-llama-fallback.ps1"

& $SharedScript `
    -GpuProfile "rtx3060" `
    -ServerExe $ServerExe `
    -ModelPath $ModelPath `
    -Port $Port `
    -ContextSize $ContextSize `
    -KvProfile $KvProfile `
    -SpecType $SpecType `
    -SpecNgramMatch $SpecNgramMatch `
    -SpecNgramMin $SpecNgramMin `
    -SpecNgramMax $SpecNgramMax `
    -WaitSeconds $WaitSeconds
