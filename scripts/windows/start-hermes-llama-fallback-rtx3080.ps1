param(
    [string]$ServerExe = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$ContextSize = 65536,
    [ValidateSet("f16v_turbo4", "f16v_q4_0", "bf16v_turbo3", "turbo4", "q4_0")]
    [string]$KvProfile = "bf16v_turbo3",
    [ValidateSet("ngram-mod", "mtp", "none")]
    [string]$SpecType = "ngram-mod",
    [int]$WaitSeconds = 180
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SharedScript = Join-Path $ScriptDir "start-hermes-llama-fallback.ps1"

& $SharedScript `
    -GpuProfile "rtx3080" `
    -ServerExe $ServerExe `
    -ModelPath $ModelPath `
    -Port $Port `
    -ContextSize $ContextSize `
    -KvProfile $KvProfile `
    -SpecType $SpecType `
    -WaitSeconds $WaitSeconds
