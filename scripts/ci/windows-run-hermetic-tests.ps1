param(
    [Parameter(Mandatory = $true)]
    [string]$TestListPath,
    [string]$Marker = "",
    [switch]$IgnoreDesktopUpdater,
    [switch]$ClearAddopts
)

$ErrorActionPreference = "Stop"
$tests = @(Get-Content $TestListPath | Where-Object { $_.Trim() })
if (-not $tests) { throw "Hermetic Windows test list is empty" }
$arguments = @($tests)
if ($IgnoreDesktopUpdater) {
    $arguments += "--ignore-glob=*test_desktop_update_windows_*.py"
}
if ($Marker) {
    $arguments += @("-m", "$Marker and not integration")
}
if ($ClearAddopts) {
    $arguments += @("-o", "addopts=")
}
$arguments += @("-v", "--tb=short", "-p", "no:cacheprovider")
$json = ConvertTo-Json -InputObject $arguments -Compress
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
& "$PSScriptRoot/windows-test-firewall.ps1" -Mode Run `
    -PytestArgsBase64 $encoded
exit $LASTEXITCODE
