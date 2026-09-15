# Child installers/builds stamp their checkout; restore the workflow identity even on failure.
function Invoke-SourceBuild([scriptblock]$Action) {
    $saved = @{}
    $names = @('GITHUB_SHA', 'GITHUB_REF', 'GITHUB_REF_NAME', 'GITHUB_HEAD_REF', 'GITHUB_BASE_REF',
        'HERMES_BUILD_COMMIT', 'HERMES_PAYLOAD_TAG', 'HERMES_PAYLOAD_VERSION', 'HERMES_DESKTOP_VARIANT')
    try {
        foreach ($name in $names) {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        & $Action
    } finally {
        foreach ($name in $names) {
            [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
        }
    }
}