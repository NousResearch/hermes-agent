# Journey-owned mock and ordinary window close, shared by source and MSIX drivers.
# Callers supply a checkout-provisioned Node; never resolve tooling from OLD.
function Start-DesktopJourneyMock([string]$Node, [string]$Assets, [string]$Work, [string]$HermesHome, [string]$Out) {
    New-Item -ItemType Directory -Path $Out -Force | Out-Null
    $urlFile = Join-Path $Work 'chat-mock-url'
    Remove-Item -LiteralPath $urlFile -Force -ErrorAction SilentlyContinue
    $mock = Start-Process -FilePath $Node -ArgumentList @(('"' + (Join-Path $Assets 'mock-provider.mjs') + '"'), ('"' + $urlFile + '"')) `
        -PassThru -RedirectStandardOutput (Join-Path $Out 'chat-mock.log') -RedirectStandardError (Join-Path $Out 'chat-mock-error.log')
    try {
        $deadline = (Get-Date).AddSeconds(30)
        while (-not (Test-Path -LiteralPath $urlFile)) {
            if ($mock.HasExited -or (Get-Date) -ge $deadline) { throw 'Desktop mock did not become ready' }
            Start-Sleep -Milliseconds 200
        }
        $env:HERMES_E2E_MOCK_URL = (Get-Content -LiteralPath $urlFile -Raw).Trim()
        Invoke-RestMethod -Uri "$env:HERMES_E2E_MOCK_URL/__e2e__/prompts" -TimeoutSec 5 | Out-Null
        $configWriter = [IO.Path]::GetFullPath((Join-Path $Assets '..\..\..\tests-js\scripts\mock-provider-config.ts'))
        & $Node $configWriter $HermesHome $env:HERMES_E2E_MOCK_URL
        if ($LASTEXITCODE -ne 0) { throw 'Desktop mock configuration failed' }
        return $mock
    } catch {
        if (-not $mock.HasExited) { Stop-Process -Id $mock.Id -ErrorAction SilentlyContinue }
        throw
    }
}

function Get-VerifiedDesktopWindows([string]$Exe) {
    # The detached updater canonicalizes the WorkRoot path while the driver can
    # retain ``..`` segments. Compare executable identity, not path spelling.
    $expectedExe = [IO.Path]::GetFullPath($Exe)
    $rows = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath) -ieq $expectedExe
    })
    return @($rows | ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } | Where-Object { $_.MainWindowHandle -ne 0 })
}

function Close-VerifiedDesktop([string]$Exe, [int]$ProcessId = 0) {
    $windows = @(Get-VerifiedDesktopWindows $Exe)
    if ($ProcessId) { $windows = @($windows | Where-Object { $_.Id -eq $ProcessId }) }
    if ($windows.Count -ne 1) { throw 'Cannot identify exactly one verified desktop window to close normally' }
    $window = $windows[0]
    if (-not $window.CloseMainWindow()) { throw 'Verified desktop refused a normal window close' }
    if (-not $window.WaitForExit(30000)) { throw 'Verified desktop did not exit normally; no force-kill or smoke relaunch attempted' }
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $remaining = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -ieq $Exe })
        if (-not $remaining.Count) { return }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    throw 'Verified desktop children did not exit normally; no smoke relaunch attempted'
}