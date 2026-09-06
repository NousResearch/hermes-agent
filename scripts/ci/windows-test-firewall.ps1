param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Enable", "Run", "Disable")]
    [string]$Mode,
    [string]$PytestArgsBase64 = ""
)

$ErrorActionPreference = "Stop"
$group = "Hermes Test Boundary"
$identity = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB" -replace '[^A-Za-z0-9_.-]', '_'
$stateDir = Join-Path $env:ProgramData "HermesTestBoundary"
$statePath = Join-Path $stateDir "$identity.json"

if ($env:GITHUB_ACTIONS -ne "true" -or $env:RUNNER_ENVIRONMENT -ne "github-hosted") {
    throw "Windows Hermes tests require a disposable GitHub-hosted runner"
}

function Read-State {
    if (-not (Test-Path $statePath)) { throw "Hermes boundary state is missing" }
    return Get-Content $statePath -Raw | ConvertFrom-Json
}

if ($Mode -eq "Disable") {
    if (Test-Path $statePath) {
        $state = Read-State
        $testUser = Get-LocalUser -Name $state.User -ErrorAction SilentlyContinue
        if ($testUser) {
            $account = "$env:COMPUTERNAME\$($state.User)"
            for ($attempt = 0; $attempt -lt 5; $attempt++) {
                $owned = @(Get-Process -IncludeUserName -ErrorAction Stop |
                    Where-Object { $_.UserName -ieq $account })
                if (-not $owned) { break }
                $owned | ForEach-Object {
                    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                }
                Start-Sleep -Milliseconds 200
            }
            $remaining = @(Get-Process -IncludeUserName -ErrorAction Stop |
                Where-Object { $_.UserName -ieq $account })
            if ($remaining) {
                throw "Restricted Hermes test processes remain; firewall left fail-closed"
            }
        }
        Remove-LocalUser -Name $state.User -ErrorAction SilentlyContinue
        Remove-Item $state.TestRoot -Recurse -Force -ErrorAction SilentlyContinue
        # Deliberately do not restore outbound access. A low-privilege test can
        # enqueue persistent broker work (for example BITS COM jobs) whose
        # service process survives the test user. Only VM destruction is a
        # complete broker cleanup boundary. Runner-only allows remain so the
        # hosted worker can report the result while every other allow stays
        # disabled and the default stays Block until teardown.
        Remove-Item $statePath -Force
    }
    exit 0
}

if ($Mode -eq "Run") {
    $state = Read-State
    if (-not $PytestArgsBase64) { throw "Run requires encoded pytest arguments" }
    $arguments = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($PytestArgsBase64)
    ) | ConvertFrom-Json
    if ($arguments -isnot [array]) { $arguments = @($arguments) }
    $password = ConvertTo-SecureString $state.ProtectedPassword
    $credential = [Management.Automation.PSCredential]::new(
        ".\$($state.User)", $password
    )
    $python = (Resolve-Path ".venv/Scripts/python.exe").Path
    $entry = (Resolve-Path "scripts/ci/windows-restricted-test-entry.py").Path
    $repo = (Resolve-Path ".").Path
    $testHome = Join-Path $state.TestRoot "home"
    New-Item -ItemType Directory -Path (Join-Path $testHome ".hermes") -Force | Out-Null

    $safeNames = @(
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
        "PROGRAMFILES", "PROGRAMFILES(X86)"
    )
    $safe = @{}
    foreach ($name in $safeNames) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($null -ne $value) { $safe[$name] = $value }
    }
    $safe["HOME"] = $testHome
    $safe["USERPROFILE"] = $testHome
    $safe["HERMES_HOME"] = Join-Path $testHome ".hermes"
    $safe["XDG_CONFIG_HOME"] = Join-Path $testHome ".config"
    $safe["XDG_CACHE_HOME"] = Join-Path $testHome ".cache"
    $safe["XDG_DATA_HOME"] = Join-Path $testHome ".local\share"
    $safe["XDG_STATE_HOME"] = Join-Path $testHome ".local\state"
    $safe["LOCALAPPDATA"] = Join-Path $testHome "AppData\Local"
    $safe["APPDATA"] = Join-Path $testHome "AppData\Roaming"
    $safe["PROGRAMDATA"] = Join-Path $state.TestRoot "ProgramData"
    $safe["TEMP"] = $state.TestRoot
    $safe["TMP"] = $state.TestRoot
    $safe["CI"] = "true"
    $safe["GITHUB_ACTIONS"] = "true"
    $safe["RUNNER_ENVIRONMENT"] = "github-hosted"
    $safe["HERMES_TEST_GUARD_ACTIVE"] = "1"
    $safe["HERMES_TEST_OS_SANDBOX"] = "windows-ephemeral-ci"
    $safe["HERMES_TEST_WINDOWS_FIREWALL"] = "1"
    $safe["HERMES_TEST_WINDOWS_RESTRICTED_USER"] = $state.User
    $safe["HERMES_TEST_WINDOWS_STATE_PATH"] = $statePath
    $safe["HERMES_TEST_REAL_HOME"] = $state.RealHome
    $safe["HERMES_TEST_REPO_ROOT"] = $repo
    $safe["PYTHONPATH"] = Join-Path $repo "scripts\hermetic_site"
    $safe["PYTHONDONTWRITEBYTECODE"] = "1"
    $safe["PYTHONHASHSEED"] = "0"
    $safe["PYTHONUTF8"] = "1"
    $safe["TZ"] = "UTC"
    foreach ($name in @("HERMES_RUN_SLOW_PET_TESTS", "HERMES_E2E_BROWSER")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { $safe[$name] = $value }
    }
    $contract = @{
        python = $python
        repo = $repo
        environment = $safe
        arguments = @($arguments)
    } | ConvertTo-Json -Compress -Depth 5
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($contract))
    $process = Start-Process -FilePath $python `
        -ArgumentList @("`"$entry`"", $encoded) `
        -WorkingDirectory $repo -Credential $credential `
        -LoadUserProfile -Wait -PassThru -NoNewWindow
    exit $process.ExitCode
}

New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
icacls $stateDir /inheritance:r /grant:r `
    "SYSTEM:(OI)(CI)F" "Administrators:(OI)(CI)F" `
    "$env:USERNAME`:(OI)(CI)F" | Out-Null

$profiles = Get-NetFirewallProfile | ForEach-Object {
    [pscustomobject]@{
        Name = $_.Name
        DefaultOutboundAction = [string]$_.DefaultOutboundAction
    }
}
$outboundAllowRules = @(Get-NetFirewallRule -Direction Outbound -Action Allow `
    -Enabled True | Select-Object -ExpandProperty Name)
$suffix = ($identity -replace '[^A-Za-z0-9]', '')
if ($suffix.Length -gt 10) { $suffix = $suffix.Substring($suffix.Length - 10) }
$user = "hermesci$suffix"
$plainPassword = "Aa1!" + [Guid]::NewGuid().ToString("N")
$securePassword = ConvertTo-SecureString $plainPassword -AsPlainText -Force
$testRoot = Join-Path $env:RUNNER_TEMP "hermes-restricted-$identity"

try {
    New-LocalUser -Name $user -Password $securePassword `
        -PasswordNeverExpires -UserMayNotChangePassword | Out-Null
    Add-LocalGroupMember -Group "Users" -Member $user
    New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
    icacls $testRoot /inheritance:r /grant:r `
        "SYSTEM:(OI)(CI)F" "Administrators:(OI)(CI)F" `
        "$user`:(OI)(CI)M" | Out-Null
    icacls $env:GITHUB_WORKSPACE /grant:r "$user`:(OI)(CI)RX" | Out-Null

    @{
        ProfilesAtStart = @($profiles)
        OutboundAllowRulesDisabled = @($outboundAllowRules)
        User = $user
        ProtectedPassword = ($securePassword | ConvertFrom-SecureString)
        TestRoot = $testRoot
        RealHome = $env:USERPROFILE
    } | ConvertTo-Json -Depth 4 | Set-Content $statePath -Encoding utf8
    icacls $statePath /inheritance:r /grant:r `
        "SYSTEM:F" "Administrators:F" "$env:USERNAME`:F" | Out-Null

    $runnerSid = ([Security.Principal.NTAccount]$env:USERNAME).Translate(
        [Security.Principal.SecurityIdentifier]
    ).Value
    $runnerSddl = "D:(A;;CC;;;$runnerSid)"
    $runnerPrograms = Get-Process -Name "Runner.Worker", "Runner.Listener" `
        -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Path } | Where-Object { $_ } | Sort-Object -Unique
    if (-not $runnerPrograms) { throw "GitHub runner process paths were not found" }
    foreach ($program in $runnerPrograms) {
        New-NetFirewallRule -DisplayName "Hermes allow runner $program" `
            -Group $group -Direction Outbound -Action Allow -Program $program `
            -LocalUser $runnerSddl -Profile Any -Enabled True | Out-Null
    }
    foreach ($ruleName in $outboundAllowRules) {
        Set-NetFirewallRule -Name $ruleName -Enabled False
    }
    Set-NetFirewallProfile -Profile Domain,Public,Private -DefaultOutboundAction Block

    $blockedPrograms = @(
        (Resolve-Path ".venv/Scripts/python.exe" -ErrorAction SilentlyContinue).Path,
        (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command cmd.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command curl.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command git.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command ssh.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command node.exe -ErrorAction SilentlyContinue).Source
    ) | Where-Object { $_ } | Sort-Object -Unique
    foreach ($program in $blockedPrograms) {
        New-NetFirewallRule -DisplayName "Hermes deny $program" `
            -Group $group -Direction Outbound -Action Block -Program $program `
            -Profile Any -Enabled True | Out-Null
    }
} catch {
    & $PSCommandPath -Mode Disable
    throw
} finally {
    $plainPassword = $null
}
