param(
    [string]$AvatarUrl = "",
    [string]$AdbPath = "",
    [string]$ScrcpyPath = "",
    [string]$PnpSnapshotPath = "",
    [string]$DriverSnapshotPath = "",
    [string]$BrowserPackage = "com.brave.browser",
    [int]$WaitForAdbSeconds = 0,
    [switch]$ConfigureDevice,
    [switch]$OpenOnDevice,
    [switch]$LockTask,
    [switch]$LaunchScrcpy
)

$ErrorActionPreference = "Stop"

function Find-Tool {
    param(
        [string]$Name,
        [string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (Test-Path -LiteralPath $ExplicitPath) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        throw "$Name was not found at explicit path: $ExplicitPath"
    }

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $wingetRoot) {
        $found = Get-ChildItem -LiteralPath $wingetRoot -Recurse -File -Filter $Name -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    return ""
}

function Get-HermesStatusJson {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & py -3 -m hermes_cli aituber-onair status 2>&1
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    $text = ($output | Out-String).Trim()
    $start = $text.IndexOf("{")
    $end = $text.LastIndexOf("}")
    if ($start -lt 0 -or $end -lt $start) {
        throw "Could not parse Hermes AITuber status JSON."
    }
    return $text.Substring($start, $end - $start + 1) | ConvertFrom-Json
}

function Test-HttpUrl {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
        return [ordered]@{
            ok = $true
            status_code = [int]$response.StatusCode
            url = $Url
        }
    } catch {
        return [ordered]@{
            ok = $false
            error = $_.Exception.Message
            url = $Url
        }
    }
}

function Get-FirewallDiagnostics {
    param(
        [bool]$ShouldCheck,
        [string]$Reason
    )

    if (-not $ShouldCheck) {
        return [ordered]@{
            checked = $false
            reason = "avatar_url_reachable_from_pc"
            raw = @()
        }
    }

    $ruleNames = @("Node.js JavaScript Runtime", "node.exe")
    $raw = @()
    foreach ($ruleName in $ruleNames) {
        $raw += "=== $ruleName ==="
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & netsh advfirewall firewall show rule name="$ruleName" verbose 2>&1
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        $raw += @($output)
    }

    $joined = ($raw | Out-String)
    return [ordered]@{
        checked = $true
        reason = $Reason
        node_inbound_allow_seen = $joined -match "Direction:\s+In" -and $joined -match "Action:\s+Allow"
        raw = @($raw)
    }
}

function Get-UsbDiagnostics {
    param([string]$SnapshotPath)

    if (-not [string]::IsNullOrWhiteSpace($SnapshotPath)) {
        $devices = @(Get-Content -LiteralPath $SnapshotPath -Raw | ConvertFrom-Json)
    } else {
        $devices = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
            Where-Object {
                $_.FriendlyName -match "Samsung|Galaxy|ADB|Android" -or
                $_.InstanceId -match "VID_04E8"
            } |
            Select-Object Class, FriendlyName, Status, InstanceId)
    }

    $summaries = @()
    foreach ($device in $devices) {
        $friendlyName = [string]$device.FriendlyName
        $instanceId = [string]$device.InstanceId
        $className = [string]$device.Class
        $summaries += [ordered]@{
            class = $className
            friendly_name = $friendlyName
            status = [string]$device.Status
            instance_id = $instanceId
        }
    }

    $joined = ($summaries | ConvertTo-Json -Depth 4)
    $samsungPresent = $joined -match "Samsung|Galaxy|VID_04E8|SAMSUNG_ANDROID"
    $adbInterfacePresent = $joined -match "ADB|Android Debug Bridge|Android ADB Interface"
    $mtpPresent = $joined -match "MTP|WPD|MS_COMP_MTP|Galaxy S9"
    $diagnosis = "no_samsung_usb_device_seen"
    if ($samsungPresent -and $adbInterfacePresent) {
        $diagnosis = "adb_interface_present"
    } elseif ($samsungPresent -and $mtpPresent) {
        $diagnosis = "samsung_mtp_present_adb_interface_missing"
    } elseif ($samsungPresent) {
        $diagnosis = "samsung_usb_present_adb_interface_missing"
    }

    return [ordered]@{
        samsung_present = $samsungPresent
        mtp_present = $mtpPresent
        adb_interface_present = $adbInterfacePresent
        diagnosis = $diagnosis
        devices = $summaries
    }
}

function Get-SamsungDriverDiagnostics {
    param(
        [string]$SnapshotPath,
        [hashtable]$UsbDiagnostics
    )

    if (-not [string]::IsNullOrWhiteSpace($SnapshotPath)) {
        $installedApps = @(Get-Content -LiteralPath $SnapshotPath -Raw | ConvertFrom-Json)
    } else {
        $uninstallRoots = @(
            "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
            "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
        )
        $installedApps = @(Get-ItemProperty $uninstallRoots -ErrorAction SilentlyContinue |
            Where-Object {
                $_.DisplayName -match "Samsung.*USB|SAMSUNG.*USB|Android.*USB|Samsung Android USB Driver"
            } |
            Select-Object DisplayName, DisplayVersion, Publisher, InstallDate)
    }

    $matches = @()
    foreach ($app in @($installedApps | Where-Object { $null -ne $_ })) {
        if ([string]::IsNullOrWhiteSpace([string]$app.DisplayName)) {
            continue
        }
        $matches += [ordered]@{
            display_name = [string]$app.DisplayName
            display_version = [string]$app.DisplayVersion
            publisher = [string]$app.Publisher
            install_date = [string]$app.InstallDate
        }
    }

    $driverInstalled = $matches.Count -gt 0
    $needsDriverIfStillEmpty = (
        -not $driverInstalled -and
        $UsbDiagnostics.samsung_present -and
        -not $UsbDiagnostics.adb_interface_present
    )
    $recommendation = "none"
    if ($needsDriverIfStillEmpty) {
        $recommendation = "install_official_samsung_usb_driver_after_usb_debugging"
    }

    return [ordered]@{
        samsung_usb_driver_installed = $driverInstalled
        official_url = "https://developer.samsung.com/android-usb-driver"
        recommendation = $recommendation
        matches = $matches
    }
}

function Get-AdbDevices {
    param([string]$ResolvedAdb)

    if ([string]::IsNullOrWhiteSpace($ResolvedAdb)) {
        return [ordered]@{
            ok = $false
            error = "adb.exe was not found."
            devices = @()
        }
    }

    $raw = & $ResolvedAdb devices -l 2>&1
    $devices = @()
    foreach ($line in $raw) {
        $trimmed = [string]$line
        if ($trimmed -match "^\s*([^\s]+)\s+(device|unauthorized|offline)(.*)$") {
            $devices += [ordered]@{
                serial = $Matches[1]
                state = $Matches[2]
                detail = $Matches[3].Trim()
            }
        }
    }

    return [ordered]@{
        ok = $true
        raw = @($raw)
        devices = $devices
    }
}

function Wait-ForAdbDevice {
    param(
        [string]$ResolvedAdb,
        [int]$TimeoutSeconds
    )

    if ($TimeoutSeconds -le 0) {
        return [ordered]@{
            waited = $false
            attempts = 0
            reset_attempted = $false
            reset_recommended = $false
            final_state = "not_waited"
            devices = @()
        }
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $attempts = 0
    $resetAttempted = $false
    $lastDevices = $null
    do {
        $attempts += 1
        $lastDevices = Get-AdbDevices -ResolvedAdb $ResolvedAdb
        $states = @($lastDevices.devices | ForEach-Object { $_.state })
        if ($states -contains "device") {
            return [ordered]@{
                waited = $true
                attempts = $attempts
                reset_attempted = $resetAttempted
                reset_recommended = $false
                final_state = "device"
                devices = $lastDevices.devices
            }
        }
        if ($states -contains "unauthorized") {
            return [ordered]@{
                waited = $true
                attempts = $attempts
                reset_attempted = $resetAttempted
                reset_recommended = $false
                final_state = "unauthorized"
                devices = $lastDevices.devices
            }
        }
        if ($states -contains "offline") {
            return [ordered]@{
                waited = $true
                attempts = $attempts
                reset_attempted = $false
                reset_recommended = $true
                final_state = "offline"
                devices = $lastDevices.devices
            }
        }
        if ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    return [ordered]@{
        waited = $true
        attempts = $attempts
        reset_attempted = $resetAttempted
        reset_recommended = $false
        final_state = "none"
        devices = $lastDevices.devices
    }
}

function Invoke-ProcessQuiet {
    param(
        [string]$FilePath,
        [string]$Arguments,
        [int]$TimeoutMilliseconds = 5000
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $finished = $proc.WaitForExit($TimeoutMilliseconds)
    if (-not $finished) {
        try {
            $proc.Kill()
        } catch {
        }
        return [ordered]@{
            ok = $false
            exit_code = $null
            timed_out = $true
            output = @()
        }
    }
    return [ordered]@{
        ok = $proc.ExitCode -eq 0
        exit_code = $proc.ExitCode
        timed_out = $false
        output = @(
            $proc.StandardOutput.ReadToEnd(),
            $proc.StandardError.ReadToEnd()
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }
}

function Invoke-AdbChecked {
    param(
        [string]$ResolvedAdb,
        [string[]]$Arguments
    )

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $ResolvedAdb @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    return [ordered]@{
        ok = $exitCode -eq 0
        exit_code = $exitCode
        command = @($ResolvedAdb) + $Arguments
        output = @($raw | ForEach-Object { [string]$_ })
    }
}

function Get-ForegroundBrowserTaskId {
    param(
        [string]$ResolvedAdb,
        [string]$PackageName
    )

    $raw = & $ResolvedAdb shell dumpsys activity activities 2>&1
    foreach ($line in $raw) {
        $text = [string]$line
        if ($text -match "TaskRecord\{[^#]+#(\d+)\s+A=$([regex]::Escape($PackageName))\s") {
            return [ordered]@{
                ok = $true
                task_id = $Matches[1]
                raw = @($raw)
            }
        }
    }
    return [ordered]@{
        ok = $false
        error = "Could not find foreground task for $PackageName."
        raw = @($raw)
    }
}

$status = $null
if ([string]::IsNullOrWhiteSpace($AvatarUrl)) {
    $status = Get-HermesStatusJson
    $AvatarUrl = [string]$status.config.url
}

$adb = Find-Tool -Name "adb.exe" -ExplicitPath $AdbPath
$scrcpy = Find-Tool -Name "scrcpy.exe" -ExplicitPath $ScrcpyPath
$http = Test-HttpUrl -Url $AvatarUrl
$firewall = Get-FirewallDiagnostics -ShouldCheck (-not $http.ok) -Reason "avatar_url_unreachable"
$usb = Get-UsbDiagnostics -SnapshotPath $PnpSnapshotPath
$driver = Get-SamsungDriverDiagnostics -SnapshotPath $DriverSnapshotPath -UsbDiagnostics $usb
$adbDevices = Get-AdbDevices -ResolvedAdb $adb
$adbWait = Wait-ForAdbDevice -ResolvedAdb $adb -TimeoutSeconds $WaitForAdbSeconds
if ($adbWait.waited -and @($adbWait.devices).Count -gt 0) {
    $adbDevices = [ordered]@{
        ok = $true
        raw = $adbDevices.raw
        devices = $adbWait.devices
    }
}
$readyDevice = @($adbDevices.devices | Where-Object { $_.state -eq "device" } | Select-Object -First 1)

$deviceConfigResult = $null
if ($ConfigureDevice) {
    if ($readyDevice.Count -eq 0) {
        $deviceConfigResult = [ordered]@{
            ok = $false
            error = "No authorized adb device is available."
        }
    } else {
        $steps = @()
        if (-not [string]::IsNullOrWhiteSpace($BrowserPackage)) {
            $steps += (Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "pm", "grant", $BrowserPackage, "android.permission.RECORD_AUDIO"))
        }
        $steps += @(
            (Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "input", "keyevent", "KEYCODE_WAKEUP")),
            (Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "settings", "put", "global", "stay_on_while_plugged_in", "3")),
            (Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "settings", "put", "system", "screen_off_timeout", "2147483647")),
            (Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "settings", "put", "secure", "lock_to_app_enabled", "1"))
        )
        $deviceConfigResult = [ordered]@{
            ok = @($steps | Where-Object { -not $_.ok }).Count -eq 0
            mode = "kiosk-light"
            note = "Keeps the Galaxy awake while powered, enables Android screen pinning, and leaves the VRM page in the foreground; true Android lock-task kiosk mode requires device-owner provisioning."
            microphone_route = "Use the VRM app's microphone button in the selected Chromium browser; the app already uses Web Speech Recognition with ja-JP."
            steps = $steps
        }
    }
}

$openResult = $null
if ($OpenOnDevice) {
    if ($readyDevice.Count -eq 0) {
        $openResult = [ordered]@{
            ok = $false
            error = "No authorized adb device is available."
        }
    } else {
        if ($LockTask) {
            [void](Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "am", "task", "lock", "stop"))
        }
        $openArgs = @("shell", "am", "start")
        if (-not [string]::IsNullOrWhiteSpace($BrowserPackage)) {
            $openArgs += @("-p", $BrowserPackage)
        }
        $openArgs += @("-a", "android.intent.action.VIEW", "-d", $AvatarUrl)
        $openResult = Invoke-AdbChecked -ResolvedAdb $adb -Arguments $openArgs
    }
}

$lockTaskResult = $null
if ($LockTask) {
    if ($readyDevice.Count -eq 0) {
        $lockTaskResult = [ordered]@{
            ok = $false
            error = "No authorized adb device is available."
        }
    } else {
        $task = Get-ForegroundBrowserTaskId -ResolvedAdb $adb -PackageName $BrowserPackage
        if (-not $task.ok) {
            $lockTaskResult = $task
        } else {
            $lock = Invoke-AdbChecked -ResolvedAdb $adb -Arguments @("shell", "am", "task", "lock", $task.task_id)
            $lockTaskResult = [ordered]@{
                ok = $lock.ok
                task_id = $task.task_id
                command = $lock.command
                output = $lock.output
            }
        }
    }
}

$scrcpyResult = $null
if ($LaunchScrcpy) {
    if ([string]::IsNullOrWhiteSpace($scrcpy)) {
        $scrcpyResult = [ordered]@{
            ok = $false
            error = "scrcpy.exe was not found."
        }
    } else {
        if ($readyDevice.Count -eq 0) {
            $scrcpyResult = [ordered]@{
                ok = $false
                error = "No authorized adb device is available."
            }
        } else {
            $proc = Start-Process -FilePath $scrcpy -ArgumentList @("--stay-awake", "--turn-screen-on") -PassThru
            $scrcpyResult = [ordered]@{
                ok = $true
                pid = $proc.Id
                path = $scrcpy
            }
        }
    }
}

$authorized = @($adbDevices.devices | Where-Object { $_.state -eq "device" }).Count -gt 0
$unauthorized = @($adbDevices.devices | Where-Object { $_.state -eq "unauthorized" }).Count -gt 0
$offline = @($adbDevices.devices | Where-Object { $_.state -eq "offline" }).Count -gt 0

$nextActions = @()
if (-not $http.ok) {
    $nextActions += "Check Windows Defender Firewall for Node/Vite on port 5175 and confirm the phone is on the same network."
}
if (-not $authorized) {
    $nextActions += "On Galaxy S9: enable Developer options, enable USB debugging, reconnect USB, and accept the RSA fingerprint prompt."
}
if (-not $authorized -and $usb.samsung_present -and -not $usb.adb_interface_present) {
    $nextActions += "Windows sees the Galaxy over USB/MTP, but no ADB interface is present. After USB debugging is enabled, install the official Samsung Android USB Driver if adb still lists no device."
}
if (-not $authorized -and $driver.recommendation -eq "install_official_samsung_usb_driver_after_usb_debugging") {
    $nextActions += "Samsung Android USB Driver was not found in installed programs. Use the official Samsung driver page if USB debugging is enabled but adb remains empty."
}
if ($unauthorized) {
    $nextActions += "Galaxy is visible but unauthorized. Unlock the phone and tap Allow on the USB debugging prompt."
}
if ($offline) {
    $nextActions += "Galaxy is visible but offline. Reconnect USB, unlock the phone, then run adb kill-server/start-server or rerun this script."
}
if ($authorized -and -not $OpenOnDevice) {
    $nextActions += "Run again with -OpenOnDevice to open the VRM URL on the Galaxy browser."
}
if ($authorized -and -not $ConfigureDevice) {
    $nextActions += "Run again with -ConfigureDevice to keep the Galaxy awake while plugged in and document the microphone route."
}
if ($authorized -and -not $LaunchScrcpy) {
    $nextActions += "Run again with -LaunchScrcpy to mirror and keep the Galaxy screen awake."
}

[ordered]@{
    ok = $http.ok -and $authorized
    avatar_url = $AvatarUrl
    hermes_status_loaded = $null -ne $status
    tools = [ordered]@{
        adb = $adb
        scrcpy = $scrcpy
    }
    http = $http
    firewall = $firewall
    usb = $usb
    driver = $driver
    adb = $adbDevices
    adb_wait = $adbWait
    device_config = $deviceConfigResult
    open_on_device = $openResult
    lock_task = $lockTaskResult
    scrcpy = $scrcpyResult
    next_actions = $nextActions
} | ConvertTo-Json -Depth 8
