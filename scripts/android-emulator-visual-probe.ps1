<#
.SYNOPSIS
Capture, launch, tap, type, and swipe an ADB-visible Android emulator or phone.

.EXAMPLES
.\scripts\android-emulator-visual-probe.ps1 -Action status
.\scripts\android-emulator-visual-probe.ps1 -Action launch
.\scripts\android-emulator-visual-probe.ps1 -Action screenshot -Open
.\scripts\android-emulator-visual-probe.ps1 -Action tap -X 520 -Y 1810
.\scripts\android-emulator-visual-probe.ps1 -Action text -Text "write a flappy bird html game"
.\scripts\android-emulator-visual-probe.ps1 -Action swipe -X 500 -Y 1800 -X2 500 -Y 400
#>
[CmdletBinding()]
param(
    [ValidateSet("status", "launch", "screenshot", "tap", "text", "keyevent", "swipe")]
    [string]$Action = "screenshot",

    [string]$Serial,

    [int]$X = -1,
    [int]$Y = -1,
    [int]$X2 = -1,
    [int]$Y2 = -1,
    [int]$DurationMs = 300,

    [string]$Text,
    [int]$KeyCode = 66,

    [string]$Package = "com.nousresearch.hermesagent",
    [string]$Activity,

    [string]$OutDir = "artifacts/emulator-visual",
    [switch]$Open
)

$ErrorActionPreference = "Stop"

function Resolve-Adb {
    $candidates = @()
    if ($env:ANDROID_HOME) {
        $candidates += Join-Path $env:ANDROID_HOME "platform-tools\adb.exe"
    }
    if ($env:ANDROID_SDK_ROOT) {
        $candidates += Join-Path $env:ANDROID_SDK_ROOT "platform-tools\adb.exe"
    }
    $candidates += "C:\Users\Ady\Documents\Codex\2026-05-02\c-users-ady-downloads-hermes-android\_android_sdk\platform-tools\adb.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $command = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "adb.exe was not found. Set ANDROID_HOME or ANDROID_SDK_ROOT, or put adb.exe on PATH."
}

function Get-AdbDevice {
    param([string]$RequestedSerial)

    $deviceLines = & $script:Adb devices | Select-Object -Skip 1 | Where-Object { $_ -match "\S+\s+device\b" }
    $serials = @($deviceLines | ForEach-Object { ($_ -split "\s+")[0] })
    if ($serials.Count -eq 0) {
        throw "No ADB device is online. Start an emulator or connect a phone, then rerun this script."
    }

    if ($RequestedSerial) {
        if ($serials -notcontains $RequestedSerial) {
            throw "Requested ADB serial '$RequestedSerial' is not online. Online devices: $($serials -join ', ')"
        }
        return $RequestedSerial
    }

    $emulator = $serials | Where-Object { $_ -like "emulator-*" } | Select-Object -First 1
    if ($emulator) {
        return $emulator
    }

    if ($serials.Count -eq 1) {
        return $serials[0]
    }

    throw "Multiple ADB devices are online. Pass -Serial. Online devices: $($serials -join ', ')"
}

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $script:Adb -s $script:Serial @Arguments
}

function Require-Coordinate {
    param([string]$Name, [int]$Value)
    if ($Value -lt 0) {
        throw "$Name must be zero or greater."
    }
}

function Convert-TextForAdbInput {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) {
        throw "-Text is required for Action text."
    }
    return ($Value -replace " ", "%s" -replace '"', '\"' -replace "'", "\'")
}

$script:Adb = Resolve-Adb
$script:Serial = Get-AdbDevice -RequestedSerial $Serial

switch ($Action) {
    "status" {
        $wmSize = (Invoke-Adb "shell" "wm" "size") -join "`n"
        $wmDensity = (Invoke-Adb "shell" "wm" "density") -join "`n"
        $sdk = (Invoke-Adb "shell" "getprop" "ro.build.version.sdk") -join "`n"
        $model = (Invoke-Adb "shell" "getprop" "ro.product.model") -join "`n"
        $packagePath = (Invoke-Adb "shell" "pm" "path" $Package) -join "`n"
        [PSCustomObject]@{
            adb = $script:Adb
            serial = $script:Serial
            model = $model.Trim()
            sdk = $sdk.Trim()
            wm_size = $wmSize.Trim()
            wm_density = $wmDensity.Trim()
            package = $Package
            package_path = $packagePath.Trim()
        } | ConvertTo-Json -Depth 4
    }
    "launch" {
        if ($Activity) {
            Invoke-Adb "shell" "am" "start" "-n" "$Package/$Activity"
        } else {
            Invoke-Adb "shell" "monkey" "-p" $Package "-c" "android.intent.category.LAUNCHER" "1"
        }
    }
    "screenshot" {
        if (!(Test-Path -LiteralPath $OutDir)) {
            New-Item -ItemType Directory -Path $OutDir | Out-Null
        }
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $fileName = "hermes-$($script:Serial)-$timestamp.png"
        $localPath = Join-Path $OutDir $fileName
        $remotePath = "/sdcard/$fileName"
        Invoke-Adb "shell" "screencap" "-p" $remotePath | Out-Null
        Invoke-Adb "pull" $remotePath $localPath | Out-Null
        Invoke-Adb "shell" "rm" $remotePath | Out-Null
        $resolved = (Resolve-Path -LiteralPath $localPath).Path
        if ($Open) {
            Invoke-Item -LiteralPath $resolved
        }
        $resolved
    }
    "tap" {
        Require-Coordinate -Name "X" -Value $X
        Require-Coordinate -Name "Y" -Value $Y
        Invoke-Adb "shell" "input" "tap" "$X" "$Y"
    }
    "text" {
        $encoded = Convert-TextForAdbInput -Value $Text
        Invoke-Adb "shell" "input" "text" $encoded
    }
    "keyevent" {
        Invoke-Adb "shell" "input" "keyevent" "$KeyCode"
    }
    "swipe" {
        Require-Coordinate -Name "X" -Value $X
        Require-Coordinate -Name "Y" -Value $Y
        Require-Coordinate -Name "X2" -Value $X2
        Require-Coordinate -Name "Y2" -Value $Y2
        Invoke-Adb "shell" "input" "swipe" "$X" "$Y" "$X2" "$Y2" "$DurationMs"
    }
}
