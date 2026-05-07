<#
.SYNOPSIS
Capture the Windows desktop or an emulator window and send mouse/keyboard input.

.DESCRIPTION
This host-side harness is useful when an Android emulator is visible on the
laptop but ADB screenshots are unavailable, blank, or too narrow for UI review.
It works with the Android SDK emulator, BlueStacks, LDPlayer, MEmu, Nox, and
other Windows apps because it operates on normal Windows windows and screen
coordinates.

.EXAMPLES
.\scripts\windows-visual-control.ps1 -Action status
.\scripts\windows-visual-control.ps1 -Action screenshot
.\scripts\windows-visual-control.ps1 -Action window-screenshot -ProcessName emulator
.\scripts\windows-visual-control.ps1 -Action click -ProcessName emulator -Relative -X 420 -Y 760
.\scripts\windows-visual-control.ps1 -Action type -ProcessName emulator -Text "write a flappy bird html game" -UseClipboard
#>
[CmdletBinding()]
param(
    [ValidateSet("status", "screenshot", "window-screenshot", "move", "click", "type", "key")]
    [string]$Action = "screenshot",

    [string]$WindowTitle,
    [string]$ProcessName,

    [int]$X = -1,
    [int]$Y = -1,
    [switch]$Relative,

    [ValidateSet("left", "right", "middle")]
    [string]$Button = "left",

    [string]$Text,
    [string]$Keys,
    [switch]$UseClipboard,

    [string]$OutDir = "artifacts/windows-visual",
    [string]$Out,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

$nativeSource = @"
using System;
using System.Runtime.InteropServices;

public static class HermesWin32Visual {
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT {
        public int X;
        public int Y;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    public static extern bool GetCursorPos(out POINT point);

    [DllImport("user32.dll")]
    public static extern void mouse_event(int flags, int dx, int dy, int data, UIntPtr extraInfo);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@

if (-not ([System.Management.Automation.PSTypeName]"HermesWin32Visual").Type) {
    Add-Type -TypeDefinition $nativeSource
}

$MouseDown = @{
    left = 0x0002
    right = 0x0008
    middle = 0x0020
}
$MouseUp = @{
    left = 0x0004
    right = 0x0010
    middle = 0x0040
}

function Get-TargetWindow {
    if (-not $WindowTitle -and -not $ProcessName) {
        return $null
    }

    $processes = Get-Process | Where-Object {
        $_.MainWindowHandle -ne 0 -and
        (
            (-not $ProcessName -or $_.ProcessName -like $ProcessName -or $_.ProcessName -like "$ProcessName*") -and
            (-not $WindowTitle -or $_.MainWindowTitle -like "*$WindowTitle*")
        )
    } | Sort-Object ProcessName, Id

    $target = $processes | Select-Object -First 1
    if (-not $target) {
        $selectors = @()
        if ($ProcessName) { $selectors += "ProcessName=$ProcessName" }
        if ($WindowTitle) { $selectors += "WindowTitle=$WindowTitle" }
        throw "No visible window matched $($selectors -join ', ')."
    }
    return $target
}

function Get-WindowRect {
    param([System.Diagnostics.Process]$Process)

    $rect = New-Object HermesWin32Visual+RECT
    if (-not [HermesWin32Visual]::GetWindowRect($Process.MainWindowHandle, [ref]$rect)) {
        throw "Unable to read window rectangle for process $($Process.ProcessName) [$($Process.Id)]."
    }
    [PSCustomObject]@{
        Left = $rect.Left
        Top = $rect.Top
        Right = $rect.Right
        Bottom = $rect.Bottom
        Width = [Math]::Max(0, $rect.Right - $rect.Left)
        Height = [Math]::Max(0, $rect.Bottom - $rect.Top)
    }
}

function Resolve-Point {
    Require-Coordinate -Name "X" -Value $X
    Require-Coordinate -Name "Y" -Value $Y
    if ($Relative) {
        $target = Get-TargetWindow
        if (-not $target) {
            throw "-Relative requires -WindowTitle or -ProcessName."
        }
        $rect = Get-WindowRect -Process $target
        return [PSCustomObject]@{ X = $rect.Left + $X; Y = $rect.Top + $Y; Target = $target }
    }
    [PSCustomObject]@{ X = $X; Y = $Y; Target = $null }
}

function Require-Coordinate {
    param([string]$Name, [int]$Value)
    if ($Value -lt 0) {
        throw "$Name must be zero or greater."
    }
}

function Get-OutputPath {
    param([string]$Prefix)

    if ($Out) {
        $path = $Out
    } else {
        if (-not (Test-Path -LiteralPath $OutDir)) {
            New-Item -ItemType Directory -Path $OutDir | Out-Null
        }
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $path = Join-Path $OutDir "$Prefix-$timestamp.png"
    }
    $parent = Split-Path -Parent $path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    return $path
}

function Save-BoundsScreenshot {
    param(
        [int]$Left,
        [int]$Top,
        [int]$Width,
        [int]$Height,
        [string]$Path
    )

    if ($Width -le 0 -or $Height -le 0) {
        throw "Screenshot bounds must have positive width and height."
    }

    $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($Left, $Top, 0, 0, $bitmap.Size)
        $bitmap.Save((Resolve-Path -LiteralPath (Split-Path -Parent $Path)).Path + "\" + (Split-Path -Leaf $Path), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($Open) {
        Invoke-Item -LiteralPath $resolved
    }
    $resolved
}

function Get-CursorStatus {
    $point = New-Object HermesWin32Visual+POINT
    [void][HermesWin32Visual]::GetCursorPos([ref]$point)
    [PSCustomObject]@{ X = $point.X; Y = $point.Y }
}

function Get-VisibleWindows {
    Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } |
        Sort-Object ProcessName, Id |
        Select-Object Id, ProcessName, MainWindowTitle, Path
}

function Set-TargetForeground {
    param([System.Diagnostics.Process]$Process)
    if ($Process) {
        [void][HermesWin32Visual]::SetForegroundWindow($Process.MainWindowHandle)
        Start-Sleep -Milliseconds 150
    }
}

switch ($Action) {
    "status" {
        $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
        [PSCustomObject]@{
            virtual_screen = [PSCustomObject]@{
                left = $bounds.Left
                top = $bounds.Top
                width = $bounds.Width
                height = $bounds.Height
            }
            cursor = Get-CursorStatus
            matching_windows = @(Get-VisibleWindows | Where-Object {
                $_.ProcessName -match "emulator|qemu|BlueStacks|HD-|Nox|dnplayer|MEmu|Genymotion|LDPlayer|chrome"
            })
        } | ConvertTo-Json -Depth 5
    }
    "screenshot" {
        $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
        $path = Get-OutputPath -Prefix "desktop"
        Save-BoundsScreenshot -Left $bounds.Left -Top $bounds.Top -Width $bounds.Width -Height $bounds.Height -Path $path
    }
    "window-screenshot" {
        $target = Get-TargetWindow
        $rect = Get-WindowRect -Process $target
        $safeName = ($target.ProcessName -replace "[^A-Za-z0-9._-]", "_")
        $path = Get-OutputPath -Prefix "window-$safeName-$($target.Id)"
        Save-BoundsScreenshot -Left $rect.Left -Top $rect.Top -Width $rect.Width -Height $rect.Height -Path $path
    }
    "move" {
        $point = Resolve-Point
        [void][HermesWin32Visual]::SetCursorPos($point.X, $point.Y)
        Get-CursorStatus | ConvertTo-Json
    }
    "click" {
        $point = Resolve-Point
        if ($point.Target) {
            Set-TargetForeground -Process $point.Target
        }
        [void][HermesWin32Visual]::SetCursorPos($point.X, $point.Y)
        [HermesWin32Visual]::mouse_event($MouseDown[$Button], 0, 0, 0, [UIntPtr]::Zero)
        Start-Sleep -Milliseconds 60
        [HermesWin32Visual]::mouse_event($MouseUp[$Button], 0, 0, 0, [UIntPtr]::Zero)
    }
    "type" {
        if ([string]::IsNullOrEmpty($Text)) {
            throw "-Text is required for Action type."
        }
        Set-TargetForeground -Process (Get-TargetWindow)
        if ($UseClipboard) {
            Set-Clipboard -Value $Text
            [System.Windows.Forms.SendKeys]::SendWait("^v")
        } else {
            [System.Windows.Forms.SendKeys]::SendWait($Text)
        }
    }
    "key" {
        if ([string]::IsNullOrEmpty($Keys)) {
            throw "-Keys is required for Action key, for example '{ENTER}' or '^l'."
        }
        Set-TargetForeground -Process (Get-TargetWindow)
        [System.Windows.Forms.SendKeys]::SendWait($Keys)
    }
}
