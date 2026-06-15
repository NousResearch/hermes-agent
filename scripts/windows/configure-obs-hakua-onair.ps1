param(
    [string]$AvatarUrl = "http://127.0.0.1:5175/",
    [string]$CollectionName = "Hakua OnAir",
    [string]$ProfileName = "Hakua OnAir",
    [int]$Width = 1920,
    [int]$Height = 1080,
    [int]$Fps = 60,
    [switch]$Launch,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-ObsExe {
    $candidates = @(
        "$env:ProgramFiles\obs-studio\bin\64bit\obs64.exe",
        "${env:ProgramFiles(x86)}\obs-studio\bin\64bit\obs64.exe",
        "${env:ProgramFiles(x86)}\Steam\steamapps\common\OBS Studio\bin\64bit\obs64.exe"
    )

    $roots = @(
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    foreach ($root in $roots) {
        Get-ItemProperty $root -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -match "OBS" -and $_.InstallLocation } |
            ForEach-Object {
                $candidates += (Join-Path $_.InstallLocation "bin\64bit\obs64.exe")
            }
    }

    foreach ($path in $candidates | Select-Object -Unique) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Backup-IfNeeded {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Copy-Item -LiteralPath $Path -Destination "$Path.bak-$stamp" -Force
    }
}

$obsRoot = Join-Path $env:APPDATA "obs-studio"
$sceneDir = Join-Path $obsRoot "basic\scenes"
$profileDir = Join-Path $obsRoot "basic\profiles\$ProfileName"
New-Item -ItemType Directory -Force -Path $sceneDir, $profileDir | Out-Null

$scenePath = Join-Path $sceneDir "$CollectionName.json"
$profilePath = Join-Path $profileDir "basic.ini"
$servicePath = Join-Path $profileDir "service.json"

if ((Test-Path -LiteralPath $scenePath) -and -not $Force) {
    throw "Scene collection already exists: $scenePath. Re-run with -Force to update it after backup."
}

Backup-IfNeeded -Path $scenePath
Backup-IfNeeded -Path $profilePath
Backup-IfNeeded -Path $servicePath

$sceneUuid = [guid]::NewGuid().ToString()
$browserUuid = [guid]::NewGuid().ToString()
$canvasUuid = [guid]::NewGuid().ToString()

$browserSource = [ordered]@{
    prev_ver = 536936449
    name = "Hakua Browser"
    uuid = $browserUuid
    id = "browser_source"
    versioned_id = "browser_source"
    settings = [ordered]@{
        url = $AvatarUrl
        width = $Width
        height = $Height
        fps = $Fps
        css = ""
        shutdown = $false
        restart_when_active = $true
        reroute_audio = $false
    }
    mixers = 255
    sync = 0
    flags = 0
    volume = 1.0
    balance = 0.5
    enabled = $true
    muted = $false
    "push-to-mute" = $false
    "push-to-mute-delay" = 0
    "push-to-talk" = $false
    "push-to-talk-delay" = 0
    hotkeys = [ordered]@{}
    deinterlace_mode = 0
    deinterlace_field_order = 0
    monitoring_type = 0
    private_settings = [ordered]@{}
}

$sceneSource = [ordered]@{
    prev_ver = 536936449
    name = "Hakua OnAir"
    uuid = $sceneUuid
    id = "scene"
    versioned_id = "scene"
    settings = [ordered]@{
        id_counter = 1
        custom_size = $false
        items = @(
            [ordered]@{
                name = "Hakua Browser"
                source_uuid = $browserUuid
                visible = $true
                locked = $false
                rot = 0.0
                pos = [ordered]@{ x = 0.0; y = 0.0 }
                scale = [ordered]@{ x = 1.0; y = 1.0 }
                align = 5
                bounds_type = 2
                bounds_align = 0
                bounds = [ordered]@{ x = [double]$Width; y = [double]$Height }
                crop_left = 0
                crop_top = 0
                crop_right = 0
                crop_bottom = 0
                id = 1
            }
        )
    }
    mixers = 0
    sync = 0
    flags = 0
    volume = 1.0
    balance = 0.5
    enabled = $true
    muted = $false
    "push-to-mute" = $false
    "push-to-mute-delay" = 0
    "push-to-talk" = $false
    "push-to-talk-delay" = 0
    hotkeys = [ordered]@{ "OBSBasic.SelectScene" = @() }
    deinterlace_mode = 0
    deinterlace_field_order = 0
    monitoring_type = 0
    canvas_uuid = $canvasUuid
    private_settings = [ordered]@{}
}

$sceneCollection = [ordered]@{
    name = $CollectionName
    sources = @($browserSource, $sceneSource)
    groups = @()
    scene_order = @([ordered]@{ name = "Hakua OnAir" })
    current_scene = "Hakua OnAir"
    current_program_scene = "Hakua OnAir"
    canvases = @()
    current_transition = "Fade"
    transition_duration = 300
    transitions = @()
    quick_transitions = @()
    saved_projectors = @()
    preview_locked = $false
    scaling_enabled = $false
    scaling_level = 0
    scaling_off_x = 0.0
    scaling_off_y = 0.0
    "virtual-camera" = [ordered]@{ type2 = 3 }
    modules = [ordered]@{}
    version = 2
}

$sceneCollection | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $scenePath -Encoding UTF8

@"
[General]
Name=$ProfileName

[Output]
Mode=Simple
Reconnect=true
RetryDelay=2
MaxRetries=25
BindIP=default
IPFamily=IPv4+IPv6

[SimpleOutput]
FilePath=$env:USERPROFILE\\Videos
RecFormat2=hybrid_mp4
VBitrate=10000
ABitrate=160
StreamAudioEncoder=aac
StreamEncoder=nvenc
RecEncoder=nvenc

[Video]
BaseCX=$Width
BaseCY=$Height
OutputCX=$Width
OutputCY=$Height
FPSType=0
FPSCommon=$Fps
ScaleType=bicubic
ColorFormat=NV12
ColorSpace=709
ColorRange=Partial

[Audio]
MonitoringDeviceId=default
MonitoringDeviceName=Default
SampleRate=48000
ChannelSetup=Stereo
"@ | Set-Content -LiteralPath $profilePath -Encoding UTF8

@"
{"type":"rtmp_common","settings":{"service":"YouTube - RTMPS","server":"rtmps://a.rtmps.youtube.com:443/live2","protocol":"RTMPS","stream_key_link":"https://www.youtube.com/live_dashboard"}}
"@ | Set-Content -LiteralPath $servicePath -Encoding UTF8

$obsExe = Get-ObsExe
$result = [ordered]@{
    ok = $true
    obs_exe = $obsExe
    collection = $CollectionName
    profile = $ProfileName
    avatar_url = $AvatarUrl
    scene_path = $scenePath
    profile_path = $profilePath
    service_path = $servicePath
}

if ($Launch) {
    if (-not $obsExe) {
        throw "OBS executable was not found."
    }
    Start-Process -FilePath $obsExe -ArgumentList @("--collection", $CollectionName, "--profile", $ProfileName)
    $result.launched = $true
}

$result | ConvertTo-Json -Depth 6
