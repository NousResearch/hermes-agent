param(
    [string]$PythonVersion = "3.13.12",
    [string]$WorkDir = "$env:LOCALAPPDATA\hermes-agent-bootstrap",
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Dim($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Fail($msg) { throw $msg }
function Ensure-Directory($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

$LogDir = Join-Path $WorkDir 'logs'
Ensure-Directory $WorkDir
Ensure-Directory $LogDir
$LogStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$TranscriptLog = Join-Path $LogDir ("bootstrap-{0}.log" -f $LogStamp)
$BuildLog = Join-Path $LogDir ("cpython-build-{0}.log" -f $LogStamp)
$BuildToolsLog = Join-Path $LogDir ("vs-buildtools-{0}.log" -f $LogStamp)
Start-Transcript -Path $TranscriptLog -Force | Out-Null

try {

function Get-HermesContext {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { $pwd.Path }
    $repoRoot = Split-Path -Parent $scriptDir
    $fromRepo = (Test-Path (Join-Path $repoRoot '.git')) -and (Test-Path (Join-Path $repoRoot 'pyproject.toml'))

    if ($fromRepo) {
        $branch = 'windows-qol-v2'
        try {
            $branch = (& git -C $repoRoot rev-parse --abbrev-ref HEAD 2>$null)
            if (-not $branch) { $branch = 'windows-qol-v2' }
        } catch {}
        return @{
            FromRepo = $true
            RepoRoot = $repoRoot
            Branch = $branch
            RawBase = "https://raw.githubusercontent.com/claudlos/hermes-agent/$branch"
        }
    }

    return @{
        FromRepo = $false
        RepoRoot = $null
        Branch = 'windows-qol-v2'
        RawBase = 'https://raw.githubusercontent.com/claudlos/hermes-agent/windows-qol-v2'
    }
}

function Download-File($url, $dest) {
    Write-Dim "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
}

function Get-CPythonZipUrl($version) {
    return "https://github.com/python/cpython/archive/refs/tags/v$version.zip"
}

function Get-VswherePath {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Get-BuildToolsInstallPath {
    $vswhere = Get-VswherePath
    if ($vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            return $installPath.Trim()
        }
    }

    $fallbacks = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools",
        "$env:ProgramFiles\Microsoft Visual Studio\18\BuildTools"
    )
    foreach ($candidate in $fallbacks) {
        if (Test-Path (Join-Path $candidate 'VC\Auxiliary\Build\vcvarsall.bat')) {
            return $candidate
        }
    }
    return $null
}

function Get-BuildToolsExitHint {
    param([int]$ExitCode)

    switch ($ExitCode) {
        0 { return 'Installer reported success.' }
        1602 { return 'The installer was cancelled, often because the UAC prompt was denied.' }
        1618 { return 'Another installer is already running. Finish it, then retry.' }
        3010 { return 'Install succeeded but requests a reboot before the toolchain is usable.' }
        default { return 'Check the Visual Studio installer log for component and setup details.' }
    }
}

function Ensure-MinFreeSpaceGB {
    param(
        [string]$Path,
        [int]$RequiredGB
    )

    $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
    $drive = New-Object System.IO.DriveInfo($root)
    $freeGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
    if ($freeGB -lt $RequiredGB) {
        Fail "Not enough free disk space on $root. Need ${RequiredGB}GB free, found ${freeGB}GB."
    }
    Write-Ok "Disk space check passed: ${freeGB}GB free on $root"
}

function Ensure-BuildTools {
    param([string]$InstallerCacheDir)

    $existing = Get-BuildToolsInstallPath
    if ($existing) {
        Write-Ok "Visual Studio Build Tools detected: $existing"
        return $existing
    }

    Write-Dim 'Visual Studio Build Tools not found; attempting bootstrap install'
    Ensure-Directory $InstallerCacheDir
    $bootstrapper = Join-Path $InstallerCacheDir 'vs_BuildTools.exe'
    if (-not (Test-Path $bootstrapper)) {
        Download-File -url 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -dest $bootstrapper
    }

    $installPath = "$env:ProgramFiles(x86)\Microsoft Visual Studio\2022\BuildTools"
    $args = @(
        '--quiet', '--wait', '--norestart', '--nocache',
        '--installPath', $installPath,
        '--add', 'Microsoft.VisualStudio.Workload.VCTools',
        '--includeRecommended',
        '--log', $BuildToolsLog
    )
    Write-Dim "Running Build Tools installer; log: $BuildToolsLog"
    $proc = Start-Process -FilePath $bootstrapper -ArgumentList $args -Verb RunAs -Wait -PassThru
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        $hint = Get-BuildToolsExitHint -ExitCode $proc.ExitCode
        Fail "Visual Studio Build Tools install failed with exit code $($proc.ExitCode). $hint See $BuildToolsLog"
    }
    if ($proc.ExitCode -eq 3010) {
        Write-Dim 'Build Tools installer requested a reboot; continuing with detection now.'
    }

    $detected = Get-BuildToolsInstallPath
    if (-not $detected) {
        Fail "Build Tools installer completed but the toolchain was still not detected. See $BuildToolsLog"
    }
    Write-Ok "Visual Studio Build Tools installed: $detected"
    return $detected
}

function Test-CustomPython {
    param(
        [string]$PythonExe,
        [string]$ExpectedVersion
    )

    if (-not (Test-Path $PythonExe)) {
        return $false
    }
    & $PythonExe -c "import socket,sys; expected = '$ExpectedVersion'; assert hasattr(socket, 'AF_UNIX'); s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.close(); assert sys.version.startswith(expected)"
    return ($LASTEXITCODE -eq 0)
}

function Write-BootstrapMetadata {
    param(
        [string]$Path,
        [string]$PythonExe,
        [string]$PythonVersion,
        [string]$BuildLog,
        [string]$TranscriptLog
    )

    $payload = @{
        python_exe = $PythonExe
        python_version = $PythonVersion
        verified = $true
        built_at = (Get-Date).ToString('o')
        build_log = $BuildLog
        transcript_log = $TranscriptLog
    }
    $payload | ConvertTo-Json | Set-Content -Path $Path -Encoding UTF8
}

function Read-BootstrapMetadata {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }
    try {
        return Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Replace-InFileRegex {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Replacement
    )
    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    if ([regex]::IsMatch($content, [regex]::Escape($Replacement))) {
        return $false
    }
    $updated = [regex]::Replace($content, $Pattern, $Replacement, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($updated -eq $content) {
        Fail "Expected patch context not found in $Path"
    }
    Set-Content -Path $Path -Value $updated -Encoding UTF8
    return $true
}

function Apply-AfUnixPatch {
    param([string]$SourceDir)

    $pyconfig = Join-Path $SourceDir 'PC\pyconfig.h.in'
    $socketH = Join-Path $SourceDir 'Modules\socketmodule.h'
    $socketC = Join-Path $SourceDir 'Modules\socketmodule.c'

    $replacement1 = @"
/* Define if you have the <sys/un.h> header file.  */
/* #define HAVE_SYS_UN_H 1 */

/* Windows SDK AF_UNIX support */
#if defined(__has_include) && __has_include(<afunix.h>)
#  define HAVE_AFUNIX_H 1
#else
#  define HAVE_AFUNIX_H 0
#endif

/* Define if you have the <sys/utime.h> header file.  */
"@

    $replacement2 = @"
#ifdef HAVE_SYS_UN_H
#  include <sys/un.h>
#elif HAVE_AFUNIX_H
#  include <afunix.h>
#else
#  undef AF_UNIX
#endif
"@

    $replacement3 = @"
static FlagRuntimeInfo win_runtime_flags[] = {
    /* available starting with Windows 10 1803 */
    {17134, "AF_UNIX"},
    /* available starting with Windows 10 1709 */
    {16299, "TCP_KEEPIDLE"},
"@

    $changed = $false
    $changed = (Replace-InFileRegex -Path $pyconfig -Pattern '(?ms)/\* Define if you have the <sys/un\.h> header file\.  \*/\r?\n/\* #define HAVE_SYS_UN_H 1 \*/\r?\n\r?\n/\* Define if you have the <sys/utime\.h> header file\.  \*/' -Replacement $replacement1) -or $changed
    $changed = (Replace-InFileRegex -Path $socketH -Pattern '(?ms)#ifdef HAVE_SYS_UN_H\r?\n# include <sys/un\.h>\r?\n#else\r?\n# undef AF_UNIX\r?\n#endif' -Replacement $replacement2) -or $changed
    $changed = (Replace-InFileRegex -Path $socketC -Pattern '(?ms)static FlagRuntimeInfo win_runtime_flags\[\] = \{\r?\n\s*/\* available starting with Windows 10 1709 \*/\r?\n\s*\{16299, "TCP_KEEPIDLE"\},' -Replacement $replacement3) -or $changed
    $changed = (Replace-InFileRegex -Path $socketC -Pattern '\*len_ret = path\.len \+ offsetof\(struct sockaddr_un, sun_path\) \+ 1;' -Replacement '*len_ret = (int)path.len + offsetof(struct sockaddr_un, sun_path) + 1;') -or $changed

    if ($changed) {
        Write-Ok 'AF_UNIX patch applied'
    } else {
        Write-Dim 'AF_UNIX patch already present'
    }
}

$ctx = Get-HermesContext
$zipDir = Join-Path $WorkDir 'downloads'
$srcRoot = Join-Path $WorkDir 'src'
$installerCacheDir = Join-Path $WorkDir 'installers'
$extractRoot = Join-Path $srcRoot "cpython-$PythonVersion"
$zipPath = Join-Path $zipDir "cpython-$PythonVersion.zip"
$metadataPath = Join-Path $WorkDir 'custom-python.json'
$installScriptLocal = if ($ctx.FromRepo) { Join-Path $ctx.RepoRoot 'scripts\install-windows.ps1' } else { Join-Path $WorkDir 'install-windows.ps1' }
$pythonExe = Join-Path $extractRoot 'PCbuild\amd64\python.exe'
$finalPythonExe = $null
$installMode = 'custom-build'
$customFailure = $null

Write-Host ''
Write-Host 'Hermes Windows AF_UNIX bootstrap' -ForegroundColor Yellow
Write-Dim "PythonVersion: $PythonVersion"
Write-Dim "WorkDir: $WorkDir"

Write-Step 1 'Preparing workspace and preflight checks'
Ensure-Directory $WorkDir
Ensure-Directory $zipDir
Ensure-Directory $srcRoot
Ensure-Directory $installerCacheDir
Write-Ok 'Workspace ready'
Write-Dim "Transcript log: $TranscriptLog"
Write-Dim "Build log: $BuildLog"
Write-Dim "Build Tools log: $BuildToolsLog"
Ensure-MinFreeSpaceGB -Path $WorkDir -RequiredGB 15

Write-Step 2 'Resolving Hermes installer'
if (-not $ctx.FromRepo) {
    Download-File -url "$($ctx.RawBase)/scripts/install-windows.ps1" -dest $installScriptLocal
    Write-Ok 'Downloaded Hermes installer script'
} else {
    Write-Dim 'Using local install-windows.ps1 from repo checkout'
}

$metadata = Read-BootstrapMetadata -Path $metadataPath
if (-not $ForceRebuild -and $metadata -and $metadata.python_version -eq $PythonVersion -and (Test-CustomPython -PythonExe $metadata.python_exe -ExpectedVersion $PythonVersion)) {
    $finalPythonExe = $metadata.python_exe
    $installMode = 'custom-reuse-metadata'
    Write-Step 3 'Reusing existing verified custom CPython'
    Write-Ok "Reusing custom Python: $finalPythonExe"
} elseif (-not $ForceRebuild -and (Test-CustomPython -PythonExe $pythonExe -ExpectedVersion $PythonVersion)) {
    $finalPythonExe = $pythonExe
    $installMode = 'custom-reuse-direct'
    Write-Step 3 'Reusing existing verified custom CPython'
    Write-Ok "Reusing custom Python: $finalPythonExe"
    Write-BootstrapMetadata -Path $metadataPath -PythonExe $finalPythonExe -PythonVersion $PythonVersion -BuildLog $BuildLog -TranscriptLog $TranscriptLog
} else {
    try {
        Write-Step 3 'Checking or bootstrapping Visual Studio Build Tools'
        $buildToolsPath = Ensure-BuildTools -InstallerCacheDir $installerCacheDir
        Write-Dim "Using toolchain at: $buildToolsPath"

        Write-Step 4 'Fetching CPython source zip'
        if ($ForceRebuild -and (Test-Path $extractRoot)) {
            Remove-Item $extractRoot -Recurse -Force
        }
        if ($ForceRebuild -and (Test-Path $zipPath)) {
            Remove-Item $zipPath -Force
        }
        if ($ForceRebuild -and (Test-Path $metadataPath)) {
            Remove-Item $metadataPath -Force
        }
        if (-not (Test-Path $zipPath)) {
            Download-File -url (Get-CPythonZipUrl $PythonVersion) -dest $zipPath
            Write-Ok "Downloaded CPython v$PythonVersion zip"
        } else {
            Write-Dim 'Reusing downloaded CPython zip'
        }

        Write-Step 5 'Extracting CPython'
        if (-not (Test-Path $extractRoot)) {
            $tempExtract = Join-Path $srcRoot "cpython-$PythonVersion-extract"
            if (Test-Path $tempExtract) {
                Remove-Item $tempExtract -Recurse -Force
            }
            Expand-Archive -Path $zipPath -DestinationPath $tempExtract -Force
            $inner = Join-Path $tempExtract "cpython-$PythonVersion"
            if (-not (Test-Path $inner)) {
                Fail "Expected extracted source dir missing: $inner"
            }
            Move-Item $inner $extractRoot
            Remove-Item $tempExtract -Recurse -Force
            Write-Ok 'Extraction complete'
        } else {
            Write-Dim 'Reusing extracted CPython source'
        }

        Write-Step 6 'Applying AF_UNIX patch'
        Apply-AfUnixPatch -SourceDir $extractRoot

        Write-Step 7 'Building CPython'
        $buildBat = Join-Path $extractRoot 'PCbuild\build.bat'
        if (-not (Test-Path $buildBat)) {
            Fail "build.bat not found: $buildBat"
        }
        Write-Dim "Streaming build output to: $BuildLog"
        Push-Location $extractRoot
        try {
            & cmd.exe /c "PCbuild\build.bat -c Release -p x64" *> $BuildLog
            if ($LASTEXITCODE -ne 0) {
                Fail "CPython build failed. See $BuildLog"
            }
        } finally {
            Pop-Location
        }
        Write-Ok 'CPython build finished'

        Write-Step 8 'Verifying AF_UNIX support'
        if (-not (Test-Path $pythonExe)) {
            Fail "Built python.exe not found: $pythonExe"
        }
        & $pythonExe -c "import socket,sys; print(sys.version); print('AF_UNIX:', hasattr(socket, 'AF_UNIX')); assert hasattr(socket, 'AF_UNIX'); s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.close(); print('socket test: ok')"
        if ($LASTEXITCODE -ne 0) {
            Fail 'AF_UNIX verification failed'
        }
        Write-Ok 'Verified custom CPython'

        $finalPythonExe = $pythonExe
        Write-BootstrapMetadata -Path $metadataPath -PythonExe $finalPythonExe -PythonVersion $PythonVersion -BuildLog $BuildLog -TranscriptLog $TranscriptLog
    } catch {
        $customFailure = $_.Exception.Message
        $installMode = 'stock-fallback'
        Write-Dim "Custom AF_UNIX build path failed: $customFailure"
        Write-Dim 'Falling back to stock Python install so Hermes still gets installed.'
    }
}

if ($finalPythonExe) {
    Write-Step 9 'Installing Hermes with custom Python'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installScriptLocal -PythonExe $finalPythonExe
    if ($LASTEXITCODE -ne 0) {
        Fail 'Hermes install failed'
    }
    Write-Ok 'Hermes installed'
} else {
    Write-Step 9 'Installing Hermes with stock Python fallback'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installScriptLocal
    if ($LASTEXITCODE -ne 0) {
        Fail 'Hermes stock-Python fallback install failed'
    }
    Write-Ok 'Hermes installed via stock Python fallback'
}

Write-Host ''
Write-Host 'Bootstrap complete' -ForegroundColor Green
Write-Dim "Install mode: $installMode"
if ($finalPythonExe) {
    Write-Dim "Custom Python: $finalPythonExe"
}
if ($customFailure) {
    Write-Dim "Custom build failure: $customFailure"
}
Write-Dim "Transcript log: $TranscriptLog"
Write-Dim "Build log: $BuildLog"
Write-Dim 'Open a new terminal and run: hermes'
}
finally {
    Stop-Transcript | Out-Null
}
