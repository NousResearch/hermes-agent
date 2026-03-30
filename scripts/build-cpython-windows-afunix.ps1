param(
    [string]$PythonVersion = "3.13.12",
    [string]$SourceDir = "$env:USERPROFILE\cpython-$($PythonVersion)-afunix",
    [string]$PatchFile = "$PSScriptRoot\patches\cpython-3.13-windows-afunix.patch"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$PatchFile = [System.IO.Path]::GetFullPath($PatchFile)

if ($PythonVersion -notmatch '^3\.13(\.|$)') {
    throw "This helper is currently pinned to the 3.13 AF_UNIX patch series. Requested: $PythonVersion"
}

function Write-Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Dim($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Test-Command($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host ""
Write-Host "CPython Windows AF_UNIX builder" -ForegroundColor Yellow
Write-Host "SourceDir: $SourceDir" -ForegroundColor DarkGray
Write-Host "PatchFile: $PatchFile" -ForegroundColor DarkGray

Write-Step 1 "Checking prerequisites"
if (-not (Test-Command git)) {
    throw "git not found in PATH"
}
if (-not (Test-Path $PatchFile)) {
    throw "Patch file not found: $PatchFile"
}
if (-not (Test-Command cmd.exe)) {
    throw "cmd.exe not found in PATH"
}
$buildBat = Join-Path $SourceDir "PCbuild\build.bat"

Write-Step 2 "Cloning CPython"
if (-not (Test-Path $SourceDir)) {
    git clone --branch "v$PythonVersion" --depth 1 https://github.com/python/cpython.git $SourceDir
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed"
    }
    Write-Ok "Cloned CPython v$PythonVersion"
} else {
    Write-Dim "Source dir already exists, reusing it"
}

Write-Step 3 "Applying AF_UNIX patch"
Push-Location $SourceDir
try {
    & git apply --check $PatchFile
    if ($LASTEXITCODE -eq 0) {
        & git apply $PatchFile
        if ($LASTEXITCODE -ne 0) {
            throw "git apply failed"
        }
        Write-Ok "Patch applied"
    } else {
        & git apply --reverse --check $PatchFile
        if ($LASTEXITCODE -eq 0) {
            Write-Dim "Patch already applied; continuing"
        } else {
            Write-Dim "Patch context did not match cleanly; current diff follows"
            git diff -- PC/pyconfig.h.in Modules/socketmodule.h Modules/socketmodule.c | Out-Host
            throw "Patch could not be applied cleanly"
        }
    }
} finally {
    Pop-Location
}

Write-Step 4 "Building CPython"
if (-not (Test-Path $buildBat)) {
    throw "build.bat not found: $buildBat"
}
cmd.exe /c "cd /d $SourceDir && PCbuild\build.bat -c Release -p x64"
if ($LASTEXITCODE -ne 0) {
    throw "CPython build failed"
}
Write-Ok "Build finished"

Write-Step 5 "Verifying AF_UNIX"
$pythonExe = Join-Path $SourceDir "PCbuild\amd64\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Built python.exe not found: $pythonExe"
}
& $pythonExe -c "import socket,sys; print(sys.version); print('AF_UNIX:', hasattr(socket, 'AF_UNIX')); assert hasattr(socket, 'AF_UNIX'); s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.close(); print('socket test: ok')"
if ($LASTEXITCODE -ne 0) {
    throw "AF_UNIX verification failed"
}
Write-Ok "Custom Python ready: $pythonExe"

Write-Host ""
Write-Host "Install Hermes with it using:" -ForegroundColor Yellow
Write-Host "  .\scripts\install-windows.ps1 -PythonExe $pythonExe" -ForegroundColor DarkGray
