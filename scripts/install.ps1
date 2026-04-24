<#
.SYNOPSIS
    Hermes Agent Installation Script for Windows 11.
.DESCRIPTION
    Installs the Hermes Agent natively on Windows 11.
    Sets up `uv`, a Python 3.12 virtual environment, dependencies,
    and the `ripgrep` executable.
#>

$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

Write-Output "============================================================"
Write-Output "          Hermes Agent Native Windows Installer            "
Write-Output "============================================================"

# 1. Check Python
$PythonExe = "python"
if (!(Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.11 or newer."
}

$PythonVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([decimal]$PythonVersion -lt 3.11) {
    Write-Error "Hermes Agent requires Python 3.11 or newer. Found $PythonVersion."
}

# 2. Install uv
if (!(Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Output "Installing 'uv' package manager..."
    Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
    powershell -ExecutionPolicy ByPass -File "install_uv.ps1"
    Remove-Item "install_uv.ps1"
    
    # Reload path to pick up uv
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    if (!(Get-Command "uv" -ErrorAction SilentlyContinue)) {
        # Fallback for Astral's default install location
        $env:Path += ";$env:USERPROFILE\.cargoin;$env:USERPROFILE\.localin"
    }
} else {
    Write-Output "'uv' is already installed."
}

# 3. Create venv
Write-Output "Creating virtual environment..."
$ProjectRoot = (Get-Item .).FullName
& uv venv venv --python $PythonVersion
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment."
}

# 4. Install dependencies
Write-Output "Installing dependencies..."
# For Windows, we install standard CLI features and exclude voice initially for compatibility unless specified
& uv pip install -e ".[all]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Python dependencies."
}

# 5. Install Ripgrep (rg.exe)
Write-Output "Installing ripgrep..."
$RgDir = "$ProjectRoot\venv\Scripts"
if (!(Test-Path "$RgDir\rg.exe")) {
    $RgUrl = "https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/ripgrep-14.1.0-x86_64-pc-windows-msvc.zip"
    $RgZip = "$ProjectRoot\rg.zip"
    Invoke-WebRequest -Uri $RgUrl -OutFile $RgZip
    Expand-Archive -Path $RgZip -DestinationPath "$ProjectRoot\rg_temp" -Force
    Move-Item -Path "$ProjectRoot\rg_temp\ripgrep-14.1.0-x86_64-pc-windows-msvc\rg.exe" -Destination $RgDir -Force
    Remove-Item -Path $RgZip -Force
    Remove-Item -Path "$ProjectRoot\rg_temp" -Recurse -Force
    Write-Output "ripgrep installed."
} else {
    Write-Output "ripgrep is already installed."
}


# 6. Generate hermes.cmd wrapper
Write-Output "Generating hermes.cmd wrapper in project root..."
$HermesCmd = "$ProjectRoot\hermes.cmd"
$WrapperContent = @'
@echo off
rem Hermes Agent native Windows launcher
rem Prioritizes the virtual environment Python if available

setlocal
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%~dp0hermes_cli\main.py" %*
) else (
    python "%~dp0hermes_cli\main.py" %*
)
endlocal
'@
Set-Content -Path $HermesCmd -Value $WrapperContent -Encoding ASCII
Write-Output "Wrapper generated at $HermesCmd."

Write-Output "============================================================"
Write-Output "Installation Complete!"
Write-Output "To start the agent, run:"
Write-Output "  .\venv\Scripts\activate"
Write-Output "  hermes"
Write-Output "============================================================"
