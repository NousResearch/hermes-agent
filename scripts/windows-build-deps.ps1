# Native build dependencies are separate from PM's application environment.
function Invoke-HermesBuildCommand {
    param([string]$Command, [string[]]$Arguments)
    $executable = (Get-Command $Command -CommandType Application -ErrorAction Stop).Source
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $executable @Arguments | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0) { throw "$Command failed with exit code $code" }
}

function Install-HermesArm64OpenSSL {
    param([string]$Vcpkg, [string]$Root)
    $prefix = Join-Path $Root 'installed\arm64-windows-static-md'
    $required = @('include\openssl\ssl.h', 'lib\libcrypto.lib', 'lib\libssl.lib')
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $prefix $_) -PathType Leaf) })
    if ($missing.Count) {
        Write-Host 'Installing static ARM64 OpenSSL development libraries via vcpkg...'
        Invoke-HermesBuildCommand $Vcpkg @('install', 'openssl:arm64-windows-static-md', '--classic', '--disable-metrics', "--x-install-root=$(Join-Path $Root 'installed')")
    } else {
        Write-Host "ARM64 OpenSSL development libraries found: $prefix"
    }
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $prefix $relative) -PathType Leaf)) {
            throw "OpenSSL installation is damaged: $prefix\$relative is missing. Repair the openssl:arm64-windows-static-md package in $Root, then rerun setup."
        }
    }
    return $prefix
}

function Get-HermesArm64VisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $found = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio discovery failed' }
    return ($found | Select-Object -First 1)
}

function Get-HermesClang {
    param([string]$VisualStudio)
    $command = Get-Command clang.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @((Join-Path $env:ProgramFiles 'LLVM\bin\clang.exe'))
    if ($VisualStudio) {
        $candidates += @(
            (Join-Path $VisualStudio 'VC\Tools\Llvm\ARM64\bin\clang.exe'),
            (Join-Path $VisualStudio 'VC\Tools\Llvm\bin\clang.exe')
        )
    }
    return ($candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1)
}

function Initialize-HermesArm64BuildTools {
    param([string]$StateRoot, [string]$OpenSSLRoot)
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'ARM64 build dependencies require Windows.'
    }
    $buildRoot = Join-Path $StateRoot 'build-tools'
    New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
    $vs = Get-HermesArm64VisualStudio
    $clangPath = Get-HermesClang -VisualStudio $vs
    if (-not $vs -or -not $clangPath) {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw 'ARM64 C++ or Clang build tools are missing. Run setup-hermes.ps1 once in an Administrator PowerShell to install them.'
        }
        $installer = Join-Path $buildRoot 'vs-buildtools.exe'
        Invoke-WebRequest -UseBasicParsing 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $installer
        $signature = Get-AuthenticodeSignature -LiteralPath $installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
            throw 'Visual Studio installer does not have a valid Microsoft signature'
        }
        $installArgs = @(
            '--quiet', '--wait', '--norestart', '--nocache',
            '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended',
            '--add', 'Microsoft.VisualStudio.Component.VC.Tools.ARM64',
            '--add', 'Microsoft.VisualStudio.Component.VC.Llvm.Clang'
        )
        if ($vs) { $installArgs = @('modify', '--installPath', ('"' + $vs + '"')) + $installArgs }
        $install = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
        if ($install.ExitCode -notin @(0, 3010)) { throw "Visual Studio installation failed: $($install.ExitCode)" }
        $vs = Get-HermesArm64VisualStudio
        if (-not $vs) { throw 'ARM64 C++ build tools remain unavailable. Restart Windows if the installer requested it.' }
        $clangPath = Get-HermesClang -VisualStudio $vs
        if (-not $clangPath) { throw 'The Clang compiler is still missing after Visual Studio setup.' }
    }
    Write-Host "ARM64 C++ build tools found: $vs"
    # CI, desktop builds and native staging can inherit the same developer
    # environment. VsDevCmd prepends its paths again on every call, eventually
    # overflowing cmd.exe's line limit. Reuse only a matching, usable environment.
    $vsReady = $env:VSINSTALLDIR -and $env:VSINSTALLDIR.TrimEnd('\') -eq $vs.TrimEnd('\') -and
        $env:VSCMD_ARG_HOST_ARCH -eq 'arm64' -and $env:VSCMD_ARG_TGT_ARCH -eq 'arm64' -and
        $env:INCLUDE -and $env:LIB -and (Get-Command cl.exe -ErrorAction SilentlyContinue)
    if (-not $vsReady) {
        $devCmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
        # Keep cmd's outer quotes intact on Windows PowerShell 5 as well as pwsh.
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $env:ComSpec
        $startInfo.Arguments = "/d /s /c `"`"$devCmd`" -no_logo -arch=arm64 -host_arch=arm64 >nul && set`""
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardOutput = $true
        $process = [System.Diagnostics.Process]::Start($startInfo)
        try {
            $lines = $process.StandardOutput.ReadToEnd() -split "`r?`n"
            $process.WaitForExit()
            if ($process.ExitCode -ne 0) { throw 'Could not initialize the ARM64 Visual Studio developer environment' }
        } finally { $process.Dispose() }
        foreach ($line in $lines) {
            if ($line -match '^([^=]+)=(.*)$') { Set-Item -LiteralPath "env:$($matches[1])" -Value $matches[2] }
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'ARM64 C++ compiler is unavailable after environment setup' }

    # Child builds isolate HOME/USERPROFILE. Keep Rust anchored to the homes
    # used here, including caller-selected locations.
    if (-not $env:CARGO_HOME) { $env:CARGO_HOME = Join-Path $HOME '.cargo' }
    if (-not $env:RUSTUP_HOME) { $env:RUSTUP_HOME = Join-Path $HOME '.rustup' }
    $env:CARGO_HOME = [IO.Path]::GetFullPath($env:CARGO_HOME)
    $env:RUSTUP_HOME = [IO.Path]::GetFullPath($env:RUSTUP_HOME)
    $cargoBin = Join-Path $env:CARGO_HOME 'bin'
    if ($cargoBin -notin ($env:PATH -split ';')) { $env:PATH = "$cargoBin;$env:PATH" }
    $rustup = Get-Command rustup.exe -ErrorAction SilentlyContinue
    if (-not $rustup) {
        $installer = Join-Path $buildRoot 'rustup-init.exe'
        $url = 'https://static.rust-lang.org/rustup/archive/1.28.2/aarch64-pc-windows-msvc/rustup-init.exe'
        Invoke-WebRequest -UseBasicParsing $url -OutFile $installer
        if ((Get-FileHash -Algorithm SHA256 $installer).Hash.ToLowerInvariant() -ne 'de9f7d29ccd39efa59a3dda3ec363b396e09b92681229b9b8f6aaa4c84285e9c') {
            throw 'rustup installer SHA256 mismatch'
        }
        Invoke-HermesBuildCommand $installer @('-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', '1.98.0-aarch64-pc-windows-msvc')
        $rustup = Get-Command rustup.exe -ErrorAction Stop
    }
    $rustc = Get-Command rustc.exe -ErrorAction SilentlyContinue
    $rustInfo = if ($rustc) { (& $rustc.Source -vV) -join "`n" } else { '' }
    if ($rustInfo -notmatch 'host: aarch64-pc-windows-msvc') {
        Invoke-HermesBuildCommand $rustup.Source @('toolchain', 'install', '1.98.0-aarch64-pc-windows-msvc', '--profile', 'minimal')
        $env:RUSTUP_TOOLCHAIN = '1.98.0-aarch64-pc-windows-msvc'
    }
    Write-Host 'ARM64 Rust toolchain ready'

    $env:CC_aarch64_pc_windows_msvc = $clangPath
    Write-Host "ARM64 Rust C compiler: $clangPath"

    $vcpkgRoot = $null
    $vcpkgCommand = Get-Command vcpkg.exe -ErrorAction SilentlyContinue
    $candidates = @($env:VCPKG_ROOT, $env:VCPKG_INSTALLATION_ROOT)
    if ($vcpkgCommand) { $candidates += Split-Path $vcpkgCommand.Source }
    $candidates += @((Join-Path $env:SystemDrive 'vcpkg'), (Join-Path $buildRoot 'vcpkg'))
    foreach ($candidate in $candidates) {
        # Visual Studio also ships a manifest-only vcpkg without a ports tree.
        if ($candidate -and (Test-Path -LiteralPath (Join-Path $candidate 'vcpkg.exe')) -and
            (Test-Path -LiteralPath (Join-Path $candidate 'ports\openssl\portfile.cmake'))) {
            $vcpkgRoot = $candidate
            break
        }
    }
    if (-not $vcpkgRoot) {
        $vcpkgRoot = Join-Path $buildRoot 'vcpkg'
        if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot '.git'))) {
            Invoke-HermesBuildCommand 'git' @('clone', 'https://github.com/microsoft/vcpkg.git', $vcpkgRoot)
            Invoke-HermesBuildCommand 'git' @('-C', $vcpkgRoot, 'checkout', '--detach', '00c5775211f45cd08b37fce0484b4cb940e422ab')
        }
        Invoke-HermesBuildCommand (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') @('-disableMetrics')
    }
    $env:VCPKG_ROOT = $vcpkgRoot
    # The install tree can be cached independently of the discovered checkout.
    if (-not $OpenSSLRoot) { $OpenSSLRoot = $vcpkgRoot }
    $env:OPENSSL_DIR = Install-HermesArm64OpenSSL -Vcpkg (Join-Path $vcpkgRoot 'vcpkg.exe') -Root $OpenSSLRoot
    $env:OPENSSL_STATIC = '1'
}
