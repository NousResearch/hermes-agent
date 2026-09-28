# Native build dependencies are separate from PM's application environment.

# An interactive console sees one status line per build command, with the
# full output in $script:HermesBuildLog; CI and a redirected stdout (pm under
# the installer, which logs that itself) keep the full stream. install.ps1
# has its own copy: this file is dot-sourced by the build entry point and
# setup-hermes.ps1, which never load the installer.
function Test-HermesBuildQuiet {
    if ($env:CI -or $env:GITHUB_ACTIONS -or $env:HERMES_INSTALL_VERBOSE -or -not $script:HermesBuildLog) { return $false }
    try { return -not [Console]::IsOutputRedirected } catch { return $false }
}

function Write-HermesBuildNote {
    # Discovery details belong in CI transcripts, not on a user's console.
    param([string]$Message)
    if (-not (Test-HermesBuildQuiet)) { Write-Host "-> $Message" }
}

function Invoke-HermesBuildCommand {
    param([string]$Command, [string[]]$Arguments, [string]$Label)
    # Windows PowerShell 5.1 returns every match from Get-Command even without
    # -All. .Source on that array is every path joined by a space, and the call
    # operator then treats the joined string as one program name. Git for
    # Windows puts git.exe in both cmd\ and bin\, so a bare lookup is that bug.
    $executable = @(Get-Command $Command -CommandType Application -ErrorAction Stop | Select-Object -First 1)[0].Source
    if ($executable -isnot [string] -or -not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Could not resolve a single executable for $Command (got: $executable)"
    }
    if (-not $Label) { $Label = "Running $(Split-Path -Leaf $executable)" }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if (Test-HermesBuildQuiet) {
            $recent = New-Object 'System.Collections.Generic.Queue[string]'
            $width = 80
            try { $width = [Math]::Max(20, $Host.UI.RawUI.WindowSize.Width) } catch { $width = 80 }
            $writer = New-Object System.IO.StreamWriter($script:HermesBuildLog, $true, (New-Object System.Text.UTF8Encoding($false)))
            try {
                $writer.WriteLine("==> $Label ($((Get-Date).ToUniversalTime().ToString('s'))Z)")
                & $executable @Arguments 2>&1 | ForEach-Object {
                    $line = "$_".TrimEnd("`r")
                    $writer.WriteLine($line)
                    $recent.Enqueue($line)
                    if ($recent.Count -gt 20) { [void]$recent.Dequeue() }
                    # git and vcpkg redraw progress with bare CRs; show the newest.
                    $shown = "  ${Label}: $((($line -split "`r")[-1]).Trim())"
                    if ($shown.Length -ge $width) { $shown = $shown.Substring(0, $width - 1) }
                    Write-Host ("`r" + $shown.PadRight($width - 1)) -NoNewline -ForegroundColor DarkGray
                }
                $code = $LASTEXITCODE
            } finally {
                $writer.Dispose()
                Write-Host ("`r" + (' ' * ($width - 1)) + "`r") -NoNewline
            }
            if ($code -ne 0) {
                Write-Host "[X] $Label failed (exit $code). Last output:" -ForegroundColor Red
                foreach ($line in $recent) { Write-Host "    $line" }
                Write-Host "    full log: $script:HermesBuildLog"
            }
        } else {
            Write-Host "-> $Label"
            & $executable @Arguments | Out-Host
            $code = $LASTEXITCODE
        }
    } finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0) { throw "$Command failed with exit code $code" }
}

function Install-HermesArm64OpenSSL {
    param([string]$Vcpkg, [string]$Root)
    $prefix = Join-Path $Root 'installed\arm64-windows-static-md'
    $required = @('include\openssl\ssl.h', 'lib\libcrypto.lib', 'lib\libssl.lib')
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $prefix $_) -PathType Leaf) })
    if ($missing.Count) {
        Invoke-HermesBuildCommand $Vcpkg @('install', 'openssl:arm64-windows-static-md', '--classic', '--disable-metrics', "--x-install-root=$(Join-Path $Root 'installed')") 'Building static ARM64 OpenSSL via vcpkg (several minutes)'
    } else {
        Write-HermesBuildNote "ARM64 OpenSSL development libraries found: $prefix"
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
    $script:HermesBuildLog = Join-Path $buildRoot 'build.log'
    $vs = Get-HermesArm64VisualStudio
    $clangPath = Get-HermesClang -VisualStudio $vs
    if (-not $vs -or -not $clangPath) {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        $elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        # A UAC prompt needs someone at the desktop; CI, ssh and scheduled
        # runs would block on it, so they keep the explicit instruction.
        $canPrompt = [Environment]::UserInteractive -and -not $env:CI -and -not $env:GITHUB_ACTIONS -and
            -not $env:SSH_CONNECTION -and -not $env:SSH_CLIENT
        $needsAdmin = 'ARM64 C++ or Clang build tools are missing. Run setup-hermes.ps1 once in an Administrator PowerShell to install them.'
        if (-not $elevated -and -not $canPrompt) { throw $needsAdmin }
        $installer = Join-Path $buildRoot 'vs-buildtools.exe'
        Write-Host '-> Installing Visual Studio Build Tools (ARM64 C++ and Clang) to compile dependencies that have no ARM64 Windows wheel (such as cryptography).'
        Write-Host '-> This downloads several GB and can take 20+ minutes. The Visual Studio installer shows its progress in its own window.'
        Invoke-WebRequest -UseBasicParsing 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $installer
        $signature = Get-AuthenticodeSignature -LiteralPath $installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
            throw 'Visual Studio installer does not have a valid Microsoft signature'
        }
        # --passive shows the installer's progress window without asking
        # anything; --quiet showed nothing for the whole install.
        $installArgs = @(
            '--passive', '--wait', '--norestart', '--nocache',
            '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended',
            '--add', 'Microsoft.VisualStudio.Component.VC.Tools.ARM64',
            '--add', 'Microsoft.VisualStudio.Component.VC.Llvm.Clang'
        )
        if ($vs) { $installArgs = @('modify', '--installPath', ('"' + $vs + '"')) + $installArgs }
        if ($elevated) {
            $install = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
        } else {
            Write-Host '-> Windows will ask for administrator approval to install them.'
            try {
                $install = Start-Process -FilePath $installer -ArgumentList $installArgs -Verb RunAs -Wait -PassThru
            } catch {
                throw "Administrator approval was declined or unavailable. $needsAdmin"
            }
        }
        if ($install.ExitCode -notin @(0, 3010)) { throw "Visual Studio installation failed: $($install.ExitCode)" }
        $vs = Get-HermesArm64VisualStudio
        if (-not $vs) { throw 'ARM64 C++ build tools remain unavailable. Restart Windows if the installer requested it.' }
        $clangPath = Get-HermesClang -VisualStudio $vs
        if (-not $clangPath) { throw 'The Clang compiler is still missing after Visual Studio setup.' }
    }
    Write-HermesBuildNote "ARM64 C++ build tools found: $vs"
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
    # With VSCMD_ARG_TGT_ARCH exported, rustc's cc crate takes link.exe from PATH
    # instead of asking vswhere. Under Git Bash that is coreutils' link.exe, so
    # pin the MSVC linker explicitly. PATH order is the caller's (a reused
    # environment keeps Git Bash's order), so take it from the developer
    # environment's own tools directory instead.
    $linker = if ($env:VCToolsInstallDir) { Join-Path $env:VCToolsInstallDir 'bin\HostARM64\ARM64\link.exe' }
    if (-not $linker -or -not (Test-Path -LiteralPath $linker)) {
        throw "MSVC link.exe is missing from VCToolsInstallDir '$env:VCToolsInstallDir'"
    }
    $env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = $linker

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
        Invoke-HermesBuildCommand $installer @('-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', '1.98.0-aarch64-pc-windows-msvc') 'Installing the Rust toolchain (1.98.0, ARM64) for those source builds'
        $rustup = Get-Command rustup.exe -ErrorAction Stop
    }
    $rustc = Get-Command rustc.exe -ErrorAction SilentlyContinue
    $rustInfo = if ($rustc) { (& $rustc.Source -vV) -join "`n" } else { '' }
    if ($rustInfo -notmatch 'host: aarch64-pc-windows-msvc') {
        Invoke-HermesBuildCommand $rustup.Source @('toolchain', 'install', '1.98.0-aarch64-pc-windows-msvc', '--profile', 'minimal') 'Installing Rust 1.98.0 for ARM64'
        $env:RUSTUP_TOOLCHAIN = '1.98.0-aarch64-pc-windows-msvc'
    }
    Write-HermesBuildNote 'ARM64 Rust toolchain ready'

    $env:CC_aarch64_pc_windows_msvc = $clangPath
    Write-HermesBuildNote "ARM64 Rust C compiler: $clangPath"

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
            # Phase lines ("Receiving objects: 42%") feed the status line;
            # git prints none to a pipe unless asked.
            $progress = @()
            if (Test-HermesBuildQuiet) { $progress = @('--progress') }
            Invoke-HermesBuildCommand 'git' (@('clone') + $progress + @('https://github.com/microsoft/vcpkg.git', $vcpkgRoot)) 'Downloading vcpkg to build OpenSSL for ARM64'
            Invoke-HermesBuildCommand 'git' @('-C', $vcpkgRoot, 'checkout', '--detach', '00c5775211f45cd08b37fce0484b4cb940e422ab') 'Pinning vcpkg'
        }
        Invoke-HermesBuildCommand (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') @('-disableMetrics') 'Building vcpkg'
    }
    $env:VCPKG_ROOT = $vcpkgRoot
    # The install tree can be cached independently of the discovered checkout.
    if (-not $OpenSSLRoot) { $OpenSSLRoot = $vcpkgRoot }
    $env:OPENSSL_DIR = Install-HermesArm64OpenSSL -Vcpkg (Join-Path $vcpkgRoot 'vcpkg.exe') -Root $OpenSSLRoot
    $env:OPENSSL_STATIC = '1'
}
