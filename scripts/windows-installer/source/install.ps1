# ============================================================================
# Hermes Agent Installer for Windows
# ============================================================================
# Installation script for Windows (PowerShell).
# Uses uv for fast Python provisioning and package management.
#
# Usage:
#   iex (irm https://hermes-agent.nousresearch.com/install.ps1)
#
# Or download and run with options:
#   .\install.ps1 -NoVenv -SkipSetup
#
# ============================================================================

param(
    [switch]$NoVenv,
    [switch]$SkipSetup,
    [switch]$SkipComputerUse,
    [string]$Branch = "main",
    # -Commit and -Tag are higher-precedence variants of -Branch for users
    # who need reproducible installs (desktop installer pinning, CI, release
    # bundles).  When set, the repository stage clones $Branch (faster than
    # cloning the full default-branch history) and then `git checkout`s the
    # exact ref.  Precedence: Commit > Tag > Branch.
    [string]$Commit = "",
    # Apply -Commit even when it would roll an existing install BACKWARDS.
    # Without this the repository stage skips a pin that is already an ancestor
    # of HEAD, so a stale baked-in BUILD_PIN_COMMIT can't downgrade a current
    # checkout. Reproducible/CI installs that genuinely want an older SHA on an
    # existing tree pass -ForceCommit.
    [switch]$ForceCommit,
    [string]$Tag = "",
    [string]$HermesHome = $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }),
    [string]$InstallDir = $(if ($env:HERMES_HOME) { "$env:HERMES_HOME\hermes-agent" } else { "$env:LOCALAPPDATA\hermes\hermes-agent" }),

    # --- Stage protocol (additive; default invocation behaves as before) ----
    # See the "Stage protocol" section near the bottom of the file for the
    # full contract.  Intended for programmatic drivers (the desktop GUI's
    # onboarding wizard, CI, future install.sh parity, etc.).  CLI users
    # running the canonical `irm | iex` one-liner never touch these flags.
    [switch]$Manifest,
    [string]$Stage,
    [switch]$ProtocolVersion,
    [switch]$NonInteractive,
    [switch]$Json,

    # Print the paths this install would use, as JSON, and exit without
    # touching anything. The first question on any "installer says a path
    # doesn't exist" report is which paths it actually resolved -- especially
    # on profiles Windows exposes through an 8.3 alias, where what the user
    # sees in Explorer and what the installer receives differ.
    #
    #   powershell -File install.ps1 -ShowResolvedPaths
    [switch]$ShowResolvedPaths,

    # --- Ensure mode (dep_ensure.py entry point) ---
    [string]$Ensure = "",
    [switch]$PostInstall,

    # --- Desktop GUI build (opt-in) ---
    # When set, install.ps1 includes Stage-Desktop in the manifest and
    # builds apps/desktop into a launchable Hermes.exe.
    #
    # Why opt-in:
    #   * Hermes-Setup.exe (the signed Tauri bootstrap installer) passes
    #     -IncludeDesktop so a user who installed via the GUI ends up
    #     with a launchable desktop binary.
    #   * The Electron desktop's own bootstrap-runner.ts runs install.ps1
    #     from inside an already-launched Hermes.exe; if THAT recursively
    #     built apps/desktop it would try to overwrite the live Hermes.exe
    #     on disk and fail. The recursive path omits the flag.
    #   * The canonical CLI one-liner (irm | iex) omits the flag too;
    #     terminal users don't need a desktop binary built for them, and
    #     `hermes desktop` already builds on demand.
    [switch]$IncludeDesktop
)

$ErrorActionPreference = "Stop"

# Suppress Invoke-WebRequest's per-chunk progress bar.  Windows PowerShell
# 5.1's progress UI repaints synchronously on every received byte, which
# pegs CPU on a single core and throttles downloads by 10-100x (a 57MB
# PortableGit grab can take 5 minutes with progress on vs 20 seconds
# with progress off, on the same network).  Every IWR call in this
# script is fire-and-forget so we never need to see the bar.  Restored
# automatically when the script exits.
$ProgressPreference = "SilentlyContinue"

# Force the console to UTF-8 so non-ASCII output from native commands
# (e.g. playwright's box-drawing progress bars and download banners,
# git's bullet glyphs, npm's check marks) renders correctly instead of
# as IBM437/Windows-1252 mojibake (sequences like 0xE2 0x95 0x94 box-
# drawing chars decoded under the legacy DOS codepage).  This is a
# DISPLAY-only fix; the underlying bytes are already correct.  We do
# NOT change the file's own encoding (it remains pure ASCII for PS 5.1
# parser compatibility; see comments at the top of the entry-point
# dispatch).  This affects only what the user sees in their terminal
# during this install run, and reverts automatically when the script
# exits and the host's console encoding is restored.
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
} catch {
    # Some constrained PowerShell hosts disallow encoding mutation.
    # Mojibake on output is then cosmetic-only, install still works.
}

# ============================================================================
# 8.3 short-path normalization
# ============================================================================
# Windows generates an 8.3 short alias for a user-profile folder whose name
# contains a space ("First Last" -> FIRST~1.LAS), a dot ("Stone.ZEN8" ->
# STONE~1.ZEN), or an accented character ("Ruben" spelled with an acute e ->
# RUBN~1). It can then expose %TEMP%, %TMP%, %LOCALAPPDATA%, %APPDATA% and
# %USERPROFILE% -- plus everything derived from them, including the default
# HERMES_HOME and InstallDir -- in that short form:
#   C:\Users\FIRST~1.LAS\AppData\Local\Temp
#
# PowerShell's FileSystem provider mishandles the aliased component when such a
# path reaches a provider cmdlet (`Tee-Object -FilePath`, `Out-File`,
# `New-Item`, `Test-Path`), throwing "An object at the specified path
# C:\Users\FIRST~1.LAS does not exist" -- localized on non-English hosts.
# Every Node/Electron stage streams its build log to %TEMP% via Tee-Object and
# the desktop stage probes the binary it produced under the profile-derived
# InstallDir, so the bootstrap aborts even though the artifact built fine.
# The Python/uv stages, which never hand a %TEMP% path to a provider cmdlet,
# sail through -- which is why the failure looks Node-specific.
#
# Expanding every profile-rooted path back to long form once, up front, lets
# every downstream cmdlet and child process see something the provider can
# resolve. Three resolvers, tried in order, because no single one covers every
# host:
#
#   1. kernel32!GetLongPathNameW -- expands any 8.3 component regardless of
#      locale, including the accented-username aliases the COM resolver misses.
#   2. Scripting.FileSystemObject -- fallback for hosts where P/Invoke is
#      blocked.
#   3. Profile-root substitution -- when the volume has 8.3 generation disabled
#      or the alias is stale, neither resolver can expand the name because it
#      no longer maps to anything on disk. The aliased component is always the
#      profile folder itself (everything below it was created long), so swap in
#      a profile root we can prove is long and reattach the tail.
#
# All three degrade to returning the input untouched, so a host where none of
# them apply -- including non-Windows -- behaves exactly as it did before.

$script:LongProfileRoot = $null

function Write-PathDiag {
    # Diagnostics for this block go to stderr, never stdout: the stage protocol
    # hands drivers a single line of JSON on stdout and a stray note would break
    # anything parsing it.
    #
    # Suppressed entirely under -ShowResolvedPaths, which is a machine-readable
    # query: Windows PowerShell 5.1 wraps any native-command stderr in a
    # NativeCommandError and folds it back into the caller's own stream, so a
    # child writing here at all is enough to corrupt a 5.1 caller's capture.
    # The JSON already carries everything these lines say.
    #
    # [Console]::Error.WriteLine specifically -- verified reaching a caller on a
    # windows-latest runner. $host.UI.WriteErrorLine was tried and silently
    # produced nothing there under a non-interactive host.
    param([string]$Message)
    if ($ShowResolvedPaths) { return }
    [Console]::Error.WriteLine("[hermes] $Message")
}

function Get-LongProfileRoot {
    # The user's profile directory in long form, or '' when every source we
    # can reach is itself aliased. Cached: this runs per env var.
    if ($null -ne $script:LongProfileRoot) { return $script:LongProfileRoot }
    $script:LongProfileRoot = ''

    # %USERPROFILE% first: it is what the rest of the install derives from, and
    # on a host handing us aliased paths the .NET known-folder lookup tends to
    # be aliased in exactly the same way. Then the HOMEDRIVE/HOMEPATH pair, then
    # the profile's parent (C:\Users never carries an alias) plus %USERNAME%,
    # which stays the long account name even when every path is short.
    $envProfile = [Environment]::GetEnvironmentVariable('USERPROFILE')
    $shellProfile = [Environment]::GetFolderPath('UserProfile')
    $candidates = @($envProfile, $shellProfile, "$env:HOMEDRIVE$env:HOMEPATH")
    foreach ($anchor in @($envProfile, $shellProfile)) {
        if ($anchor -and $env:USERNAME) {
            $parent = Split-Path -Parent $anchor.TrimEnd('\', '/')
            if ($parent) { $candidates += (Join-Path $parent $env:USERNAME) }
        }
    }

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        # Trailing separators make Split-Path -Parent return the directory
        # itself, which would silently break the ancestry check downstream.
        $candidate = $candidate.TrimEnd('\', '/')
        if (-not $candidate) { continue }
        if ($candidate -match '~\d') { continue }
        try {
            if (Test-Path -LiteralPath $candidate -PathType Container) {
                $script:LongProfileRoot = $candidate
                break
            }
        } catch {
            # Unreadable candidate (denied, malformed): try the next one.
        }
    }

    # Say which root we landed on. When someone reports "still broken" this is
    # the first thing worth knowing, and it costs one line on the rare path
    # where an alias actually showed up.
    if ($script:LongProfileRoot) {
        Write-PathDiag "long profile root: $script:LongProfileRoot"
    } else {
        Write-PathDiag "no long profile root found; 8.3 paths left as-is (tried: $($candidates -join ', '))"
    }
    return $script:LongProfileRoot
}

function Expand-ShortProfileRoot {
    # Rebuild $Path onto a known-long profile root when its aliased component
    # is the profile folder. Returns $Path unchanged when it isn't, so a custom
    # TEMP on another volume (D:\SHORT~1\Temp) is never rewritten.
    param([string]$Path)

    $longRoot = Get-LongProfileRoot
    if (-not $longRoot) { return $Path }
    $longRootParent = Split-Path -Parent $longRoot
    if (-not $longRootParent) { return $Path }

    $node = $Path
    $tail = ''
    while ($node -and ($node -match '~\d')) {
        $leaf = Split-Path -Leaf $node
        $parent = Split-Path -Parent $node
        if (-not $parent) { return $Path }
        if ($leaf -match '~\d') {
            # Candidate profile folder. Only substitute when it sits in the
            # same directory as the real profile (both C:\Users).
            if ($parent -ne $longRootParent) { return $Path }
            if ($tail) { return (Join-Path $longRoot $tail) }
            return $longRoot
        }
        $tail = if ($tail) { Join-Path $leaf $tail } else { $leaf }
        $node = $parent
    }
    return $Path
}

function ConvertTo-LongPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
    # Only 8.3 short names carry a tilde+digit ("~1"); skip every resolver for
    # ordinary long paths, which is the overwhelmingly common case.
    if ($Path -notmatch '~\d') {
        $script:LastResolver = 'skipped-long-path'
        return $Path
    }

    # 1. kernel32. Compiled on first use only, so a normal profile never pays
    #    the Add-Type cost (this file is re-entered once per install stage).
    try {
        if (-not ([System.Management.Automation.PSTypeName]'HermesInstall.LongPath').Type) {
            Add-Type -Namespace 'HermesInstall' -Name 'LongPath' -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern int GetLongPathNameW(string lpszShortPath, System.Text.StringBuilder lpszLongPath, int cchBuffer);
'@
        }
        $buffer = New-Object System.Text.StringBuilder 4096
        $length = [HermesInstall.LongPath]::GetLongPathNameW($Path, $buffer, $buffer.Capacity)
        if ($length -gt $buffer.Capacity) {
            $buffer = New-Object System.Text.StringBuilder $length
            $length = [HermesInstall.LongPath]::GetLongPathNameW($Path, $buffer, $buffer.Capacity)
        }
        if ($length -gt 0) {
            $expanded = $buffer.ToString()
            if ($expanded -and $expanded -notmatch '~\d') {
                $script:LastResolver = 'kernel32'
                return $expanded
            }
        }
    } catch {
        # Not Windows, or P/Invoke denied by policy: try the next resolver.
    }

    # 2. COM. Validate the result the same way the kernel32 branch does: this
    # resolver can report success and still hand back a path that carries the
    # alias (observed on a windows-latest runner, where it "resolved"
    # C:\Users\FIRST~1.LAS\... to itself). Accepting that silently is what let a
    # short path reach the provider cmdlets in the first place, so an
    # unexpanded result counts as failure and falls through.
    try {
        $fso = New-Object -ComObject Scripting.FileSystemObject
        $resolved = $null
        if ($fso.FolderExists($Path))   { $resolved = $fso.GetFolder($Path).Path }
        elseif ($fso.FileExists($Path)) { $resolved = $fso.GetFile($Path).Path }
        if ($resolved -and $resolved -notmatch '~\d') {
            $script:LastResolver = 'com'
            return $resolved
        }
    } catch {
        # COM unavailable / locked-down host: try the next resolver.
    }

    # 3. The alias resolves to nothing. Rebuild from a long profile root.
    $rebuilt = Expand-ShortProfileRoot $Path
    $script:LastResolver = if ($rebuilt -ne $Path) { 'profile-root' } else { 'none' }
    return $rebuilt
}

function Set-LongProfileEnvVars {
    # Normalize every profile-rooted variable the install reads, not just
    # %TEMP%: the desktop stage derives InstallDir from %LOCALAPPDATA%, and a
    # short root there fails the post-build probe after a successful build.
    # Returns $true when anything was rewritten.
    $rewrote = $false
    $script:NormalizedPathRewrites = @{}
    foreach ($name in @('TEMP', 'TMP', 'LOCALAPPDATA', 'APPDATA', 'USERPROFILE')) {
        $current = [Environment]::GetEnvironmentVariable($name)
        if (-not $current) { continue }
        $expanded = ConvertTo-LongPath $current
        if ($expanded -and $expanded -ne $current) {
            Set-Item -Path "Env:$name" -Value $expanded
            $rewrote = $true
            $script:NormalizedPathRewrites[$name] = $expanded
            # Rewriting a profile path is rare and corrective; say so. Every
            # report of this bug class arrived as a bare "does not exist" with
            # no hint that a short alias was involved. stderr, so the stage
            # protocol's stdout JSON stays parseable.
            Write-PathDiag "expanded 8.3 short path in %$name%: $current -> $expanded"
        }
    }
    return $rewrote
}

# ConvertTo-LongPath only assigns $script:LastResolver when a ~\d short path
# actually needs expansion, so an ordinary long profile leaves it unset -- and
# the ResolvedPathReport below reads it unconditionally, which is fatal under
# Set-StrictMode before any stage starts. 'none' is the resolver's own value
# for "nothing ran".
$script:LastResolver = 'none'
$script:NormalizedProfilePaths = Set-LongProfileEnvVars

# Re-derive the install paths now that the env vars behind their defaults are
# long. An explicitly passed -HermesHome / -InstallDir is normalized in place
# rather than replaced, so a caller's choice is never overwritten by a default.
# $PSBoundParameters is only meaningful at script scope, so this stays inline.
if ($PSBoundParameters.ContainsKey('HermesHome')) {
    $HermesHome = ConvertTo-LongPath $HermesHome
} else {
    $HermesHome = ConvertTo-LongPath $(
        if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }
    )
}
if ($PSBoundParameters.ContainsKey('InstallDir')) {
    $InstallDir = ConvertTo-LongPath $InstallDir
} else {
    $InstallDir = ConvertTo-LongPath $(
        if ($env:HERMES_HOME) { "$env:HERMES_HOME\hermes-agent" } else { "$env:LOCALAPPDATA\hermes\hermes-agent" }
    )
}
if ($script:NormalizedProfilePaths) {
    # Which paths the install actually settled on. Absent from every report of
    # this bug class, and the whole question once a short alias is in play.
    Write-PathDiag "resolved install paths: HermesHome=$HermesHome InstallDir=$InstallDir"
}

# Captured here, where the values are final, and emitted from the entry-point
# dispatch at the bottom (alongside -ProtocolVersion / -Manifest) so
# -ShowResolvedPaths exits before any stage runs.
#
# The report goes to STDOUT as JSON: on Windows a child's stderr does not
# reliably reach a parent process -- three separate capture mechanisms each came
# back empty on a windows-latest runner while stdout arrived intact -- and the
# first question on any "installer says a path doesn't exist" report is which
# paths it actually resolved.
$script:ResolvedPathReport = @{
    long_profile_root = (Get-LongProfileRoot)
    normalized        = $script:NormalizedPathRewrites
    resolver          = $script:LastResolver
    temp              = $env:TEMP
    hermes_home       = $HermesHome
    install_dir       = $InstallDir
}

# ============================================================================
# Configuration
# ============================================================================

$RepoUrlSsh = "git@github.com:NousResearch/hermes-agent.git"
$RepoUrlHttps = "https://github.com/NousResearch/hermes-agent.git"
$PythonVersion = "3.11"
# Minor versions the installer accepts when the requested $PythonVersion isn't
# available, in preference order. Only checkout-private uv-managed interpreters
# are eligible. Single source of truth shared by Test-Python's fallback and
# Resolve-AvailablePythonVersion.
$PythonFallbackVersions = @("3.12", "3.13", "3.10")
$PythonFindTimeoutMs = 30000
$NodeVersion = "22"
# The npm range the root package.json pins in `engines.npm`.  A constant rather
# than a manifest read like the POSIX side does: Test-Node runs BEFORE the repo
# is cloned, so there is usually no package.json on disk yet (and none at all
# when install.ps1 is piped straight from the web). Keep this fallback in sync
# with package.json; Get-NpmRange prefers the manifest once a checkout exists.
$NpmRange = "<11.10.0 || >=11.17.0"

# Stage-protocol version.  Bumped only for genuinely breaking changes to the
# manifest schema, stage-name set semantics, or stdout JSON shape.  Adding a
# new stage does NOT bump this -- drivers iterate the manifest dynamically.
$InstallStageProtocolVersion = 1

# ============================================================================
# Helper functions

# Return the real OS processor architecture as a lowercase string suitable for
# Node.js / electron download URL slugs: "arm64", "x64", or "x86".
#
# Why not just trust [Environment]::Is64BitOperatingSystem or
# [RuntimeInformation]::OSArchitecture?  On Windows on ARM, when this script
# is invoked from Windows PowerShell 5.1 (the default `powershell.exe`) or
# any x64 PowerShell host, the process runs under Prism x64 emulation and
# BOTH of those APIs report `X64` -- they describe the emulated view, not
# the real OS.  We've seen this concretely on Snapdragon X1 hardware: an
# ARM64-based Surface Laptop returns OSArchitecture=X64 from an emulated
# PowerShell session.
#
# Win32_Processor.Architecture is invariant to emulation.  Values:
#   0=x86, 5=ARM, 9=AMD64/x64, 12=ARM64.  We fall back to
#   PROCESSOR_ARCHITEW6432 (set on WoW64 with the real OS arch) and then
#   PROCESSOR_ARCHITECTURE so we still produce a sensible answer if CIM
#   isn't available (locked-down WMI, container, etc.).
function Get-WindowsArch {
    try {
        $proc = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop |
            Select-Object -First 1
        switch ([int]$proc.Architecture) {
            12 { return "arm64" }
            9  { return "x64" }
            0  { return "x86" }
            5  { return "arm" }
        }
    } catch {
        # CIM unavailable -- fall through to env-var path
    }

    $envArch = if ($env:PROCESSOR_ARCHITEW6432) {
        $env:PROCESSOR_ARCHITEW6432
    } else {
        $env:PROCESSOR_ARCHITECTURE
    }
    switch ($envArch) {
        "ARM64" { return "arm64" }
        "AMD64" { return "x64" }
        "x86"   { return "x86" }
        default {
            # Last-resort: respect 64-bitness so we don't ship a 32-bit
            # toolchain to anyone.
            if ([Environment]::Is64BitOperatingSystem) { return "x64" } else { return "x86" }
        }
    }
}

# ============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|             * Hermes Agent Installer                    |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|  An open source AI agent by Nous Research.              |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host ""
}

function Write-Info {
    param([string]$Message)
    Write-Host "-> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Invoke-NativeWithRelaxedErrorAction {
    param([scriptblock]$Script)

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Script
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}
function Discard-LockfileChurn {
    param([string]$Repo = $InstallDir)

    if (-not $Repo -or -not (Test-Path (Join-Path $Repo ".git"))) { return }

    try {
        $diff = & git -c windows.appendAtomically=false -C $Repo diff --name-only 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $diff) { return }

        $dirtyPackageDirs = [System.Collections.Generic.HashSet[string]]::new(
            [System.StringComparer]::OrdinalIgnoreCase
        )
        foreach ($path in $diff) {
            if ($path -like "*package.json") {
                $null = $dirtyPackageDirs.Add((Split-Path $path -Parent))
            }
        }

        $dirtyLocks = [System.Collections.Generic.List[string]]::new()
        foreach ($path in $diff) {
            if ($path -notlike "*package-lock.json") { continue }
            $lockDir = Split-Path $path -Parent
            if ($dirtyPackageDirs.Contains($lockDir)) { continue }
            $dirtyLocks.Add($path)
        }

        if ($dirtyLocks.Count -eq 0) { return }
        & git -c windows.appendAtomically=false -C $Repo checkout -- @($dirtyLocks) 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Discarded npm lockfile churn ($($dirtyLocks.Count) file(s))"
        }
    } catch {
        # Best-effort only; never let cleanup block the installer update path.
    }
}
# Inspect npm output for a TLS-trust failure and, if found, print actionable
# remediation. npm/Node surface corporate MITM proxies and missing root CAs as
# "unable to get local issuer certificate" / "self-signed certificate in
# certificate chain" / UNABLE_TO_GET_ISSUER_CERT_LOCALLY -- most commonly while
# Electron's install.js postinstall downloads the Electron binary. The reporter
# usually misreads this as an admin-rights or generic install failure (see
# issue #38016), so detect it once here and route every npm stage through this
# hint. Returns $true when a cert error was detected (caller may adjust its own
# messaging), $false otherwise.
function Show-NpmCertHint {
    param([string]$NpmOutput)
    if (-not $NpmOutput) { return $false }
    $isCertError = $NpmOutput -match "unable to get local issuer certificate" `
        -or $NpmOutput -match "self.signed certificate" `
        -or $NpmOutput -match "UNABLE_TO_GET_ISSUER_CERT_LOCALLY" `
        -or $NpmOutput -match "SELF_SIGNED_CERT_IN_CHAIN" `
        -or $NpmOutput -match "CERT_HAS_EXPIRED"
    if (-not $isCertError) { return $false }
    Write-Warn "This looks like a TLS certificate-trust failure, not a permissions problem."
    Write-Info "  A corporate proxy or antivirus is likely intercepting HTTPS and presenting a"
    Write-Info "  certificate Node.js doesn't trust. To fix, point Node at your org's root CA:"
    Write-Info "    1. Get the corporate root CA as a .pem/.crt from your IT team."
    Write-Info "    2. setx NODE_EXTRA_CA_CERTS `"C:\path\to\corp-ca.pem`""
    Write-Info "    3. Open a NEW terminal (so the env var takes effect) and re-run the installer."
    Write-Info "  Quick (less secure) alternative -- disable TLS verification just for the install:"
    Write-Info "    npm config set strict-ssl false   (re-enable afterwards: npm config set strict-ssl true)"
    return $true
}

function Write-NpmDebugLogTail {
    # On failure npm prints only a terse summary to stdout/stderr; the real
    # evidence (postinstall script stderr like Electron's install.js, network
    # traces, EBUSY retries) lives in npm's own debug log under
    # <npm-cache>\_logs\<timestamp>-debug-0.log. The bootstrap installer's
    # streaming sink only captures what WE emit, so on any npm failure this
    # helper locates that debug log and replays its tail into our output
    # stream -- making the bootstrap log a self-contained diagnosis instead
    # of "exit 1, details in a file on a VM nobody can reach".
    param(
        [string]$NpmOutput,
        [int]$TailLines = 200
    )
    $logPath = $null
    # Preferred: npm names the exact file in its failure summary.
    if ($NpmOutput -and $NpmOutput -match "A complete log of this run can be found in:\s*(?<path>[^\r\n]+)") {
        $candidate = $Matches['path'].Trim()
        if (Test-Path -LiteralPath $candidate) { $logPath = $candidate }
    }
    # Fallback (covers --silent runs, truncated output): newest debug log in
    # npm's cache _logs directory.
    if (-not $logPath) {
        try {
            $npm = Resolve-NpmCmd
            if ($npm) {
                $prevEAPLocal = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $cacheDir = (& $npm config get cache 2>$null | Select-Object -Last 1)
                $ErrorActionPreference = $prevEAPLocal
                if ($cacheDir) {
                    $logsDir = Join-Path ("$cacheDir").Trim() "_logs"
                    if (Test-Path -LiteralPath $logsDir) {
                        $newest = Get-ChildItem -LiteralPath $logsDir -Filter "*-debug-*.log" -ErrorAction SilentlyContinue |
                            Sort-Object LastWriteTime -Descending | Select-Object -First 1
                        if ($newest) { $logPath = $newest.FullName }
                    }
                }
            }
        } catch { }
    }
    if (-not $logPath) {
        Write-Warn "npm debug log could not be located -- no further npm detail available"
        return
    }
    $tail = $null
    try {
        $tail = Get-Content -LiteralPath $logPath -Tail $TailLines -ErrorAction Stop
    } catch {
        Write-Warn "Could not read npm debug log ${logPath}: $($_.Exception.Message)"
        return
    }
    Write-Warn "---- npm debug log: last $TailLines lines of $logPath ----"
    foreach ($line in $tail) { Write-Host "    $line" -ForegroundColor DarkGray }
    Write-Warn "---- end npm debug log ----"
}

# --- Ensure-mode helpers ---

function Resolve-NpmCmd {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) { return $null }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $npmCmdSibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $npmCmdSibling) { return $npmCmdSibling }
    }
    return $npmExe
}

# @include browser-ensure.ps1
# @include runtime-policy.ps1
# @include host-prerequisites.ps1
# @include repository.ps1
# @include python-environment.ps1
# @include installation-registration.ps1
# @include tool-dependencies.ps1
# @include desktop.ps1
# @include post-install.ps1
# ============================================================================
# Stage protocol
# ============================================================================
#
# install.ps1 supports a small, stable "stage protocol" that lets programmatic
# callers (the desktop GUI's onboarding wizard, CI, future install.sh, etc.)
# drive the install one step at a time and surface progress/errors with their
# own UI.  CLI users running the canonical `irm | iex` one-liner never
# encounter this -- default invocation behaves exactly as before.
#
# Entry points:
#
#   install.ps1                       Interactive install (today's behavior).
#   install.ps1 -ProtocolVersion      Emit the protocol version integer.
#   install.ps1 -Manifest             Emit the stage manifest as JSON.
#   install.ps1 -Stage <name>         Run one stage and emit its result.
#   install.ps1 -NonInteractive       Disable all Read-Host prompts (also
#                                     skips the setup wizard and the gateway
#                                     autostart prompt).  Can be combined
#                                     with default invocation to do a full
#                                     non-interactive install.
#   install.ps1 -Json                 Emit machine-readable JSON instead of
#                                     the human-readable success banner at
#                                     the end of a full install.
#
# Manifest schema (the JSON returned by -Manifest):
#
#   {
#     "protocol_version": 1,
#     "stages": [
#       {
#         "name": "uv",
#         "title": "Installing uv package manager",
#         "category": "prereqs",
#         "needs_user_input": false
#       },
#       ...
#     ]
#   }
#
# Stage result (the JSON written by -Stage <name>):
#
#   {
#     "stage": "uv",
#     "ok": true,
#     "skipped": false,
#     "reason": null,
#     "duration_ms": 1234
#   }
#
# Exit codes:
#
#   0 -- success (stage ran, or stage was deliberately skipped).
#   1 -- generic failure; the stage threw.
#   2 -- unknown stage name passed to -Stage.
#
# Adding a stage:
#
#   1. Append an entry to $InstallStages below.
#   2. Make sure the worker function it points at is idempotent and respects
#      $NonInteractive when it has prompts.  Add it before "configure"
#      (the wizard) or "gateway" (autostart) if it should run unconditionally;
#      after those if it's optional post-install glue.
#   3. Do NOT bump $InstallStageProtocolVersion -- adding stages is additive.
#      Drivers iterate the manifest dynamically.
#
# ============================================================================

# Stage definitions -- the single source of truth.  Each entry maps a stable
# stage name (the API contract drivers depend on) to the worker function that
# implements it.  ``Title`` is what UIs show; ``Category`` lets UIs group
# stages; ``NeedsUserInput`` tells UIs "this stage prompts -- either skip it
# or arrange to provide answers another way."
$InstallStages = @(
    @{ Name = "uv";               Title = "Installing uv package manager";        Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Uv" }
    @{ Name = "git";              Title = "Installing Git";                       Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Git" }
    @{ Name = "node";             Title = "Detecting Node.js";                    Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Node" }
    @{ Name = "system-packages";  Title = "Installing ripgrep and ffmpeg";        Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-SystemPackages" }
    @{ Name = "repository";       Title = "Cloning Hermes repository";            Category = "install";      NeedsUserInput = $false; Worker = "Stage-Repository" }
    # Managed Python lives under $InstallDir\.hermes-runtime, so the checkout
    # must exist before this stage creates that directory. Otherwise the later
    # repository stage treats the runtime-only directory as a broken checkout,
    # parks it, and leaves Stage-Venv with no managed interpreter.
    @{ Name = "python";           Title = "Verifying Python $PythonVersion";      Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Python" }
    @{ Name = "venv";             Title = "Creating Python virtual environment";  Category = "install";      NeedsUserInput = $false; Worker = "Stage-Venv" }
    @{ Name = "dependencies";     Title = "Installing Python dependencies";       Category = "install";      NeedsUserInput = $false; Worker = "Stage-Dependencies" }
    @{ Name = "node-deps";        Title = "Installing Node.js dependencies";      Category = "install";      NeedsUserInput = $false; Worker = "Stage-NodeDeps" }
)
if ($IncludeDesktop) {
    # Insert AFTER node-deps so workspace npm is already installed when
    # the desktop build runs. Inserted only when explicitly requested
    # (Hermes-Setup.exe), never via the irm|iex CLI one-liner.
    $InstallStages += @{ Name = "desktop"; Title = "Building desktop app"; Category = "install"; NeedsUserInput = $false; Worker = "Stage-Desktop" }
}
$InstallStages += @(
    @{ Name = "path";             Title = "Adding Hermes to PATH";                Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-Path" }
    @{ Name = "config-templates"; Title = "Writing configuration templates";      Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-ConfigTemplates" }
    @{ Name = "platform-sdks";    Title = "Installing messaging platform SDKs";   Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-PlatformSdks" }
    @{ Name = "bootstrap-marker"; Title = "Marking install complete";              Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-BootstrapMarker" }
    # Interactive stages.  In non-interactive mode these become no-ops; the
    # caller (GUI / CI) handles the equivalent UX themselves.
    @{ Name = "configure";        Title = "Configuring API keys and models";      Category = "post-install"; NeedsUserInput = $true;  Worker = "Stage-Configure" }
    @{ Name = "gateway";          Title = "Starting messaging gateway";           Category = "post-install"; NeedsUserInput = $true;  Worker = "Stage-Gateway" }
)

# Stage workers -- thin wrappers that delegate to the existing Install-* /
# Test-* / Invoke-* functions while preserving their error semantics.  Kept
# as a separate layer so the existing functions remain callable directly
# (helpful for one-off recovery: ``. install.ps1; Install-Venv``).
#
# Stages that depend on uv (anything after Stage-Uv) call Resolve-UvCmd
# first so they work in cross-process driver mode where $script:UvCmd
# set by Stage-Uv in a sibling powershell process is not visible here.
# Resolve-UvCmd is a fast no-op when $script:UvCmd is already populated
# (the default-invocation case where Main runs everything in one
# process), and throws cleanly if uv truly isn't installed yet.
function Stage-Uv               { if (-not (Install-Uv))     { throw "uv installation failed" } }
function Stage-Python           { Resolve-UvCmd; if (-not (Test-Python))    { throw "Python $PythonVersion not available" } }
function Stage-Git              {
    if (-not (Install-Git)) {
        if ($script:GitInstallFailureReason) { throw $script:GitInstallFailureReason }
        throw "Git not available and auto-install failed -- install from https://git-scm.com/download/win then re-run"
    }
}
# Node is optional (browser tools degrade gracefully without it).  Surface
# failure to the JSON contract as skipped=true / reason rather than ok=true,
# so a GUI driver consuming the manifest can distinguish "node ready" from
# "node missing".  Install flow continues either way -- matches the
# existing Write-Completion behavior that prints a "Note: Node.js could
# not be installed" hint instead of aborting.
function Stage-Node             {
    if (-not (Test-Node)) {
        $script:_StageSkippedReason = "Node.js not available; browser tools will be unavailable until node is installed manually from https://nodejs.org/en/download/"
    }
}
function Stage-SystemPackages   { Install-SystemPackages }
function Stage-Repository       { Install-Repository }
function Stage-Venv             { Resolve-UvCmd; Install-Venv }
function Stage-Dependencies     { Resolve-UvCmd; Install-Dependencies }
function Stage-NodeDeps         { Install-NodeDeps }
function Stage-Desktop          { Install-DesktopVoiceDeps; Install-Desktop }
function Stage-Path             { Set-PathVariable }
function Stage-ConfigTemplates  { Copy-ConfigTemplates }
function Stage-PlatformSdks     { Resolve-UvCmd; Install-PlatformSdks }
function Stage-BootstrapMarker  { Write-BootstrapMarker }
function Stage-Configure        { Invoke-SetupWizard }
function Stage-Gateway          { Start-GatewayIfConfigured }

function Get-InstallStage {
    param([string]$Name)
    foreach ($s in $InstallStages) {
        if ($s.Name -eq $Name) { return $s }
    }
    return $null
}

function Step-OutOfInstallDir {
    # Windows refuses to delete a directory any shell is currently cd'd
    # inside -- and silently leaves orphan files behind, which then wedge
    # "is this a valid git repo" probes on re-install.  Harmless when the
    # caller ran the installer from somewhere else.
    try {
        $currentResolved = (Get-Location).ProviderPath
        $installResolved = $null
        if (Test-Path $InstallDir) {
            $installResolved = (Resolve-Path $InstallDir -ErrorAction SilentlyContinue).ProviderPath
        }
        if ($installResolved -and $currentResolved.ToLower().StartsWith($installResolved.ToLower())) {
            Write-Info "Stepping out of $InstallDir so Windows can replace files there if needed..."
            Set-Location $env:USERPROFILE
        }
    } catch {}
}

function Invoke-Stage {
    param(
        [Parameter(Mandatory=$true)] [hashtable]$StageDef
    )

    # Refresh PATH from registry so this stage sees binaries installed by
    # prior stages, even when each stage runs in its own powershell process.
    # No-op in cost-relevant cases (default invocation path syncs once per
    # foreach pass; cross-process drivers get the necessary freshening).
    Sync-EnvPath

    # Per-stage soft-skip channel.  A worker can populate
    # $script:_StageSkippedReason to surface "ran, but the thing it was
    # supposed to set up is not available" as skipped=true in the JSON
    # frame, without throwing.  Used by Stage-Node so the install flow
    # doesn't abort when an optional capability is missing while still
    # being honest in the protocol contract.  Reset before each stage so
    # a prior stage's reason can never leak into a later stage's frame.
    $script:_StageSkippedReason = $null

    $start = [DateTime]::UtcNow
    $result = @{
        stage        = $StageDef.Name
        ok           = $false
        skipped      = $false
        reason       = $null
        duration_ms  = 0
    }

    try {
        & $StageDef.Worker
        $result.ok = $true
        if ($script:_StageSkippedReason) {
            $result.skipped = $true
            $result.reason  = $script:_StageSkippedReason
        }
    } catch {
        $result.ok = $false
        $result.reason = "$_"
        throw
    } finally {
        $result.duration_ms = [int]([DateTime]::UtcNow - $start).TotalMilliseconds
        if ($Json -or $Stage) {
            # In stage-driver mode every stage emits a JSON line so the
            # caller can stream progress.  In default interactive mode we
            # stay silent here (the worker already wrote human output).
            $result | ConvertTo-Json -Compress | Write-Output
            # Tell the entry-point catch that we've already emitted a
            # frame for this failure (when $result.ok = $false), so it
            # doesn't double-emit a second JSON object and break the
            # one-line-per-stage contract the driver protocol promises.
            if (-not $result.ok) {
                $script:_StageEmittedErrorFrame = $true
            }
        }
    }
}

# ============================================================================
# Main
# ============================================================================

function Invoke-AllStages {
    Step-OutOfInstallDir
    foreach ($s in $InstallStages) {
        Invoke-Stage -StageDef $s
    }
}

function Invoke-EnsureMode {
    param([string]$Deps)
    $depList = $Deps -split ","
    foreach ($dep in $depList) {
        $dep = $dep.Trim()
        switch ($dep) {
            "node" {
                [void](Test-Node)
                if (-not $script:HasNode) {
                    Write-Err "Node.js could not be installed"
                    exit 1
                }
            }
            "browser" {
                [void](Test-Node)
                if ($script:HasNode) {
                    Install-AgentBrowser
                } else {
                    Write-Err "Node.js is required for browser tools but could not be installed"
                    exit 1
                }
            }
            "ripgrep" {
                Write-Info "ripgrep: install manually on Windows (scoop install ripgrep)"
            }
            "ffmpeg" {
                Write-Info "ffmpeg: install manually on Windows (scoop install ffmpeg)"
            }
            default {
                Write-Err "Unknown dependency: $dep"
                exit 1
            }
        }
    }
}

function Invoke-PostInstallMode {
    Write-Info "Running post-install setup..."
    Invoke-EnsureMode -Deps "node,browser"
    Write-Info "Post-install complete"
}

function Main {
    Write-Banner
    Invoke-AllStages
    if (-not $Json) {
        Write-Completion
    } else {
        @{ ok = $true; protocol_version = $InstallStageProtocolVersion } | ConvertTo-Json -Compress | Write-Output
    }
}

# ----------------------------------------------------------------------------
# Entry-point dispatch
# ----------------------------------------------------------------------------
#
# All branches funnel through one try/catch so errors don't kill an `irm |
# iex` PowerShell session, and so failures in stage-driver mode produce a
# structured JSON error frame instead of a bare exception.

# Dot-sourcing loads the installer's real functions for isolated behavioral
# tests without running an install. Normal script and `irm | iex` entry points
# are unchanged.
if ($MyInvocation.InvocationName -eq ".") {
    return
}

try {
    if ($Ensure -ne "") {
        if ($PSBoundParameters.ContainsKey("Stage")) {
            Write-Err "Cannot use -Ensure and -Stage simultaneously"
            exit 1
        }
        Invoke-EnsureMode -Deps $Ensure
        exit 0
    }
    if ($PostInstall) {
        Invoke-PostInstallMode
        exit 0
    }

    if ($ProtocolVersion) {
        Write-Output $InstallStageProtocolVersion
        exit 0
    }

    if ($ShowResolvedPaths) {
        $script:ResolvedPathReport | ConvertTo-Json -Depth 5 -Compress | Write-Output
        exit 0
    }

    if ($Manifest) {
        $payload = @{
            protocol_version = $InstallStageProtocolVersion
            stages = @($InstallStages | ForEach-Object {
                @{
                    name             = $_.Name
                    title            = $_.Title
                    category         = $_.Category
                    needs_user_input = $_.NeedsUserInput
                }
            })
        }
        $payload | ConvertTo-Json -Depth 5 -Compress | Write-Output
        exit 0
    }

    # Use PSBoundParameters rather than $Stage truthiness so that an
    # explicit `-Stage ""` from a misbehaving driver doesn't fall through
    # to the full-install Main path and silently kick off a destructive
    # operation.  Empty string is a contract violation; surface it as
    # unknown-stage exit 2 with a structured JSON frame.
    if ($PSBoundParameters.ContainsKey("Stage")) {
        $def = Get-InstallStage -Name $Stage
        if (-not $def) {
            $err = @{
                ok     = $false
                stage  = $Stage
                reason = "unknown stage: $Stage. Run install.ps1 -Manifest to list valid stages."
            }
            $err | ConvertTo-Json -Compress | Write-Output
            exit 2
        }
        Step-OutOfInstallDir
        Invoke-Stage -StageDef $def
        exit 0
    }

    # Default: full install (today's behavior, plus optional -NonInteractive
    # and -Json layered on by the params above).
    Main
} catch {
    if ($Json -or $Stage) {
        # Stage-driver mode: caller wants JSON they can parse.  Emit a
        # structured error frame and exit non-zero -- BUT only if
        # Invoke-Stage didn't already emit one for this same failure.
        # The inner finally emits the authoritative per-stage frame
        # (with duration_ms + skipped fields); a second emit here
        # would produce two concatenated JSON objects on stdout and
        # break drivers that parse one-line-per-invocation.
        if (-not $script:_StageEmittedErrorFrame) {
            $err = @{
                ok     = $false
                stage  = if ($Stage) { $Stage } else { $null }
                reason = "$_"
            }
            $err | ConvertTo-Json -Compress | Write-Output
        }
        exit 1
    }

    # Interactive mode: keep today's friendly recovery hint.
    Write-Host ""
    Write-Err "Installation failed: $_"
    Write-Host ""
    Write-Info "If the error is unclear, try downloading and running the script directly:"
    Write-Host "  Invoke-WebRequest -Uri 'https://hermes-agent.nousresearch.com/install.ps1' -OutFile install.ps1" -ForegroundColor Yellow
    Write-Host "  .\install.ps1" -ForegroundColor Yellow
    Write-Host ""
}
