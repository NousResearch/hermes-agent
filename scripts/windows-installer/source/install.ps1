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
# Clear the cached Electron download + any half-written unpacked output so the
# next `npm run pack` re-downloads and re-stages from scratch. A corrupt zip in
# the per-user Electron download cache - most often a partial download resumed
# into the same file, leaving concatenated junk - makes electron-builder's
# `app-builder unpack-electron` extract a tree MISSING the electron binary, so
# the final `electron` -> `Hermes` rename dies with ENOENT and every re-run
# repeats the broken extraction forever.
#
# We deliberately do not validate the zip ourselves: the common
# prepended/concatenated-junk corruption slips past naive checks, so a
# self-rolled gate would skip the real-world case. We unconditionally drop the
# cached electron-*.zip (loose copy and any @electron/get hash-subdir copy) plus
# the stale unpacked dir, then let the caller retry once - @electron/get
# re-downloads with its own SHASUM verification, the real source of truth.
#
# Returns the removed paths. Best-effort: never throws.
function Clear-ElectronBuildCache {
    param([string]$DesktopDir)
    $removed = @()

    # Per-user Electron download cache dirs, honoring the overrides @electron/get
    # respects, then the Windows default (%LOCALAPPDATA%\electron\Cache).
    $cacheDirs = @()
    if ($env:electron_config_cache) { $cacheDirs += $env:electron_config_cache }
    if ($env:ELECTRON_CACHE)        { $cacheDirs += $env:ELECTRON_CACHE }
    if ($env:LOCALAPPDATA)          { $cacheDirs += (Join-Path $env:LOCALAPPDATA 'electron\Cache') }
    $cacheDirs += (Join-Path $HOME 'AppData\Local\electron\Cache')

    foreach ($dir in $cacheDirs) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        # Recurse: the bad copy may be the top-level zip OR a copy inside an
        # @electron/get hash subdir.
        $removed += @(Get-ChildItem -LiteralPath $dir -Recurse -Filter 'electron-*.zip' -File -ErrorAction SilentlyContinue | ForEach-Object {
            try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop; $_.FullName } catch { }
        })
    }

    # A half-written unpacked dir from an interrupted prior pack poisons the
    # rename even after the zip is fixed (win-unpacked / win-arm64-unpacked).
    $releaseDir = Join-Path $DesktopDir 'release'
    if (Test-Path -LiteralPath $releaseDir) {
        $removed += @(Get-ChildItem -LiteralPath $releaseDir -Directory -Filter '*-unpacked' -ErrorAction SilentlyContinue | ForEach-Object {
            try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop; $_.FullName } catch { }
        })
    }

    return $removed
}

# Last-resort Electron mirror after GitHub download fails (#47266).
$script:DesktopElectronFallbackMirror = "https://npmmirror.com/mirrors/electron/"

# Electron package dir -- workspace-local nest first, then root hoist.
function Get-ElectronDir {
    param([string]$InstallDir)
    $desktopLocal = Join-Path $InstallDir 'apps\desktop\node_modules\electron'
    if (Test-Path -LiteralPath $desktopLocal) { return $desktopLocal }
    return (Join-Path $InstallDir 'node_modules\electron')
}

# True when dist/ holds a usable Electron binary (#38673 / run-electron-builder.mjs).
function Test-ElectronDist {
    param([string]$InstallDir)
    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    $distExe = Join-Path $electronDir 'dist\electron.exe'
    return (Test-Path -LiteralPath $distExe)
}

# Best-effort: run electron/install.js to populate dist/ (optional mirror).
function Restore-ElectronDist {
    param([string]$InstallDir, [string]$Mirror)
    if (Test-ElectronDist -InstallDir $InstallDir) { return $true }

    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    $distExe = Join-Path $electronDir 'dist\electron.exe'
    $installer = Join-Path $electronDir 'install.js'
    if (-not (Test-Path -LiteralPath $installer)) { return $false }
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { return $false }

    $distDir = Join-Path $electronDir 'dist'
    if (Test-Path -LiteralPath $distDir) {
        Remove-Item -LiteralPath $distDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath (Join-Path $electronDir 'path.txt') -Force -ErrorAction SilentlyContinue

    $prevMirror = $env:ELECTRON_MIRROR
    if ($Mirror) { $env:ELECTRON_MIRROR = $Mirror }
    try {
        # Out-Host so the downloader's progress shows on the console WITHOUT
        # leaking into this function's return value (PowerShell returns every
        # object left on the output stream, so a bare pipe here would make the
        # boolean below ambiguous).
        & $node.Source $installer 2>&1 | ForEach-Object { "$_" } | Out-Host
    } catch {
    } finally {
        $env:ELECTRON_MIRROR = $prevMirror
    }
    return (Test-Path -LiteralPath $distExe)
}

function Test-ElectronPkgStagedMissingDist {
    param([string]$InstallDir)
    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    return (
        (Test-Path -LiteralPath (Join-Path $electronDir 'package.json')) -and
        (Test-Path -LiteralPath (Join-Path $electronDir 'install.js')) -and
        (-not (Test-ElectronDist -InstallDir $InstallDir))
    )
}

function Try-RestoreElectronDist {
    param([string]$InstallDir)
    if (Restore-ElectronDist -InstallDir $InstallDir) { return $true }
    if ($env:ELECTRON_MIRROR) { return $false }
    return Restore-ElectronDist -InstallDir $InstallDir -Mirror $script:DesktopElectronFallbackMirror
}

function Install-DesktopVoiceDeps {
    # Desktop ships with working voice out of the box: eagerly install the
    # wake-word + local-STT stacks ([wake] + [voice] extras) instead of
    # leaving them to lazy first-use install. Policy change (Teknium, July
    # 2026, #70509 testing): the first ear-click used to trigger a
    # multi-minute onnxruntime pip install that froze the UI and blew RPC
    # timeouts. Best-effort -- lazy install remains the fallback for anything
    # this step fails to fetch.
    if (-not $script:UvCmd) { Resolve-UvCmd }
    if (-not $script:UvCmd) {
        Write-Warn "uv unavailable -- voice/wake deps will lazy-install at first use instead"
        return
    }
    $env:VIRTUAL_ENV = "$InstallDir\venv"
    Write-Info "Installing voice + wake-word dependencies (onnxruntime, faster-whisper -- 1-3min)..."
    Push-Location $InstallDir
    try {
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install -e ".[wake,voice]" }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Voice + wake-word dependencies installed"
        } else {
            Write-Warn "Voice/wake dependency install failed (exit $LASTEXITCODE) -- they will lazy-install at first use"
        }
    } finally {
        Pop-Location
    }
}

function Install-Desktop {
    # Build apps/desktop into a launchable Hermes.exe. Only called from
    # Stage-Desktop, which is itself only included in the manifest when
    # -IncludeDesktop was passed to install.ps1.
    #
    # The workspace npm install at repo root (done by Install-NodeDeps for
    # browser tools) does NOT pull apps/desktop's dependencies, because the
    # browser-tools workspace at $InstallDir\package.json is a separate
    # workspace from apps/*. We do a full root-level `npm install` here
    # so the workspace resolves apps/desktop's deps (including Electron
    # itself, ~150MB), then run `npm run pack` in apps/desktop which
    # produces the unpacked binary at apps/desktop/release/<os>-unpacked/.
    #
    # The Tauri bootstrap installer's launch_hermes_desktop command
    # resolves apps/desktop/release/win-unpacked/Hermes.exe directly,
    # so an "unpacked" build (electron-builder --dir) is enough -- we
    # don't need to produce an NSIS/MSI artifact here.

    # Always re-resolve Node here. Stages run in separate PowerShell processes,
    # so $script:HasNode from Stage-Node isn't visible; more importantly Test-Node
    # enforces the supported Node lines and prepends the Hermes-managed Node to
    # PATH, so the build never runs on an unsupported system Node -- the cause
    # of the opaque "Build desktop app ... exit code 1" failure (Vite crashes on
    # old Node).
    Test-Node | Out-Null
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warn "Skipping desktop build (Node.js / npm not on PATH)"
        $script:_StageSkippedReason = "Node.js not available"
        return
    }

    $desktopDir = "$InstallDir\apps\desktop"
    if (-not (Test-Path "$desktopDir\package.json")) {
        Write-Warn "Skipping desktop build (apps/desktop not present in checkout)"
        $script:_StageSkippedReason = "apps/desktop not present"
        return
    }

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Warn "Skipping desktop build (npm not on PATH)"
        $script:_StageSkippedReason = "npm not found"
        return
    }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $sibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $sibling) { $npmExe = $sibling }
    }

    # 1. Workspace-level install so apps/desktop's deps (Electron, Vite,
    # node-pty prebuilds, etc.) actually land in node_modules. This is
    # the SAME `npm install` Install-NodeDeps does for browser tools,
    # but at the root rather than the browser-tools workspace, so all
    # apps/* workspaces resolve.
    Write-Info "Installing desktop workspace dependencies (this includes Electron ~150MB, takes 1-3min)..."
    Push-Location $InstallDir
    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        # Drop --silent so npm emits its full progress + error trail.
        # When this fails on a non-dev box (e.g. native-module build
        # without VS Build Tools, ETARGET on a transitive, etc.), the
        # actual reason needs to reach the Tauri installer's log; with
        # --silent it was completely suppressed and the user just saw
        # "exit 1" with no actionable detail.
        #
        # The streaming sink in bootstrap.rs's run_install_script
        # captures every stdout/stderr line as it's emitted, so we don't
        # need a side TEMP log file -- the installer's bootstrap log
        # IS the artifact a support engineer reads.
        #
        # Prefer `npm ci`: it wipes node_modules and reinstalls from the
        # lockfile, always producing a complete tree. Bare `npm install`
        # can report "up to date" against a stale
        # node_modules\.package-lock.json marker while node_modules is
        # actually empty (Windows workspace-hoisting flake), leaving
        # tsc/typescript unresolved so `npm run pack`'s `tsc -b` dies with
        # no obvious cause. Fall back to `npm install` only if `npm ci`
        # fails (lockfile out of sync / very old npm without ci).
        #
        # Tee the merged output into $npmOut while still emitting every line
        # live. We don't need a side log file (the bootstrap streaming sink
        # is the artifact), but on failure we scan $npmOut for the TLS-trust
        # signature so corporate-proxy users get the NODE_EXTRA_CA_CERTS hint
        # instead of an opaque "exit 1" (issue #38016).
        & $npmExe ci 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable npmOut
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            Write-Info "  npm ci failed (exit $code) -- retrying with npm install..."
            & $npmExe install 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable npmOut
            $code = $LASTEXITCODE
        }
        $ErrorActionPreference = $prevEAP
        if ($code -ne 0) {
            if (Test-ElectronPkgStagedMissingDist -InstallDir $InstallDir) {
                Write-Warn "Desktop dependency install failed with a missing Electron dist; attempting self-heal..."
                Try-RestoreElectronDist -InstallDir $InstallDir | Out-Null
            } else {
                Show-NpmCertHint ($npmOut -join "`n") | Out-Null
                # Replay npm's own debug log into our stream: the terse
                # summary above rarely contains the postinstall stderr
                # (e.g. Electron's install.js) that explains the failure.
                Write-NpmDebugLogTail -NpmOutput ($npmOut -join "`n")
                throw "desktop workspace npm install failed (exit $code) -- see lines above for cause"
            }
        } else {
            Write-Success "Desktop workspace dependencies installed"
        }
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Pop-Location
        throw
    }
    Pop-Location

    # 2. Build apps/desktop. `npm run pack` runs:
    #      assert-root-install + write-build-stamp + stage-native-deps +
    #      tsc -b + vite build + electron-builder --dir
    # The --dir mode produces an unpacked Hermes.exe in
    # apps/desktop/release/win-unpacked/ without bundling NSIS/MSI;
    # we don't need a distributable installer artifact, just a
    # launchable binary the Tauri installer can spawn.
    #
    # CSC_IDENTITY_AUTO_DISCOVERY=false tells electron-builder we are
    # NOT signing the output. Combined with signAndEditExecutable=false in
    # apps/desktop/package.json's build.win block, electron-builder never
    # invokes signtool and therefore never fetches/extracts winCodeSign
    # (whose macOS symlinks crash 7-Zip on non-admin Windows -- a dead end we
    # are NOT trying to work around). The Hermes icon + product name are
    # stamped onto Hermes.exe by our own rcedit step (Set-DesktopExeIdentity)
    # AFTER this build, completely decoupled from electron-builder signing.
    #
    # WIN_CSC_LINK and WIN_CSC_KEY_PASSWORD explicitly cleared as
    # belt-and-suspenders: if the user's environment has them set
    # for some other tool, electron-builder would still try to sign.
    Write-Info "Building desktop app (this takes 1-3 minutes)..."
    $buildLog = "$env:TEMP\hermes-desktop-build-$(Get-Random).log"
    # Seed GITHUB_SHA for write-build-stamp.mjs. The stamp prefers CI env vars
    # over `git rev-parse`, so this covers: (1) node can't find git.exe on PATH
    # even though this PowerShell session can, (2) ZIP/init trees that still
    # lack a HEAD after a failed post-extract fetch. Without it the desktop
    # pack dies with "could not determine git commit" (#50823).
    if (-not $env:GITHUB_SHA) {
        if ($Commit) {
            $env:GITHUB_SHA = $Commit
        } else {
            Push-Location $InstallDir
            try {
                $global:LASTEXITCODE = 0
                $resolvedSha = & git -c windows.appendAtomically=false rev-parse HEAD 2>$null
                if ($LASTEXITCODE -ne 0 -or -not $resolvedSha) {
                    # ZIP path may have FETCH_HEAD after a fetch even when HEAD is unset.
                    $global:LASTEXITCODE = 0
                    $resolvedSha = & git -c windows.appendAtomically=false rev-parse FETCH_HEAD 2>$null
                }
                if ($LASTEXITCODE -eq 0 -and $resolvedSha) {
                    $env:GITHUB_SHA = ("$resolvedSha").Trim()
                }
            } catch { } finally {
                Pop-Location
            }
        }
    }
    if (-not $env:GITHUB_REF_NAME) {
        $env:GITHUB_REF_NAME = if ($Branch) { $Branch } else { "main" }
    }
    if ($env:GITHUB_SHA) {
        $shaPreview = if ($env:GITHUB_SHA.Length -ge 12) { $env:GITHUB_SHA.Substring(0, 12) } else { $env:GITHUB_SHA }
        Write-Info "Desktop build stamp: $shaPreview ($($env:GITHUB_REF_NAME))"
    } else {
        Write-Warn "Could not resolve a git commit for the desktop stamp -- write-build-stamp will use its non-git fallback"
    }
    Push-Location $desktopDir
    $prevEAP = $ErrorActionPreference
    $prevCSCAuto = $env:CSC_IDENTITY_AUTO_DISCOVERY
    $prevWinCscLink = $env:WIN_CSC_LINK
    $prevWinCscKeyPassword = $env:WIN_CSC_KEY_PASSWORD
    try {
        $ErrorActionPreference = "Continue"
        $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
        $env:WIN_CSC_LINK = ""
        $env:WIN_CSC_KEY_PASSWORD = ""
        & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            $purged = @()
            $restored = $false
            if (-not (Test-ElectronDist -InstallDir $InstallDir)) {
                $purged = @(Clear-ElectronBuildCache -DesktopDir $desktopDir)
                $restored = Restore-ElectronDist -InstallDir $InstallDir
            }
            if ($restored) {
                Write-Warn "Desktop build failed - refreshed the Electron download, retrying once:"
                foreach ($p in $purged) { Write-Info "  - $p" }
                & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
                $code = $LASTEXITCODE
            }
        }
        if ($code -ne 0 -and -not $env:ELECTRON_MIRROR) {
            $mirror = $script:DesktopElectronFallbackMirror
            Write-Warn "Desktop build still failing - the Electron download from GitHub looks blocked."
            Write-Warn "Re-downloading Electron via a public mirror ($mirror), then rebuilding:"
            Write-Info "  (set ELECTRON_MIRROR yourself to use a different/trusted mirror)"
            if (-not (Test-ElectronDist -InstallDir $InstallDir)) {
                Restore-ElectronDist -InstallDir $InstallDir -Mirror $mirror | Out-Null
            }
            $prevMirror = $env:ELECTRON_MIRROR
            $env:ELECTRON_MIRROR = $mirror
            try {
                & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
                $code = $LASTEXITCODE
            } finally {
                $env:ELECTRON_MIRROR = $prevMirror
            }
        }
        $ErrorActionPreference = $prevEAP
        if ($code -ne 0) {
            $errText = Get-Content $buildLog -Raw -ErrorAction SilentlyContinue
            if ($errText) {
                $snippet = if ($errText.Length -gt 1800) { $errText.Substring(0, 1800) + "..." } else { $errText }
                Write-Info "  desktop build output:"
                foreach ($line in $snippet -split "`n") { Write-Host "    $line" -ForegroundColor DarkGray }
                Write-Info "  Full log: $buildLog"
            }
            # `npm run pack` failures (lifecycle script exits) also land in
            # npm's debug log; replay it so the bootstrap log carries the
            # full evidence even when $buildLog's tail cuts off the cause.
            Write-NpmDebugLogTail -NpmOutput $errText
            throw "apps/desktop build failed (exit $code)"
        }
        Write-Success "Desktop app built"
        Remove-Item -LiteralPath $buildLog -Force -ErrorAction SilentlyContinue
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Pop-Location
        throw
    } finally {
        # Restore env to whatever the caller had -- don't leak our
        # signing-off override into anything install.ps1 invokes later
        # (Stage-PlatformSdks, etc.).
        $env:CSC_IDENTITY_AUTO_DISCOVERY = $prevCSCAuto
        $env:WIN_CSC_LINK = $prevWinCscLink
        $env:WIN_CSC_KEY_PASSWORD = $prevWinCscKeyPassword
    }
    Pop-Location

    # 3. Sanity-check the produced binary. Probe both arches so this works
    # on x64 and arm64 build machines.
    $exeCandidates = @(
        "$desktopDir\release\win-unpacked\Hermes.exe",
        "$desktopDir\release\win-arm64-unpacked\Hermes.exe"
    )
    $found = $false
    $desktopExe = $null
    foreach ($cand in $exeCandidates) {
        if (Test-Path $cand) {
            Write-Success "Desktop ready: $cand"
            $desktopExe = $cand
            $found = $true
            break
        }
    }
    if (-not $found) {
        throw "Desktop build completed but no Hermes.exe was found under $desktopDir\release\*-unpacked\"
    }

    # 3b. The Hermes icon + identity are stamped onto Hermes.exe by the
    #     electron-builder `afterPack` hook (apps/desktop/scripts/after-pack.mjs)
    #     during `npm run pack` above -- for every build, so the installer's
    #     --update rebuild stays branded too. No separate stamp step needed here.
    #     electron-builder's own rcedit step stays disabled (signAndEditExecutable
    #     =false) because enabling it drags in signtool -> winCodeSign -> the
    #     unfixable symlink crash; the afterPack hook runs rcedit directly.

    # 3c. Grant ALL APPLICATION PACKAGES (S-1-15-2-2) RX on the unpacked app
    #     directory. Chromium's GPU/renderer sandboxes CHECK-fail with
    #     0x80000003 when this ACE is missing alongside orphan AppContainer
    #     SIDs under %LOCALAPPDATA% (electron/electron#51761, hermes-agent#38216).
    #     Best-effort -- never fail an otherwise-good install over ACL repair.
    try {
        $appDir = Split-Path -Parent $desktopExe
        & icacls $appDir /grant "*S-1-15-2-2:(OI)(CI)(RX)" /T /C /Q | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Granted AppContainer read access on $appDir"
        } else {
            Write-Warn "icacls AppContainer grant returned exit $LASTEXITCODE for $appDir"
        }
    } catch {
        Write-Warn "Could not grant AppContainer ACL: $($_.Exception.Message)"
    }

    # 4. Create Start Menu + Desktop shortcuts pointing DIRECTLY at the packed
    #    Hermes.exe. We deliberately do NOT point them at `hermes desktop`: that
    #    command rebuilds (npm install + electron-builder) on every launch,
    #    which would cost minutes each time. The packed exe is the consumer --
    #    launching it directly is instant, and updates flow through the
    #    installer's --update path (which rebuilds once, then relaunches).
    New-DesktopShortcuts -TargetExe $desktopExe
}

function New-DesktopShortcuts {
    param([Parameter(Mandatory = $true)][string]$TargetExe)

    # Best-effort: a shortcut failure must never fail an otherwise-good install.
    try {
        $shell = New-Object -ComObject WScript.Shell
        $workDir = Split-Path -Parent $TargetExe

        # Prefer the standalone icon.ico (shipped beside the exe via
        # electron-builder extraResources -> resources/icon.ico) over the exe's
        # embedded resource. An explicit .ico path is more stable across update
        # cycles: pointing at "$TargetExe,0" makes Windows cache the icon it
        # extracted from the exe at shortcut-creation time, and that cached
        # bitmap can persist (showing the OLD/Electron icon) even after the exe
        # is re-stamped on update. A dedicated .ico sidesteps that extraction.
        $iconIco = Join-Path $workDir 'resources\icon.ico'
        if (Test-Path $iconIco) {
            $iconLocation = "$iconIco,0"
        } else {
            $iconLocation = "$TargetExe,0"
        }

        $targets = @(
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'Hermes.lnk'),
            (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Hermes.lnk')
        )

        foreach ($lnkPath in $targets) {
            try {
                $parent = Split-Path -Parent $lnkPath
                if (-not (Test-Path $parent)) {
                    New-Item -ItemType Directory -Force -Path $parent | Out-Null
                }
                $sc = $shell.CreateShortcut($lnkPath)
                $sc.TargetPath = $TargetExe
                $sc.WorkingDirectory = $workDir
                $sc.IconLocation = $iconLocation
                $sc.Description = 'Hermes Agent'
                $sc.Save()
                Write-Success "Shortcut created: $lnkPath"
            } catch {
                Write-Warn "Could not create shortcut $lnkPath : $($_.Exception.Message)"
            }
        }

        # Bust the Windows shell icon cache so the desktop/Start-Menu shortcut
        # repaints with the (possibly newly-stamped) icon instead of a stale
        # cached bitmap. Critical on the --update path: the exe was re-stamped
        # with the Hermes icon, but without this the shortcut can keep drawing
        # the old Electron icon until the user manually refreshes / reboots.
        # Best-effort and silent -- never fail the install over a cosmetic cache.
        try {
            & ie4uinit.exe -show 2>$null
        } catch {
            # ie4uinit may be absent/renamed on some SKUs -- ignore.
        }
    } catch {
        Write-Warn "Skipping shortcut creation: $($_.Exception.Message)"
    }
}

function Install-PlatformSdks {
    # Ensure messaging-platform SDKs matching tokens the user added to
    # ~/.hermes/.env are importable.  Two problems this solves:
    #
    # 1. The tiered `uv pip install` cascade above can fall through to a
    #    lower tier when the first fails (common when RL git deps choke),
    #    which silently skips some messaging SDKs from [messaging].
    # 2. `uv` creates the venv without pip.  If a messaging SDK ends up
    #    missing, the user can't `pip install python-telegram-bot` to
    #    recover -- pip simply isn't in their venv.
    #
    # Strategy: bootstrap pip via `python -m ensurepip` (idempotent), then
    # for each token set in .env, verify the matching SDK imports.  If not,
    # run one targeted `pip install` as last-chance recovery.  Keeps fresh
    # Windows installs from hitting silent "python-telegram-bot not installed"
    # at runtime.
    if ($NoVenv) {
        Write-Info "Skipping platform-SDK verification (-NoVenv: no venv to bootstrap)"
        return
    }

    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Warn "Skipping platform-SDK verification: $pythonExe not found"
        return
    }

    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) { return }
    $envLines = Get-Content $envPath -ErrorAction SilentlyContinue

    # Map: env var set in .env -> (import name, pip spec matching [messaging] extra).
    # Specs mirror pyproject.toml to avoid version drift.
    $sdkMap = @(
        @{ Var = "TELEGRAM_BOT_TOKEN"; Import = "telegram";  Spec = "python-telegram-bot[webhooks]>=22.6,<23" },
        @{ Var = "DISCORD_BOT_TOKEN";  Import = "discord";   Spec = "discord.py[voice]>=2.7.1,<3" },
        @{ Var = "SLACK_BOT_TOKEN";    Import = "slack_sdk"; Spec = "slack-sdk>=3.27.0,<4" },
        @{ Var = "SLACK_APP_TOKEN";    Import = "slack_bolt";Spec = "slack-bolt>=1.18.0,<2" },
        @{ Var = "WHATSAPP_ENABLED";   Import = "qrcode";    Spec = "qrcode>=7.0,<8" }
    )

    # Which tokens are actually set (not placeholder)?
    $needed = @()
    foreach ($sdk in $sdkMap) {
        $match = $envLines | Where-Object {
            $_ -match ("^" + [regex]::Escape($sdk.Var) + "=.+") `
            -and $_ -notmatch "your-token-here" `
            -and $_ -notmatch "^\s*#"
        }
        if ($match) { $needed += $sdk }
    }
    if ($needed.Count -eq 0) { return }

    Write-Host ""
    Write-Info "Verifying platform SDKs for tokens found in $envPath ..."

    # Verify each SDK's import without triggering side-effect imports.
    # Quirk: PowerShell wraps non-zero-exit native stderr as a
    # NativeCommandError that prints even with `2>$null` / `*> $null`
    # unless we set $ErrorActionPreference to SilentlyContinue for the
    # span.  Save + restore rather than nuking globally.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $missing = @()
        foreach ($sdk in $needed) {
            & $pythonExe -c "import $($sdk.Import)" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $missing += $sdk
                Write-Warn "  $($sdk.Import) NOT importable (needed for $($sdk.Var))"
            } else {
                Write-Success "  $($sdk.Import) OK"
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($missing.Count -eq 0) { return }

    # Bootstrap pip into the venv if it isn't there.  `uv` creates venvs
    # without pip; ensurepip is the stdlib-blessed way to add it.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & $pythonExe -m pip --version 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Info "Bootstrapping pip into venv (uv doesn't ship pip)..."
            & $pythonExe -m ensurepip --upgrade 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "ensurepip failed -- can't auto-install missing SDKs."
                Write-Info "Manual recovery: $UvCmd pip install `"$($missing[0].Spec)`""
                return
            }
        }

        foreach ($sdk in $missing) {
            Write-Info "  Installing $($sdk.Spec) ..."
            & $pythonExe -m pip install $sdk.Spec 2>&1 | ForEach-Object { Write-Host "    $_" }
            if ($LASTEXITCODE -eq 0) {
                Write-Success "  Installed $($sdk.Import)"
            } else {
                Write-Warn "  Failed to install $($sdk.Spec). Recover manually: $pythonExe -m pip install `"$($sdk.Spec)`""
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}

function Invoke-SetupWizard {
    if ($SkipSetup) {
        Write-Info "Skipping setup wizard (-SkipSetup)"
        return
    }

    if ($NonInteractive) {
        # The setup wizard prompts for API keys, model choice, persona, etc.
        # Non-interactive callers (GUI installer) own that UX themselves; let
        # them drive it after install.ps1 returns.
        Write-Info "Skipping setup wizard (non-interactive). Configure via the GUI or 'hermes setup'."
        return
    }

    Write-Host ""
    Write-Info "Starting setup wizard..."
    Write-Host ""

    Push-Location $InstallDir

    # Run hermes setup using the venv Python directly (no activation needed)
    if (-not $NoVenv) {
        & ".\venv\Scripts\python.exe" -m hermes_cli.main setup
    } else {
        python -m hermes_cli.main setup
    }

    Pop-Location
}

function Start-GatewayIfConfigured {
    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) { return }

    $hasMessaging = $false
    $content = Get-Content $envPath -ErrorAction SilentlyContinue
    foreach ($var in @("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "WHATSAPP_ENABLED")) {
        $match = $content | Where-Object { $_ -match "^${var}=.+" -and $_ -notmatch "your-token-here" }
        if ($match) { $hasMessaging = $true; break }
    }

    if (-not $hasMessaging) { return }

    $hermesCmd = "$InstallDir\venv\Scripts\hermes.exe"
    if (-not (Test-Path $hermesCmd)) {
        $hermesCmd = "hermes"
    }

    # If WhatsApp is enabled but not yet paired, run foreground for QR scan
    $whatsappEnabled = $content | Where-Object { $_ -match "^WHATSAPP_ENABLED=true" }
    $whatsappSession = "$HermesHome\whatsapp\session\creds.json"
    if ($whatsappEnabled -and -not (Test-Path $whatsappSession)) {
        Write-Host ""
        Write-Info "WhatsApp is enabled but not yet paired."
        Write-Info "Running 'hermes whatsapp' to pair via QR code..."
        Write-Host ""
        # Non-interactive callers (GUI installer, CI) skip the QR-pair prompt;
        # WhatsApp pairing requires a human looking at a phone camera, so the
        # downstream UI is responsible for surfacing this when it makes sense.
        if (-not $NonInteractive) {
            $response = Read-Host "Pair WhatsApp now? [Y/n]"
            if ($response -eq "" -or $response -match "^[Yy]") {
                try {
                    & $hermesCmd whatsapp
                } catch {
                    # Expected after pairing completes
                }
            }
        } else {
            Write-Info "Skipping WhatsApp pairing prompt (non-interactive)."
        }
    }

    Write-Host ""
    Write-Info "Messaging platform token detected!"
    Write-Info "The gateway handles messaging platforms and cron job execution."
    Write-Host ""

    # In non-interactive mode the gateway lifecycle is the caller's problem
    # (the GUI manages its own gateway process, CI doesn't want background
    # services on the build agent, etc.).  Treat it like the user declined.
    if ($NonInteractive) {
        Write-Info "Skipping gateway autostart prompt (non-interactive)."
        Write-Info "Start the gateway later with: hermes gateway"
        return
    }

    $response = Read-Host "Would you like to start the gateway now? [Y/n]"

    if ($response -eq "" -or $response -match "^[Yy]") {
        Write-Info "Starting gateway in background..."
        try {
            $logFile = "$HermesHome\logs\gateway.log"
            Start-Process -FilePath $hermesCmd -ArgumentList "gateway" `
                -RedirectStandardOutput $logFile `
                -RedirectStandardError "$HermesHome\logs\gateway-error.log" `
                -WindowStyle Hidden
            Write-Success "Gateway started! Your bot is now online."
            Write-Info "Logs: $logFile"
            Write-Info "To stop: close the gateway process from Task Manager"
        } catch {
            Write-Warn "Failed to start gateway. Run manually: hermes gateway"
        }
    } else {
        Write-Info "Skipped. Start the gateway later with: hermes gateway"
    }
}

function Write-Completion {
    Write-Host ""
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Green
    Write-Host "|              [OK] Installation Complete!                |" -ForegroundColor Green
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Green
    Write-Host ""
    
    # Show file locations
    Write-Host "* Your files:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Config:    " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\config.yaml"
    Write-Host "   API Keys:  " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\.env"
    Write-Host "   Data:      " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\cron\, sessions\, logs\"
    Write-Host "   Code:      " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\hermes-agent\"
    Write-Host ""
    
    Write-Host "---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "* Commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   hermes              " -NoNewline -ForegroundColor Green
    Write-Host "Start chatting"
    Write-Host "   hermes setup        " -NoNewline -ForegroundColor Green
    Write-Host "Configure API keys & settings"
    Write-Host "   hermes config       " -NoNewline -ForegroundColor Green
    Write-Host "View/edit configuration"
    Write-Host "   hermes config edit  " -NoNewline -ForegroundColor Green
    Write-Host "Open config in editor"
    Write-Host "   hermes gateway      " -NoNewline -ForegroundColor Green
    Write-Host "Start messaging gateway (Telegram, Discord, etc.)"
    Write-Host "   hermes update       " -NoNewline -ForegroundColor Green
    Write-Host "Update to latest version"
    Write-Host ""
    
    Write-Host "---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[*] Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $HasNode) {
        Write-Host "Note: Node.js could not be installed automatically." -ForegroundColor Yellow
        Write-Host "Browser tools need Node.js. Install manually:" -ForegroundColor Yellow
        Write-Host "  https://nodejs.org/en/download/" -ForegroundColor Yellow
        Write-Host ""
    }
    
    if (-not $HasRipgrep) {
        Write-Host "Note: ripgrep (rg) was not installed. For faster file search:" -ForegroundColor Yellow
        Write-Host "  winget install BurntSushi.ripgrep.MSVC" -ForegroundColor Yellow
        Write-Host ""
    }
}

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
