# Hermes Agent bootstrap: git checkout + venv + hermes command on PATH.
# Heavy dependencies (tool binaries, browsers, node) are pm's job after
# this: `hermes pm install`. Stage protocol kept for Hermes-Setup:
#   -Manifest             print the stage list as JSON
#   -Stage NAME [-Json]   run one stage
#   -NonInteractive       skip stages that need input
#   -IncludeDesktop       add the desktop build stage
#   -ProtocolVersion      print the stage protocol version
param(
    [string]$Branch = "main",
    [string]$Commit = "",
    [string]$HermesHome = $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }),
    [string]$InstallDir = $(if ($env:HERMES_HOME) { "$env:HERMES_HOME\hermes-agent" } else { "$env:LOCALAPPDATA\hermes\hermes-agent" }),
    [switch]$Manifest,
    [string]$Stage,
    [switch]$ProtocolVersion,
    [switch]$NonInteractive,
    [switch]$Json,
    [switch]$IncludeDesktop,
    # Print the paths this install would use, as JSON on stdout, and exit
    # without touching anything. The first question on any "installer says a
    # path doesn't exist" report is which paths it actually resolved --
    # especially on profiles Windows exposes through an 8.3 alias.
    #   powershell -File install.ps1 -ShowResolvedPaths
    [switch]$ShowResolvedPaths
)

$ErrorActionPreference = "Stop"

# --- Dot-source guard (part 1: detect) ---------------------------------------
# Tests (and any embedding host) dot-source this file (`. install.ps1`) to get
# at its FUNCTIONS. Only the definitions must enter the caller's session --
# the install itself must never run, not even its side-effectful-looking
# prologue (the 8.3 normalization below rewrites process env vars). Dot-sourced
# files see InvocationName '.'; a real invocation sees the script
# path/expression. The flag is checked before the entry dispatch at the bottom
# (part 2), so dot-sourcing still loads every function definition.
$script:IsDotSourced = $MyInvocation.InvocationName -eq '.'
# $PSBoundParameters inside a FUNCTION refers to the function's own binding,
# so the script's binding is captured here, once, at script scope.
$script:BoundParams = $PSBoundParameters
$RepoUrl = if ($env:HERMES_REPO_URL) { $env:HERMES_REPO_URL } else { "https://github.com/NousResearch/hermes-agent.git" }

# --- BEGIN GENERATED: bootstrap pins (scripts/gen-bootstrap-pins.py) ---
# Derived from pm/lock.json. DO NOT EDIT BY HAND:
# run scripts/gen-bootstrap-pins.py after a pin bump.
$script:UvPinVersion = "0.12.3"
$script:UvPinFiles = @{
    "win32-x64" = @{
        Url    = "https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-pc-windows-msvc.zip"
        MirrorUrl = "https://hermes-assets.nousresearch.com/upstream/sha256/b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"
        Sha256 = "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"
    }
    "win32-arm64" = @{
        Url    = "https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-pc-windows-msvc.zip"
        MirrorUrl = "https://hermes-assets.nousresearch.com/upstream/sha256/4343217d668727b8a8eb5cad92389a1d2eeead93c89940d1b955ba1bb15462eb"
        Sha256 = "4343217d668727b8a8eb5cad92389a1d2eeead93c89940d1b955ba1bb15462eb"
    }
}

$script:GitPinVersion = "2.53.0+3"
$script:GitPinFiles = @{
    "win32-x64" = @{
        Url    = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.3/Git-2.53.0.3-64-bit.tar.bz2"
        MirrorUrl = "https://hermes-assets.nousresearch.com/upstream/sha256/1661f02e85a7901ad7920e2a358ee3772ed9066b00d8590bf2d9046ef10aa8b2"
        Sha256 = "1661f02e85a7901ad7920e2a358ee3772ed9066b00d8590bf2d9046ef10aa8b2"
    }
    "win32-arm64" = @{
        Url    = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.3/Git-2.53.0.3-arm64.tar.bz2"
        MirrorUrl = "https://hermes-assets.nousresearch.com/upstream/sha256/4015f05a68bd2bcf3cc6c426e8d44b65d670fbb879225bb7b7c347cfc3a2758a"
        Sha256 = "4015f05a68bd2bcf3cc6c426e8d44b65d670fbb879225bb7b7c347cfc3a2758a"
    }
}
# --- END GENERATED: bootstrap pins ---

# ============================================================================
# 8.3 short-path normalization
# ============================================================================
# Windows generates an 8.3 short alias for a user-profile folder whose name
# contains a space ("First Last" -> FIRST~1.LAS), a dot, or an accented
# character. It can then expose %TEMP%, %TMP%, %LOCALAPPDATA%, %APPDATA% and
# %USERPROFILE% -- plus everything derived from them, including the default
# HERMES_HOME and InstallDir -- in that short form:
#   C:\Users\FIRST~1.LAS\AppData\Local\Temp
# PowerShell's FileSystem provider mishandles the aliased component once it
# reaches a provider cmdlet (Tee-Object -FilePath, Out-File, New-Item,
# Test-Path), throwing "An object at the specified path ... does not exist".
# Expanding every profile-rooted path back to long form once, up front, lets
# every downstream cmdlet and child process see something the provider can
# resolve. Three resolvers, tried in order, because no single one covers every
# host:
#   1. kernel32!GetLongPathNameW -- expands any 8.3 component regardless of
#      locale.
#   2. Scripting.FileSystemObject -- fallback where P/Invoke is blocked.
#   3. Profile-root substitution -- when the volume has 8.3 generation
#      disabled or the alias is stale, neither resolver can expand the name
#      because it no longer maps to anything on disk. The aliased component
#      is always the profile folder itself (everything below it was created
#      long), so swap in a profile root we can prove is long and reattach
#      the tail.
# All three degrade to returning the input untouched, so a host where none
# of them apply -- including non-Windows -- behaves exactly as before.

$script:LongProfileRoot = $null

function Write-PathDiag {
    # Diagnostics for this block go to stderr, never stdout: the stage
    # protocol hands drivers a single line of JSON on stdout and a stray note
    # would break anything parsing it. Suppressed entirely under
    # -ShowResolvedPaths, which is a machine-readable query: Windows
    # PowerShell 5.1 wraps any native-command stderr in a NativeCommandError
    # and folds it back into the caller's own stream, so a child writing here
    # at all is enough to corrupt a 5.1 caller's capture. The JSON already
    # carries everything these lines say.
    param([string]$Message)
    if ($ShowResolvedPaths) { return }
    [Console]::Error.WriteLine("[hermes] $Message")
}

function Get-LongProfileRoot {
    # The user's profile directory in long form, or '' when every source we
    # can reach is itself aliased. Cached: this runs per env var.
    if ($null -ne $script:LongProfileRoot) { return $script:LongProfileRoot }
    $script:LongProfileRoot = ''

    # %USERPROFILE% first: it is what the rest of the install derives from.
    # Then the HOMEDRIVE/HOMEPATH pair, then the profile's parent (C:\Users
    # never carries an alias) plus %USERNAME%, which stays the long account
    # name even when every path is short.
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

    if ($script:LongProfileRoot) {
        Write-PathDiag "long profile root: $script:LongProfileRoot"
    } else {
        Write-PathDiag "no long profile root found; 8.3 paths left as-is (tried: $($candidates -join ', '))"
    }
    return $script:LongProfileRoot
}

function Expand-ShortProfileRoot {
    # Rebuild $Path onto a known-long profile root when its aliased component
    # is the profile folder. Returns $Path unchanged when it isn't, so a
    # custom TEMP on another volume (D:\SHORT~1\Temp) is never rewritten.
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
    # Only 8.3 short names carry a tilde+digit ("~1"); skip every resolver
    # for ordinary long paths, which is the overwhelmingly common case.
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
    #    resolver can report success and still hand back a path that carries
    #    the alias (observed on a windows-latest runner). An unexpanded
    #    result counts as failure and falls through.
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
            Write-PathDiag "expanded 8.3 short path in %$name%: $current -> $expanded"
        }
    }
    return $rewrote
}

# ConvertTo-LongPath only assigns $script:LastResolver when a ~\d short path
# actually needs expansion, so an ordinary long profile leaves it unset --
# and the report below reads it unconditionally. 'none' is the resolver's own
# value for "nothing ran".
$script:LastResolver = 'none'
$script:NormalizedPathRewrites = @{}

# (Dot-source guard, prologue side: a dot-source must not rewrite the
# caller's process env, so the normalization prologue runs only on real
# entry. Called from the entry dispatch below, before -ProtocolVersion and
# every other switch, so the resolved paths are always the install's own.)
function Initialize-ResolvedPaths {
    $script:NormalizedProfilePaths = Set-LongProfileEnvVars

    # Re-derive the install paths now that the env vars behind their defaults
    # are long. An explicitly passed -HermesHome / -InstallDir is normalized
    # in place rather than replaced, so a caller's choice is never
    # overwritten by a default. The script's own $PSBoundParameters was
    # captured at script scope ($script:BoundParams) because a function body
    # sees its own binding, not the script's. The re-derived paths land at
    # script scope so every stage below sees them.
    if ($script:BoundParams.ContainsKey('HermesHome')) {
        $script:HermesHome = ConvertTo-LongPath $script:HermesHome
    } else {
        $script:HermesHome = ConvertTo-LongPath $(
            if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }
        )
    }
    if ($script:BoundParams.ContainsKey('InstallDir')) {
        $script:InstallDir = ConvertTo-LongPath $script:InstallDir
    } else {
        $script:InstallDir = Join-Path $script:HermesHome 'hermes-agent'
    }
    $env:HERMES_HOME = $script:HermesHome
    if ($script:NormalizedProfilePaths) {
        Write-PathDiag "resolved install paths: HermesHome=$script:HermesHome InstallDir=$script:InstallDir"
    }

    # Captured here, where the values are final. The report goes to STDOUT as
    # JSON under -ShowResolvedPaths: on Windows a child's stderr does not
    # reliably reach a parent process, and the first question on any
    # "installer says a path doesn't exist" report is which paths it
    # actually resolved.
    $script:ResolvedPathReport = @{
        long_profile_root = (Get-LongProfileRoot)
        normalized        = $script:NormalizedPathRewrites
        resolver          = $script:LastResolver
        temp              = $env:TEMP
        hermes_home       = $script:HermesHome
        install_dir       = $script:InstallDir
    }
}

# Resolve the pm store root (same resolution as pm's store_root()):
# $env:HERMES_RUNTIME_DIR wins, else <HermesHome>\tools.
function Get-PmStoreRoot {
    if ($env:HERMES_RUNTIME_DIR) { return $env:HERMES_RUNTIME_DIR }
    return (Join-Path $HermesHome "tools")
}

# The MACHINE's architecture (registry PROCESSOR_ARCHITECTURE), not the
# interpreter's — an x64 powershell on Windows-on-ARM must stage arm64.
function Get-WindowsArch {
    $machineArch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -ErrorAction SilentlyContinue).PROCESSOR_ARCHITECTURE
    if ($machineArch -eq 'ARM64') { return 'arm64' }
    return 'x64'
}

# Mirror bytes must match the same pin; corruption is never a cache miss.
function Invoke-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [string]$MirrorUrl = ""
    )
    $urls = @($Url)
    if ($MirrorUrl -and $MirrorUrl -ne $Url) { $urls += $MirrorUrl }
    $httpFailure = ""
    foreach ($candidate in $urls) {
        try {
            Invoke-WebRequest -Uri $candidate -OutFile $OutFile -UseBasicParsing
        } catch {
            $errorType = $_.Exception.GetType().FullName
            if ($_.Exception -is [System.Net.WebException]) {
                if ($_.Exception.Status -in @('TrustFailure', 'SecureChannelFailure')) { throw }
            } elseif ($errorType -ne 'Microsoft.PowerShell.Commands.HttpResponseException') {
                throw
            }
            $httpFailure = $_.Exception.Message
            continue
        }
        $digest = (Get-FileHash -Path $OutFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -eq $Sha256.ToLowerInvariant()) { return }
        Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue
        # Wrong bytes = tampering or a corrupt mirror, not a routing problem.
        Fail "download digest mismatch for $candidate (expected $Sha256, got $digest)"
    }
    $tried = $urls -join " or "
    if ($httpFailure) {
        Fail "failed to download from $tried : $httpFailure"
    }
    Fail "failed to download from $tried"
}

# Provision uv for this host from the pinned pm/lock.json artifact. Stages
# the EXACT artifact pm itself uses into the same store slot
# (<store>\uv-<version>-<target>\), sha256-verified, so pm adopts the same
# bytes — no astral-latest, no irm|iex. Returns the uv.exe path.
function Get-Uv {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }  # dev shortcut; fetches nothing
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:UvPinFiles[$target]
    if (-not $pin) {
        Fail "no pinned uv artifact for $target; install uv manually: https://docs.astral.sh/uv/"
    }
    $entry = Join-Path (Get-PmStoreRoot) "uv-$($script:UvPinVersion)-$target"
    $uvExe = Join-Path $entry "uv.exe"
    if (Test-Path $uvExe) { return $uvExe }
    Log "staging pinned uv $($script:UvPinVersion) ($target) into the pm store"
    $tmpDir = Join-Path ([IO.Path]::GetTempPath()) "hermes-uv-bootstrap-$PID"
    try {
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $zipPath = Join-Path $tmpDir "uv.zip"
        Invoke-VerifiedDownload -Url $pin.Url -MirrorUrl $pin.MirrorUrl -Sha256 $pin.Sha256 -OutFile $zipPath
        $extractDir = Join-Path $tmpDir "unpacked"
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        # The zip carries uv.exe (+ uvx.exe) at the root or under one
        # versioned wrapper dir — take whichever layout arrived.
        $found = Get-ChildItem -Path $extractDir -Filter "uv.exe" -Recurse | Select-Object -First 1
        if (-not $found) { Fail "uv.exe not found in the downloaded archive" }
        New-Item -ItemType Directory -Force -Path $entry | Out-Null
        Move-Item -Path $found.FullName -Destination $uvExe -Force
        $uvx = Get-ChildItem -Path $extractDir -Filter "uvx.exe" -Recurse | Select-Object -First 1
        if ($uvx) { Move-Item -Path $uvx.FullName -Destination (Join-Path $entry "uvx.exe") -Force }
    } finally {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (& $uvExe --version 2>$null)) { Fail "pinned uv staged but does not run on this host" }
    return $uvExe
}

# Provision git for this host from the pinned pm/lock.json artifact, into
# the same store slot (<store>\git-<version>-<target>\) pm uses. Returns the
# git.exe path, or $null when no pinned artifact exists for this target.
function Get-PinnedGit {
    $existing = Get-Command git -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }  # dev shortcut; fetches nothing
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:GitPinFiles[$target]
    if (-not $pin) { return $null }
    $entry = Join-Path (Get-PmStoreRoot) "git-$($script:GitPinVersion)-$target"
    $gitExe = Join-Path $entry "cmd\git.exe"
    if (Test-Path $gitExe) { return $gitExe }
    Log "staging pinned git $($script:GitPinVersion) ($target) into the pm store"
    $tmpDir = Join-Path ([IO.Path]::GetTempPath()) "hermes-git-bootstrap-$PID"
    try {
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $tarPath = Join-Path $tmpDir "git.tar.bz2"
        Invoke-VerifiedDownload -Url $pin.Url -MirrorUrl $pin.MirrorUrl -Sha256 $pin.Sha256 -OutFile $tarPath
        $extractDir = Join-Path $tmpDir "unpacked"
        New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
        # The pinned artifact is a git-for-windows tar.bz2 (the same one pm
        # itself extracts). Windows 10+ ships bsdtar with bzip2 support.
        & tar.exe -xf $tarPath -C $extractDir
        if ($LASTEXITCODE) { Fail "failed to extract pinned git archive" }
        # Layout: Git-<ver>/cmd\git.exe — flatten the single wrapper dir.
        $inner = @(Get-ChildItem $extractDir)
        $src = $extractDir
        if ($inner.Count -eq 1 -and $inner[0].PSIsContainer) { $src = $inner[0].FullName }
        if (-not (Test-Path (Join-Path $src "cmd\git.exe"))) { Fail "git.exe not found in the downloaded archive" }
        if (Test-Path $entry) { Remove-Item -Recurse -Force $entry }
        Move-Item $src $entry
    } finally {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $gitExe
}

# Ensure a usable git for the rest of the ladder: pinned pm store slot
# first, then PATH. Returns $true on success.
function Ensure-Git {
    $g = Get-PinnedGit
    if (-not $g) { return $false }
    if ($g -ne "git") {
        # Store-staged git: expose cmd + usr\bin on this process's PATH so
        # bare `git` works for the rest of the ladder (the same dirs pm's
        # git package env() composes).
        $gitEntry = Split-Path (Split-Path $g -Parent) -Parent
        $env:Path = "$gitEntry\cmd;$gitEntry\usr\bin;$env:Path"
    }
    return $true
}

function Log([string]$msg) { Write-Host "[hermes] $msg" -ForegroundColor Blue }
function Fail([string]$msg) { Write-Host "[hermes] $msg" -ForegroundColor Red; exit 1 }

function Emit-Frame([bool]$ok, [string]$name, [bool]$skipped, [string]$reason = "") {
    $frame = [ordered]@{ ok = $ok; stage = $name; skipped = $skipped }
    if ($reason) { $frame.reason = $reason }
    $frame | ConvertTo-Json -Compress | Write-Output
}

$Stages = @(
    @{ name = "prerequisites"; title = "System prerequisites"; category = "runtime"; needs_user_input = $false },
    @{ name = "repository"; title = "Download Hermes Agent"; category = "runtime"; needs_user_input = $false },
    @{ name = "venv"; title = "Create Python environment"; category = "runtime"; needs_user_input = $false },
    @{ name = "python-deps"; title = "Install Python dependencies"; category = "runtime"; needs_user_input = $false },
    @{ name = "node-deps"; title = "Install tool dependencies"; category = "runtime"; needs_user_input = $false },
    @{ name = "path"; title = "Install hermes command"; category = "runtime"; needs_user_input = $false },
    @{ name = "config"; title = "Prepare config and skills"; category = "configuration"; needs_user_input = $false },
    @{ name = "setup"; title = "Configure API keys and settings"; category = "configuration"; needs_user_input = $true },
    @{ name = "gateway"; title = "Configure gateway service"; category = "configuration"; needs_user_input = $true }
)
if ($IncludeDesktop) {
    $Stages += @{ name = "desktop"; title = "Build desktop app"; category = "runtime"; needs_user_input = $false }
}
$Stages += @{ name = "complete"; title = "Finish install"; category = "runtime"; needs_user_input = $false }
function Stage-Prerequisites {
    if (-not (Ensure-Git)) {
        Fail "git is required. Install Git for Windows: https://git-scm.com/download/win"
    }
    Log "prerequisites ok (git)"
}

function Stage-Repository {
    if (Test-Path (Join-Path $InstallDir ".git")) {
        Log "updating $InstallDir"
        git -C $InstallDir fetch origin $Branch; if ($LASTEXITCODE) { Fail "git fetch failed" }
        git -C $InstallDir checkout $Branch; if ($LASTEXITCODE) { Fail "git checkout failed" }
        git -C $InstallDir pull --ff-only origin $Branch
        if ($LASTEXITCODE) { Log "not fast-forwardable; keeping local state" }
    } else {
        Log "cloning $RepoUrl ($Branch) into $InstallDir"
        New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) | Out-Null
        git clone --branch $Branch $RepoUrl $InstallDir; if ($LASTEXITCODE) { Fail "git clone failed" }
    }
    if ($Commit) {
        git -C $InstallDir checkout $Commit; if ($LASTEXITCODE) { Fail "could not pin commit $Commit" }
    }
}

function Stage-Venv {
    $uv = Get-Uv
    Log "creating venv"
    Push-Location $InstallDir
    & $uv venv --allow-existing venv; $code = $LASTEXITCODE
    Pop-Location
    if ($code) { Fail "uv venv failed" }
}

# Delegate the whole python+venv+tools install to pm: stage the pinned uv,
# let uv locate Python and exit before PM starts. PM provisions the interpreter,
# the venv (default extras = [all], matching `hermes update`), and the
# tool store — all hash-verified against pm/lock.json + uv.lock. install.ps1
# no longer runs `uv sync` directly; pm is the single install authority
# (the run_locked_uv_sync contract moved into pm/packages.py::uv_env).
# This tool-only bootstrap runs before PM's own dependencies exist. pm.cli
# prepares and enters its independently locked runtime before installing apps.
function Get-BootstrapPython {
    $uv = Get-Uv
    $lock = Get-Content (Join-Path $InstallDir "pm\lock.json") -Raw | ConvertFrom-Json
    $pyPin = $lock.packages.python
    $pyVersion = if ($pyPin) { ($pyPin.version -split '\+')[0] -replace '^(\d+\.\d+).*', '$1' } else { '3.14' }
    & $uv python install --no-bin $pyVersion | Out-Host
    if ($LASTEXITCODE) { Fail "bootstrap Python installation failed" }
    $bootPy = (& $uv python find --managed-python --no-project $pyVersion) -join "`n"
    if ($LASTEXITCODE -or -not $bootPy) { Fail "bootstrap Python lookup failed" }
    return $bootPy.Trim()
}

function Invoke-BootstrapPm {
    $bootPy = Get-BootstrapPython
    Log "delegating python + venv + tools to pm (hash-verified via uv.lock)"
    Push-Location $InstallDir
    try {
        # Finish bootstrap uv before PM replaces or cleans its store entry.
        & $bootPy -m pm.cli install
        if ($LASTEXITCODE) { Fail "pm install failed" }
    } finally {
        Pop-Location
    }
}

function Stage-PythonDeps {
    Invoke-BootstrapPm
}

function Stage-NodeDeps {
    Log "tool dependencies are managed by pm (hermes pm install)"
}

function Stage-Path {
    $binDir = Join-Path $HermesHome "bin"
    $bootPy = Get-BootstrapPython
    Push-Location $InstallDir
    try {
        & $bootPy -I -X utf8 hermes_cli/_launchers.py $binDir
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code) { Fail "launcher staging failed" }
    Set-LauncherUserPath $binDir
    Log "hermes command installed at $binDir"
}

function Set-LauncherUserPath([string]$binDir) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$binDir;$userPath", "User")
        Log "added $binDir to your user PATH (new shells pick it up)"
    }
}

function Stage-Config {
    foreach ($d in @("cron","sessions","logs","pairing","hooks","image_cache","audio_cache","memories","skills")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $HermesHome $d) | Out-Null
    }
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path $envFile)) {
        $example = Join-Path $InstallDir ".env.example"
        if (Test-Path $example) { Copy-Item $example $envFile } else { New-Item -ItemType File -Path $envFile | Out-Null }
    }
    $cfg = Join-Path $HermesHome "config.yaml"
    $cfgExample = Join-Path $InstallDir "cli-config.yaml.example"
    if (-not (Test-Path $cfg) -and (Test-Path $cfgExample)) { Copy-Item $cfgExample $cfg }
    Log "config prepared in $HermesHome"
}

function Stage-Setup {
    if ($NonInteractive) { return }
    & (Join-Path $InstallDir "venv\Scripts\python.exe") (Join-Path $InstallDir "hermes") setup
}

function Stage-Gateway {
    if ($NonInteractive) { return }
    & (Join-Path $InstallDir "venv\Scripts\python.exe") (Join-Path $InstallDir "hermes") gateway install
}

function Stage-Desktop {
    # Desktop support via the CURRENT runtime paths only. wake/voice extras
    # ride pm's venv sync ([all] does not include them; lazy install at
    # first use remains the fallback -- policy: Teknium, July 2026, #70509).
    # The build is `hermes desktop --build-only`, the same authority as
    # `hermes gui` and the update flow; the deleted installer-local
    # npm/Electron helpers must not reappear here.
    $venvPython = Join-Path $InstallDir "venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { Fail "venv python missing at $venvPython" }
    Push-Location $InstallDir
    try {
        Log "ensuring desktop voice/wake dependencies via pm venv sync"
        & $venvPython -c "from pm.ensure import sync_venv; sync_venv(['wake', 'voice'], explicit=True)"
        if ($LASTEXITCODE) {
            Write-Host "[hermes] voice/wake dependency sync failed (exit $LASTEXITCODE) -- they will lazy-install at first use" -ForegroundColor Yellow
        }
        Log "building desktop app (hermes desktop --build-only)"
        & $venvPython (Join-Path $InstallDir "hermes") desktop --build-only
        $code = $LASTEXITCODE
        if ($code) { Fail "desktop build failed (hermes desktop --build-only exited $code)" }

        # Probe the produced artifact -- the same candidates
        # hermes_cli/main_desktop._desktop_packaged_executable resolves
        # (verified: --build-only returns the packaged app under
        # apps/desktop/release/, not the --source dist/).
        $desktopDir = Join-Path $InstallDir "apps\desktop"
        $candidates = @(
            (Join-Path $desktopDir "release\win-unpacked\Hermes.exe"),
            (Join-Path $desktopDir "release\win-ia32-unpacked\Hermes.exe"),
            (Join-Path $desktopDir "release\win-arm64-unpacked\Hermes.exe")
        )
        $desktopExe = $null
        foreach ($cand in $candidates) {
            if (Test-Path $cand) { $desktopExe = $cand; break }
        }
        if (-not $desktopExe) {
            Fail "desktop build produced no Hermes.exe under $desktopDir
elease\*-unpacked\"
        }
        Log "Desktop ready: $desktopExe"

        # Grant ALL APPLICATION PACKAGES (S-1-15-2-2) RX on the unpacked
        # app directory: Chromium's GPU/renderer sandboxes CHECK-fail with
        # 0x80000003 without this ACE beside orphan AppContainer SIDs under
        # %LOCALAPPDATA% (electron/electron#51761, hermes-agent#38216).
        # Best-effort -- never fail an otherwise-good install over ACL.
        try {
            $appDir = Split-Path -Parent $desktopExe
            & icacls $appDir /grant "*S-1-15-2-2:(OI)(CI)(RX)" /T /C /Q | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Log "Granted AppContainer read access on $appDir"
            } else {
                Write-Host "[hermes] icacls AppContainer grant returned exit $LASTEXITCODE for $appDir" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[hermes] Could not grant AppContainer ACL: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
    New-DesktopShortcuts -TargetExe $desktopExe
}

function Stage-Complete {
    $commit = $Commit
    if (-not $commit) { $commit = git -C $InstallDir rev-parse HEAD 2>$null }
    if ($commit) {
        $marker = [ordered]@{
            schemaVersion = 1
            pinnedCommit = "$commit"
            pinnedBranch = $Branch
            completedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        $marker | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $InstallDir ".hermes-bootstrap-complete") -Encoding UTF8
        Log "bootstrap complete marker written (pinned $commit)"
    }
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
                Write-Host "[hermes] Shortcut created: $lnkPath" -ForegroundColor Green
            } catch {
                Write-Host "[hermes] Could not create shortcut $lnkPath : $($_.Exception.Message)" -ForegroundColor Yellow
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
        Write-Host "[hermes] Skipping shortcut creation: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Invoke-StageByName([string]$name) {
    switch ($name) {
        "prerequisites" { Stage-Prerequisites }
        "repository" { Stage-Repository }
        "venv" { Stage-Venv }
        "python-deps" { Stage-PythonDeps }
        "node-deps" { Stage-NodeDeps }
        "path" { Stage-Path }
        "config" { Stage-Config }
        "setup" { Stage-Setup }
        "gateway" { Stage-Gateway }
        "desktop" { Stage-Desktop }
        "complete" { Stage-Complete }
        default { Write-Error "unknown stage: $name"; exit 2 }
    }
}

# --- Dot-source guard (part 2: stop before entry) ----------------------------
# Every function definition above has loaded; now stop before any real work.
if ($script:IsDotSourced) {
    Write-Verbose "[hermes] install.ps1 was dot-sourced; definitions only, no execution"
    return
}

# The normalization prologue runs exactly once per real entry, before any
# switch is honored, so every contract below sees long-form paths.
Initialize-ResolvedPaths

if ($ProtocolVersion) { Write-Output 1; exit 0 }

if ($ShowResolvedPaths) {
    # Side-effect-free contract: by this point every mutation the prologue
    # performs (process-env 8.3 normalization) has already happened, and no
    # stage, download, or write has run. This process's env is private to it,
    # so the parent's environment is untouched. Stdout carries the resolved
    # path report; diagnostics were suppressed by Write-PathDiag.
    $script:ResolvedPathReport | ConvertTo-Json -Depth 5 -Compress | Write-Output
    exit 0
}

if ($Manifest) {
    @{ protocol_version = 1; stages = $Stages } | ConvertTo-Json -Depth 4 -Compress | Write-Output
    exit 0
}

if ($Stage) {
    # The $Stages table is the single authoritative list: it drives the
    # -Manifest output AND the no-flag ladder, so -IncludeDesktop affects
    # the real run exactly as the manifest advertises. "desktop" stays
    # directly dispatchable via -Stage even without the flag (long-standing
    # external-caller contract; the bootstrap frontend always pairs it
    # with -IncludeDesktop when it lists the stage).
    $known = @($Stages | ForEach-Object { $_.name })
    if ($known -notcontains $Stage -and $Stage -ne "desktop") {
        if ($Json) { Emit-Frame $false $Stage $false "unknown stage: $Stage" }
        else { [Console]::Error.WriteLine("unknown stage: $Stage") }
        exit 2
    }
    $stageDef = $Stages | Where-Object { $_.name -eq $Stage } | Select-Object -First 1
    $needsInput = $stageDef -and $stageDef.needs_user_input
    if ($NonInteractive -and $needsInput) {
        if ($Json) { Emit-Frame $true $Stage $true "needs user input" }
        exit 0
    }
    try {
        Invoke-StageByName $Stage
        if ($Json) { Emit-Frame $true $Stage $false }
        exit 0
    } catch {
        if ($Json) { Emit-Frame $false $Stage $false "$_" }
        exit 1
    }
}

# No -Stage: run the whole ladder — the same authoritative list the
# manifest prints, so -IncludeDesktop inserts desktop here too.
foreach ($s in $Stages) {
    Invoke-StageByName $s.name
}
