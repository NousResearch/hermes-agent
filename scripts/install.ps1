# Hermes Agent bootstrap: git checkout + venv + hermes command on PATH.
# Heavy dependencies (tool binaries, browsers, node) are pm's job after
# this: `hermes pm install`. Stage protocol kept for Hermes-Setup:
#   -Manifest             print the stage list as JSON
#   -Stage NAME [-Json]   run one stage
#   -NonInteractive       skip stages that need input
#   -IncludeDesktop       add the desktop build stage
#   -ProtocolVersion      print the stage protocol version
#   -SkipBrowser          do not install the browser tools (agent-browser +
#                         Chromium); remembered by later installs and
#                         `hermes update`, undone by
#                         `hermes pm install agent-browser`
#   -Verbose              stream every child command's output (the default
#                         with redirected output and in CI)
[CmdletBinding(PositionalBinding=$false)]
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
    # Same opt-out as install.sh --skip-browser: PM records it, so later
    # installs and `hermes update` keep the browser tools off until
    # `hermes pm install agent-browser` opts back in.
    [switch]$SkipBrowser,
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
# `iex (irm .../install.ps1)` runs this text inside the caller's session,
# where `exit` closes their PowerShell window (or ends their script). Only a
# script file (-File, `& .\install.ps1`) owns its process and may exit with a
# code. A scriptblock literal records the file its text was parsed from;
# iex'd text has none. ($MyInvocation.MyCommand.Path is the CALLER's script
# under iex, so it cannot tell the two apart.)
$script:RunAsFile = [bool]{}.File
# $PSBoundParameters inside a FUNCTION refers to the function's own binding,
# so the script's binding is captured here, once, at script scope.
$script:BoundParams = $PSBoundParameters
# Under iex, script scope is the caller's session and outlives a run; start
# each run without the previous run's answer (see Set-LauncherUserPath).
$script:BinDirOnCallerPath = $null
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
    # sees its own binding, not the script's.
    $resolvedHome = if ($script:BoundParams.ContainsKey('HermesHome')) {
        ConvertTo-LongPath $HermesHome
    } else {
        ConvertTo-LongPath $(
            if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }
        )
    }
    $resolvedDir = if ($script:BoundParams.ContainsKey('InstallDir')) {
        ConvertTo-LongPath $InstallDir
    } else {
        Join-Path $resolvedHome 'hermes-agent'
    }
    # The param() variables live in the CALLER's scope, which is the script
    # scope only under -File. Under the documented
    # `& ([scriptblock]::Create((irm ...)))` install they live in the
    # scriptblock's scope and `$script:` names the caller's session instead,
    # so `$script:HermesHome` read '' and every stage's bare $HermesHome kept
    # the un-normalized value. Scope 1 is where param() bound in every mode
    # (-File, scriptblock, dot-source).
    Set-Variable -Scope 1 -Name HermesHome -Value $resolvedHome
    Set-Variable -Scope 1 -Name InstallDir -Value $resolvedDir
    $env:HERMES_HOME = $resolvedHome

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
        hermes_home       = $resolvedHome
        install_dir       = $resolvedDir
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
            Invoke-DownloadWithProgress -Uri $candidate -OutFile $OutFile
        } catch {
            $errorType = $_.Exception.GetType().FullName
            if ($_.Exception -is [System.Net.WebException]) {
                # Windows PowerShell 5.1: DNS/connect/HTTP failures.
                if ($_.Exception.Status -in @('TrustFailure', 'SecureChannelFailure')) { throw }
            } elseif ($errorType -eq 'System.Net.Http.HttpRequestException') {
                # pwsh 7: DNS/connect failures. A TLS trust failure arrives
                # with an AuthenticationException inside and is never a routing
                # problem. (Matched by name: 5.1 may not load System.Net.Http.)
                $inner = $_.Exception.InnerException
                if ($inner -and $inner.GetType().FullName -eq 'System.Security.Authentication.AuthenticationException') { throw }
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

# Best-effort: how big is $Uri, per the server? Returns 0 when the server
# doesn't say (missing/blocked Content-Length on a redirect chain), never
# throws -- a failed probe here must fall back to an indeterminate bar, not
# abort a download that Invoke-WebRequest itself would still complete.
function Get-RemoteContentLength([string]$Uri) {
    try {
        $resp = Invoke-WebRequest -Uri $Uri -Method Head -UseBasicParsing -ErrorAction Stop
        $len = $resp.Headers['Content-Length']
        if ($len) { return [long]([string]$len -split ',' | Select-Object -First 1) }
    } catch {
        # HEAD unsupported / blocked: fall back silently.
    }
    return 0
}

# Runs the same Invoke-WebRequest call the direct version made, on a
# separate runspace, so the main thread can drive Write-Progress off
# $OutFile's size on disk while it downloads. This preserves the exact
# exception TYPE Invoke-VerifiedDownload's catch block dispatches on for
# both PS 5.1 and pwsh 7 -- EndInvoke's terminating error is unwrapped via
# .InnerException before it is rethrown, so the caller sees the same
# WebException / HttpRequestException / HttpResponseException it would
# have gotten from a direct call.
function Invoke-DownloadWithProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    if (Test-Path $OutFile) { Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue }

    $totalBytes = Get-RemoteContentLength $Uri
    $activity = "Downloading $(Split-Path -Leaf $Uri)"

    $ps = [powershell]::Create()
    $ps.AddScript({
        param($Uri, $OutFile)
        # Invoke-WebRequest's own progress bar fights ours (and is a known
        # throughput killer); we're rendering progress from outside, so
        # turn it off inside the runspace.
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
    }).AddArgument($Uri).AddArgument($OutFile) | Out-Null

    $handle = $ps.BeginInvoke()
    try {
        while (-not $handle.IsCompleted) {
            Start-Sleep -Milliseconds 200
            $haveBytes = if (Test-Path $OutFile) { (Get-Item $OutFile).Length } else { 0 }
            if ($totalBytes -gt 0) {
                $pct = [math]::Min(100, [math]::Round(($haveBytes / $totalBytes) * 100))
                $haveMb = [math]::Round($haveBytes / 1MB, 1)
                $totalMb = [math]::Round($totalBytes / 1MB, 1)
                Write-Progress -Activity $activity -Status "$haveMb MB / $totalMb MB" -PercentComplete $pct
            } else {
                # Unknown size: PercentComplete -1 draws an indeterminate/marquee
                # bar in hosts that support it, and is simply ignored elsewhere.
                $haveMb = [math]::Round($haveBytes / 1MB, 1)
                Write-Progress -Activity $activity -Status "$haveMb MB (size unknown)" -PercentComplete -1
            }
        }
        $ps.EndInvoke($handle) | Out-Null
        # Invoke-WebRequest's HTTP/DNS failures are non-terminating inside the
        # runspace: EndInvoke returns normally and the error sits in the stream.
        $streamError = if ($ps.Streams.Error.Count) { $ps.Streams.Error[0].Exception } else { $null }
    } catch {
        $inner = $_.Exception.InnerException
        if ($inner) { throw $inner } else { throw }
    } finally {
        Write-Progress -Activity $activity -Completed
        $ps.Dispose()
    }
    # Rethrown as-is (outside the unwrapping catch) so the caller classifies it
    # and tries the next candidate.
    if ($streamError) { throw $streamError }
}

# Provision uv for this host from the pinned pm/lock.json artifact. Stages
# the EXACT artifact pm itself uses into the same store slot
# (<store>\uv-<version>-<target>\), sha256-verified, so pm adopts the same
# bytes — no astral-latest, no irm|iex. Returns the uv.exe path.
function Get-Uv {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) {
        # Developer shortcut: fetches nothing, but only for a new-enough uv.
        if (Test-UvAtLeastPin $existing.Source) { return $existing.Source }
        Log "uv on PATH ($($existing.Source)) is older than the pinned $($script:UvPinVersion) or does not run; downloading our own copy"
    }
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:UvPinFiles[$target]
    if (-not $pin) {
        Fail "no pinned uv artifact for $target; install uv manually: https://docs.astral.sh/uv/"
    }
    $entry = Join-Path (Get-PmStoreRoot) "uv-$($script:UvPinVersion)-$target"
    $uvExe = Join-Path $entry "uv.exe"
    if (Test-Path $uvExe) {
        if (Test-UvAtLeastPin $uvExe) { return $uvExe }
        Log "cached pinned uv does not run; downloading our own copy"
        Remove-Item -Path $uvExe -Force
    }
    Log "downloading uv $($script:UvPinVersion) ($target)"
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
    if (-not (Test-UvAtLeastPin $uvExe)) { Fail "pinned uv staged but does not run on this host" }
    return $uvExe
}

# Provision git for this host from the pinned pm/lock.json artifact, into
# the same store slot (<store>\git-<version>-<target>\) pm uses. Returns the
# git.exe path, or $null when no pinned artifact exists for this target.
function Get-PinnedGit {
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:GitPinFiles[$target]
    if (-not $pin) { return $null }
    $entry = Join-Path (Get-PmStoreRoot) "git-$($script:GitPinVersion)-$target"
    $gitExe = Join-Path $entry "cmd\git.exe"
    if (Test-Path $gitExe) { return $gitExe }
    Log "installing git $($script:GitPinVersion) ($target)"
    $tmpDir = Join-Path ([IO.Path]::GetTempPath()) "hermes-git-bootstrap-$PID"
    try {
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $tarPath = Join-Path $tmpDir "git.tar.bz2"
        Invoke-VerifiedDownload -Url $pin.Url -MirrorUrl $pin.MirrorUrl -Sha256 $pin.Sha256 -OutFile $tarPath
        $extractDir = Join-Path $tmpDir "unpacked"
        New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
        # The pinned artifact is a git-for-windows tar.bz2 (the same one pm
        # itself extracts). Windows 10+ ships bsdtar with bzip2 support in
        # System32; a GNU tar earlier on PATH (Cygwin/MSYS) reads C:\ as a
        # remote host, so never resolve it from PATH.
        $inboxTar = Join-Path $env:SystemRoot 'System32\tar.exe'
        # MSYS ships these as symlinks into /proc. Without symlink rights (not
        # elevated, no Developer Mode) tar cannot create them and fails the
        # whole extract. Skip exactly the links pm's own extractor skips
        # (pm/store.py extract_tar git_msys) so any other failure still fails.
        # '^' anchors bsdtar's otherwise any-path-component match.
        $msysProcLinks = @('dev/fd', 'dev/stdin', 'dev/stdout', 'dev/stderr', 'etc/mtab')
        $excludes = foreach ($link in $msysProcLinks) { '--exclude'; "^$link" }
        Invoke-Native { & $inboxTar @excludes -xf $tarPath -C $extractDir }
        if ($LASTEXITCODE) { Fail "failed to extract pinned git archive" }
        # Layout: Git-<ver>/cmd\git.exe — flatten the single wrapper dir.
        $inner = @(Get-ChildItem $extractDir)
        $src = $extractDir
        if ($inner.Count -eq 1 -and $inner[0].PSIsContainer) { $src = $inner[0].FullName }
        if (-not (Test-Path (Join-Path $src "cmd\git.exe"))) { Fail "git.exe not found in the downloaded archive" }
        if (Test-Path $entry) { Remove-Item -Recurse -Force $entry }
        # Prerequisites run first, so on a fresh host the store root does not
        # exist yet; Move-Item never creates the destination's parent.
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $entry) | Out-Null
        Move-Item $src $entry
    } finally {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $gitExe
}

# Each -Stage is a new PowerShell process. Restore the pinned pm store Git
# PATH in every stage that invokes git; never inherit an unpinned system Git.
function Ensure-Git {
    $g = Get-PinnedGit
    if (-not $g) { return $false }
    # The same dirs pm's git package env() composes.
    $gitEntry = Split-Path (Split-Path $g -Parent) -Parent
    $env:Path = "$gitEntry\cmd;$gitEntry\usr\bin;$env:Path"
    return $true
}

# The pre-pm installer's line style. ASCII glyphs: Windows PowerShell 5.1
# reads a BOM-less script as the ANSI code page, so arrows would mojibake.
function Log([string]$msg) { Write-Host "-> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg) { Write-Host "[X] $msg" -ForegroundColor Red }

function Write-Banner {
    Write-Host ""
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|             * Hermes Agent Installer                    |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|  An open source AI agent by Nous Research.              |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host ""
}

# Windows PowerShell 5.1 turns a native command's stderr into an ErrorRecord
# whenever that stream is redirected inside PowerShell (`2>$null`, `2>&1`),
# and under $ErrorActionPreference = "Stop" the record terminates the script
# -- even when the tool exits 0, or the caller meant to tolerate its failure.
# Native calls run through here; the exit code stays in $LASTEXITCODE for the
# caller to judge. (The relaxed preference lives in this function's scope and
# reaches only the block invoked from it.)
function Invoke-Native([scriptblock]$Command) {
    $ErrorActionPreference = 'Continue'
    & $Command
}

# Windows PowerShell 5.1 decodes a native command's CAPTURED output with
# [Console]::OutputEncoding -- the console's OEM code page (936/gb2312 on a
# zh-CN box) unless something changed it -- while uv prints UTF-8
# unconditionally. A non-ASCII install path (C:\Users\<name-with-CJK>\...)
# therefore arrives mangled, `uv python find` hands back a path that does not
# exist, and the dependency stage (and the whole bootstrap) fails. Capture
# under a UTF-8 window so the bytes survive; the caller still judges
# $LASTEXITCODE, exactly like Invoke-Native.
function Invoke-NativeUtf8([scriptblock]$Command) {
    $previous = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
        & $Command
    } finally {
        [Console]::OutputEncoding = $previous
    }
}

# Interactive runs collapse child-process output (git, uv, pm, the builds)
# into one status line. CI, -Verbose and redirected output -- the
# Hermes-Setup -Json driver, E2E transcripts -- keep the full stream those
# readers parse.
function Test-QuietOutput {
    if ($env:CI -or $env:GITHUB_ACTIONS -or $env:HERMES_INSTALL_VERBOSE) { return $false }
    if ($VerbosePreference -ne 'SilentlyContinue') { return $false }
    try { return -not [Console]::IsOutputRedirected } catch { return $false }
}

function Write-StatusLine([string]$Text, [int]$Width) {
    $line = "  $Text"
    if ($line.Length -ge $Width) { $line = $line.Substring(0, $Width - 1) }
    Write-Host ("`r" + $line.PadRight($Width - 1)) -NoNewline -ForegroundColor DarkGray
}

# Run a native command block like Invoke-Native: $LASTEXITCODE stays the
# caller's to judge. Quiet mode shows $StatusLabel with the block's newest
# output line rewritten in place, appends everything to the install log and,
# on failure, prints the tail and the log path (-MayFail: the caller handles
# the failure, so no report). Otherwise the label is logged and the output
# streams to the host -- never to the pipeline, so a function returning a
# value can call this. The block resolves its variables through this
# function's scope, so locals here avoid the names call sites use.
function Invoke-Logged {
    param([string]$StatusLabel, [scriptblock]$NativeBlock, [switch]$MayFail)
    $logWriter = $null
    if (Test-QuietOutput) {
        $logPath = Join-Path (Join-Path $HermesHome 'logs') 'install.log'
        try {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
            $logWriter = New-Object System.IO.StreamWriter($logPath, $true, (New-Object System.Text.UTF8Encoding($false)))
        } catch {
            # An unwritable log must not stop the install: stream instead.
            $logWriter = $null
        }
    }
    if (-not $logWriter) {
        Log $StatusLabel
        Invoke-Native $NativeBlock | Out-Host
        return
    }
    $columns = 80
    try { $columns = [Math]::Max(20, $Host.UI.RawUI.WindowSize.Width) } catch { $columns = 80 }
    $recentLines = New-Object 'System.Collections.Generic.Queue[string]'
    try {
        $logWriter.WriteLine("==> $StatusLabel ($((Get-Date).ToUniversalTime().ToString('s'))Z)")
        Write-StatusLine $StatusLabel $columns
        Invoke-Native { & $NativeBlock 2>&1 } | ForEach-Object {
            $outputLine = "$_".TrimEnd("`r")
            $logWriter.WriteLine($outputLine)
            $recentLines.Enqueue($outputLine)
            if ($recentLines.Count -gt 20) { [void]$recentLines.Dequeue() }
            # git and uv redraw progress with bare CRs; show the newest.
            $newest = ($outputLine -split "`r")[-1].Trim()
            if ($newest) { Write-StatusLine "${StatusLabel}: $newest" $columns }
        }
        $exitCode = $LASTEXITCODE
    } finally {
        $logWriter.Dispose()
        Write-Host ("`r" + (' ' * ($columns - 1)) + "`r") -NoNewline
    }
    if ($exitCode -and -not $MayFail) {
        Write-Err "$StatusLabel failed (exit $exitCode). Last output:"
        foreach ($recent in $recentLines) { Write-Host "    $recent" }
        Write-Host "    full log: $logPath"
    }
    $global:LASTEXITCODE = $exitCode
}

# Does the uv at $Path run, and is it at least the pinned version? The
# bootstrap passes flags an older uv lacks (`python install --no-bin` arrived
# in 0.7), and a broken shim can exist without running.
function Test-UvAtLeastPin([string]$Path) {
    $global:LASTEXITCODE = 0
    $out = Invoke-Native { & $Path --version 2>$null }
    if ($LASTEXITCODE -or -not $out) { return $false }
    $have = ("$out".Trim() -split '\s+')[1] -replace '[^0-9.].*$', ''
    try { return ([version]$have -ge [version]$script:UvPinVersion) } catch { return $false }
}
function Fail([string]$msg) {
    # Throw, never exit: the entry points below own reporting and the exit
    # code, and the stage dispatcher's catch emits the -Json failure frame.
    throw $msg
}

function Emit-Frame([bool]$ok, [string]$name, [bool]$skipped, [string]$reason = "") {
    $frame = [ordered]@{ ok = $ok; stage = $name; skipped = $skipped }
    if ($reason) { $frame.reason = $reason }
    $frame | ConvertTo-Json -Compress | Write-Output
}

$ProductTitle = if ($IncludeDesktop) { "Install command and app + desktop" } else { "Install command and app" }
$Stages = @(
    @{ name = "prerequisites"; title = "System prerequisites"; category = "runtime"; needs_user_input = $false },
    @{ name = "repository"; title = "Download Hermes Agent"; category = "runtime"; needs_user_input = $false },
    @{ name = "venv"; title = "Create Python environment"; category = "runtime"; needs_user_input = $false },
    @{ name = "python-deps"; title = "Install Python dependencies"; category = "runtime"; needs_user_input = $false },
    @{ name = "config"; title = "Prepare config and skills"; category = "configuration"; needs_user_input = $false },
    # The shared completion tail -- the same call `hermes update` makes -- so
    # the manifest and the run cannot disagree. -IncludeDesktop selects the
    # desktop product inside this stage instead of adding a second build stage.
    @{ name = "products"; title = $ProductTitle; category = "runtime"; needs_user_input = $false },
    @{ name = "setup"; title = "Configure API keys and settings"; category = "configuration"; needs_user_input = $true },
    @{ name = "gateway"; title = "Configure gateway service"; category = "configuration"; needs_user_input = $true }
)
$Stages += @{ name = "complete"; title = "Finish install"; category = "runtime"; needs_user_input = $false }
function Stage-Prerequisites {
    if (-not (Ensure-Git)) {
        Fail "no pinned Git artifact for this Windows architecture"
    }
    Write-Ok "prerequisites ok (git)"
}

function Stage-Repository {
    # Refuse an occupied non-checkout before provisioning Git. This check
    # needs no tool download and must not overwrite a user's existing files.
    if (-not (Test-Path (Join-Path $InstallDir ".git")) -and (Test-Path -LiteralPath $InstallDir)) {
        $item = Get-Item -LiteralPath $InstallDir -Force
        $empty = $item.PSIsContainer -and -not $item.LinkType -and -not (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1)
        if (-not $empty) {
            Fail "$InstallDir exists and is not a Hermes git checkout. Move it aside, or install elsewhere with -InstallDir <path>."
        }
    }
    if (-not (Ensure-Git)) { Fail "no pinned Git artifact for this Windows architecture" }
    # An interrupted clone from an older installer can leave a .git with no
    # initial commit, where stash/checkout abort ("You do not have the initial
    # commit yet", #40998). Move it aside -- never delete it, it may hold
    # something the user wants -- and clone fresh below.
    if (Test-Path (Join-Path $InstallDir ".git")) {
        Invoke-Native { git -C $InstallDir rev-parse --verify HEAD 2>$null } | Out-Null
        if ($LASTEXITCODE) {
            $broken = "$InstallDir.broken-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
            Write-Warn "$InstallDir has no commits (interrupted clone); moving it aside to $broken"
            Move-Item -LiteralPath $InstallDir -Destination $broken
        }
    }
    if (Test-Path (Join-Path $InstallDir ".git")) {
        Log "Updating $InstallDir ($Branch)"
        # An explicit HERMES_REPO_URL names the source for reruns too, not
        # just the first clone.
        if ($env:HERMES_REPO_URL) {
            Invoke-Native { git -C $InstallDir remote set-url origin $RepoUrl }
            if ($LASTEXITCODE) { Fail "cannot point origin at $RepoUrl" }
        }
        Invoke-Logged "Fetching origin/$Branch" { git -C $InstallDir fetch origin $Branch }
        if ($LASTEXITCODE) { Fail "git fetch failed" }
        $stamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
        # Park local work BEFORE switching branches: checkout refuses a dirty
        # tree that conflicts, and the reset below would discard it. Work that
        # cannot be parked stops the install -- never overwrite it.
        if (Invoke-Native { git -C $InstallDir status --porcelain }) {
            # An interrupted update can leave unmerged index entries, where
            # stash aborts ("could not write index"). Dropping only the
            # index-level conflict state keeps the working-tree changes for
            # the stash below (#4735).
            if (Invoke-Native { git -C $InstallDir ls-files --unmerged }) {
                Write-Warn "clearing unmerged index entries from a previous conflict"
                Invoke-Native { git -C $InstallDir reset -q }
                if ($LASTEXITCODE) { Fail "cannot clear the unmerged index in $InstallDir" }
            }
            Invoke-Logged "Stashing local changes" { git -C $InstallDir stash push --include-untracked -m "hermes-install-autostash-$stamp" }
            if ($LASTEXITCODE) { Fail "could not stash local changes in $InstallDir; commit or move them aside, then rerun" }
            Write-Warn "local changes stashed as hermes-install-autostash-$stamp"
        }
        Invoke-Logged "Checking out $Branch" { git -C $InstallDir checkout $Branch }
        if ($LASTEXITCODE) { Fail "git checkout failed" }
        # --no-stat: across a large gap (v2026.7.1 -> today is ~27k lines) the
        # diffstat arrives as one burst. Hermes-Setup.exe forwards every line
        # to its window as a separate event; the burst overflows the Windows
        # posted-message queue (10k), events drop, and the installer's Launch
        # button can then hang on "Launching" forever.
        Invoke-Logged -MayFail "Fast-forwarding to origin/$Branch" { git -C $InstallDir merge --ff-only --no-stat "origin/$Branch" }
        if ($LASTEXITCODE) {
            # A release cut off the main line, a force-pushed remote, or the
            # user's own commits cannot fast-forward. Every stage below reads
            # files only the new tree has (pm/), so an install left on the old
            # tree cannot finish -- match the remote the way `hermes update`
            # does, after parking the old tip. Mirrors scripts/install.sh.
            # Keep commits absent from origin in the updater's rescue namespace.
            $droppedText = (Invoke-Native { git -C $InstallDir rev-list --count "origin/$Branch..HEAD" 2>$null })
            if ($LASTEXITCODE) { Fail "cannot count commits before reset" }
            [long]$dropped = 0
            if (-not [long]::TryParse("$droppedText".Trim(), [ref]$dropped)) { Fail "cannot count commits before reset" }
            if ($dropped -gt 0) {
                Invoke-Native { git -C $InstallDir merge-base HEAD "origin/$Branch" 2>$null } | Out-Null
                $rescueKind = if ($LASTEXITCODE -eq 0) { 'diverged' } else { 'orphan' }
                $prior = (Invoke-Native { git -C $InstallDir rev-parse --short=12 HEAD 2>$null })
                if ($LASTEXITCODE -or -not $prior) { Fail "cannot identify commits before reset" }
                $rescue = "refs/hermes-update-backups/$rescueKind-$Branch-$stamp-$prior"
                Invoke-Native { git -C $InstallDir update-ref $rescue HEAD 2>$null }
                if ($LASTEXITCODE) { Fail "cannot back up $dropped local commit(s); refusing to reset" }
                Write-Warn "$dropped commit(s) not on origin/$Branch backed up to $rescue"
                Log "List them with: git -C `"$InstallDir`" log origin/$Branch..$rescue"
            }
            Invoke-Logged "Resetting to origin/$Branch" { git -C $InstallDir reset --hard "origin/$Branch" }
            if ($LASTEXITCODE) { Fail "git reset failed" }
            Write-Warn "not fast-forwardable; reset to origin/$Branch"
        }
    } else {
        # Moving a clone onto an existing directory would nest it. The
        # preflight above already refused nonempty or linked destinations.
        if (Test-Path -LiteralPath $InstallDir) {
            Remove-Item -LiteralPath $InstallDir -Force
        }
        $parent = Split-Path $InstallDir
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        # Clone into a sibling staging dir and publish only a complete,
        # materialized checkout: a clone that dies half-way must not leave a
        # .git behind that the next rerun would try to update.
        $staged = Join-Path $parent ".hermes-clone-$PID-$(Get-Random)"
        $tree = Join-Path $staged "tree"
        New-Item -ItemType Directory -Force -Path $staged | Out-Null
        # Phase lines ("Receiving objects: 42%") feed the status line; git
        # prints none to a pipe unless asked.
        $progress = @()
        if (Test-QuietOutput) { $progress = @('--progress') }
        try {
            $cloned = $false
            foreach ($attempt in 1..3) {
                # Treeless: every commit and release tag (runtime identity is the
                # nearest reachable release; -Commit pins and branch switches
                # still resolve), trees and blobs fetched on demand, so the
                # download stays close to a --depth 1 clone.
                $cloneLabel = "Cloning $RepoUrl ($Branch) into $InstallDir"
                if ($attempt -gt 1) { $cloneLabel += " (attempt $attempt of 3)" }
                Invoke-Logged $cloneLabel { git clone @progress --filter=tree:0 --branch $Branch $RepoUrl $tree }
                if (-not $LASTEXITCODE) { $cloned = $true; break }
                Remove-Item -LiteralPath $tree -Recurse -Force -ErrorAction SilentlyContinue
                if ($attempt -lt 3) { Start-Sleep -Seconds ($attempt * 5) }
            }
            if (-not $cloned) {
                # The checkout step is where throttled downloads die: clone the
                # graph alone, then retry materializing the tree separately.
                Write-Warn "direct clone failed; trying deferred checkout"
                Invoke-Logged "Cloning history" { git clone @progress --filter=tree:0 --no-checkout --branch $Branch $RepoUrl $tree }
                if (-not $LASTEXITCODE) {
                    foreach ($attempt in 1..2) {
                        Invoke-Logged "Checking out files (attempt $attempt of 2)" { git -C $tree reset --hard HEAD }
                        if (-not $LASTEXITCODE) { $cloned = $true; break }
                        if ($attempt -lt 2) { Start-Sleep -Seconds 5 }
                    }
                }
            }
            if (-not $cloned) { Fail "git clone failed; no checkout published" }
            Move-Item -LiteralPath $tree -Destination $InstallDir
            Write-Ok "Hermes Agent cloned"
        } finally {
            Remove-Item -LiteralPath $staged -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ($Commit) {
        # A pin must come from the branch being installed: the complete marker
        # records both, and a commit off that branch would make the next plain
        # rerun "update" onto a different line.
        Invoke-Native { git -C $InstallDir merge-base --is-ancestor $Commit "origin/$Branch" 2>$null }
        if ($LASTEXITCODE) { Fail "commit $Commit is not on branch $Branch" }
        Invoke-Logged "Pinning $Commit" { git -C $InstallDir checkout $Commit }
        if ($LASTEXITCODE) { Fail "could not pin commit $Commit" }
    }
}

function Stage-Venv {
    # Keep the installer stage protocol; PM alone creates dependency environments.
    Get-BootstrapPython | Out-Null
    Write-Ok "bootstrap Python ready; PM prepares the dependency environment"
}

# Delegate the whole python+venv+tools install to pm: stage the pinned uv,
# let uv locate Python and exit before PM starts. PM provisions the interpreter,
# the venv (default extras = [all], matching `hermes update`), and the
# tool store — all hash-verified against pm/lock.json + uv.lock. install.ps1
# no longer runs `uv sync` directly; pm is the single install authority
# (the run_locked_uv_sync contract moved into pm/environment.py).
# This tool-only bootstrap runs before PM's own dependencies exist. pm.cli
# prepares and enters its independently locked runtime before installing apps.
function Get-BootstrapPython {
    # The full ladder runs every stage in one process and four of them need
    # this interpreter; resolve uv and Python once per process.
    if ($script:BootstrapPython) { return $script:BootstrapPython }
    $uv = Get-Uv
    $lock = Get-Content (Join-Path $InstallDir "pm\lock.json") -Raw | ConvertFrom-Json
    $pyPin = $lock.packages.python
    $pyVersion = if ($pyPin) { ($pyPin.version -split '\+')[0] -replace '^(\d+\.\d+).*', '$1' } else { '3.14' }
    # A bare version lets uv pick emulated x86_64 on Windows-on-ARM.
    $pyArch = if ((Get-WindowsArch) -eq 'arm64') { 'aarch64' } else { 'x86_64' }
    $pyRequest = "cpython-$pyVersion-windows-$pyArch-none"
    $bootPy = (Invoke-NativeUtf8 { & $uv python find --managed-python --no-project $pyRequest 2>$null }) -join "`n"
    if ($LASTEXITCODE -or -not $bootPy) {
        Invoke-Logged "Downloading Python $pyVersion" { & $uv python install --no-bin --no-registry $pyRequest }
        if ($LASTEXITCODE) { Fail "bootstrap Python installation failed" }
        $bootPy = (Invoke-NativeUtf8 { & $uv python find --managed-python --no-project $pyRequest }) -join "`n"
    }
    if ($LASTEXITCODE -or -not $bootPy) { Fail "bootstrap Python lookup failed" }
    $script:BootstrapPython = $bootPy.Trim()
    return $script:BootstrapPython
}

function Invoke-BootstrapPm {
    $bootPy = Get-BootstrapPython
    Push-Location $InstallDir
    try {
        # Finish bootstrap uv before PM replaces or cleans its store entry.
        # Bare $SkipBrowser, like $InstallDir: under iex/scriptblock entry the
        # param() binding is not in $script: scope (see Initialize-ResolvedPaths).
        $pmArgs = @('install')
        if ($SkipBrowser) { $pmArgs += @('--without', 'agent-browser') }
        Invoke-Logged "Installing dependencies (hash-verified via uv.lock)" { & $bootPy -m pm.cli @pmArgs }
        if ($LASTEXITCODE) { Fail "dependency install failed" }
    } finally {
        Pop-Location
    }
    Write-Ok "dependencies installed"
}

function Stage-PythonDeps {
    Invoke-BootstrapPm
}

function Invoke-SourceCompletion([bool]$Desktop) {
    # The whole tail in one place, by calling the completion an update calls:
    # publish the commands, build the products (tui/web, plus the desktop app
    # when asked), then run the post-build maintenance that syncs bundled
    # skills and migrates config. Node, browsers and the frontend build tools
    # arrive through pm as the build asks for them; the bootstrap interpreter
    # itself only re-enters the tree on PM's selected Python.
    $bootPy = Get-BootstrapPython
    $completionArgs = @('-I', '-B', '-X', 'utf8', 'hermes_cli/source_completion.py', '--source', $InstallDir)
    if ($Desktop) { $completionArgs += '--desktop' }
    Push-Location $InstallDir
    try {
        Invoke-Logged "Building the hermes command and apps" { & $bootPy @completionArgs }
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code) { Fail "app products or command publication failed (exit $code)" }
    Write-Ok "app products and hermes command ready"
}

function Publish-UserCommand {
    # PATH exposure stays installer-owned on Windows: expose_cli() answers
    # "windows-installer-owned" rather than creating the user-facing command,
    # so the install-scoped launchers the completion publishes are not the ones
    # the user's PATH points at.
    $binDir = Join-Path $HermesHome "bin"
    $bootPy = Get-BootstrapPython
    Push-Location $InstallDir
    try {
        Invoke-Logged "Publishing the hermes command" { & $bootPy -I -X utf8 hermes_cli/_launchers.py $binDir }
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code) { Fail "launcher staging failed" }
    Set-LauncherUserPath $binDir
    Write-Ok "hermes command installed at $binDir"
}

function Test-DesktopProductPresent {
    # Does this checkout already carry a built desktop app? A plain repair or
    # upgrade rerun on a desktop install must REBUILD it rather than leave a
    # bundle built by the previous code: the app is part of that install and its
    # artifacts live inside the tree, so an update makes them stale, not gone.
    $release = Join-Path $InstallDir "apps/desktop/release"
    foreach ($candidate in @("win-unpacked", "linux-unpacked", "mac", "mac-arm64")) {
        if (Test-Path (Join-Path $release $candidate)) { return $true }
    }
    return $false
}

function Stage-Products {
    $desktop = [bool]$IncludeDesktop -or [bool](Test-DesktopProductPresent)
    Invoke-SourceCompletion $desktop
    Publish-UserCommand
    if ($desktop) { Confirm-DesktopArtifact }
}

function Set-LauncherUserPath([string]$binDir) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$binDir;$userPath", "User")
        Write-Ok "added $binDir to your user PATH (new shells pick it up)"
    }
    # The registry write only reaches shells started later. $env:Path is
    # process-wide, so prepending it here makes `hermes` resolve in the
    # caller's own window whenever this code runs in the caller's process
    # (`irm | iex`, `& .\install.ps1`); a -File child just discards it.
    # Recorded before the first prepend only (the -IncludeDesktop ladder
    # publishes twice): it is what the caller's shell inherited.
    $sessionEntries = @($env:Path -split ';' | ForEach-Object { $_.TrimEnd('\') })
    $onPath = $sessionEntries -contains $binDir.TrimEnd('\')
    if ($null -eq $script:BinDirOnCallerPath) { $script:BinDirOnCallerPath = $onPath }
    if (-not $onPath) { $env:Path = "$binDir;$env:Path" }
}

function Write-PathReloadHint {
    # A script file may be a separate powershell.exe (-File), whose $env:Path
    # dies with it; the parent keeps the PATH it started with until reloaded.
    # iex'd text always runs in the caller's process, where the prepend in
    # Set-LauncherUserPath already made `hermes` resolvable.
    if (-not $script:RunAsFile -or $script:BinDirOnCallerPath -ne $false) { return }
    Log 'Restart your terminal to use hermes, or run: $env:Path = [Environment]::GetEnvironmentVariable(''Path'',''User'') + '';'' + [Environment]::GetEnvironmentVariable(''Path'',''Machine'')'
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
    Write-Ok "config prepared in $HermesHome"
}

function Invoke-InstalledHermes([string[]]$CommandArgs) {
    # Load the helper from its text, not its path. Under `irm | iex` this
    # installer runs as a string that execution policy never checks, but
    # dot-sourcing a .ps1 from disk is a file load. The default Restricted
    # policy (Windows Sandbox, fresh machines) refuses that load.
    $runtimeHelper = Join-Path $InstallDir 'scripts/desktop-update/runtime.ps1'
    . ([ScriptBlock]::Create([IO.File]::ReadAllText($runtimeHelper)))
    # Not `$command`: Invoke-Native's `$Command` parameter shadows it
    # (names are case-insensitive) and the block would invoke itself.
    $runtimeCommand = @(Get-HermesRuntimeCommand -InstallRoot $InstallDir)
    $runtimeArgs = @($runtimeCommand | Select-Object -Skip 1) + $CommandArgs
    Invoke-Native { & $runtimeCommand[0] @runtimeArgs }
    if ($LASTEXITCODE) { Fail "hermes $($CommandArgs -join ' ') failed (exit $LASTEXITCODE)" }
}

function Stage-Setup {
    if ($NonInteractive) { return }
    Invoke-InstalledHermes @('setup')
}

function Stage-Gateway {
    if ($NonInteractive) { return }
    # Setup installs the service when it handles the gateway; ask only if it did not.
    Invoke-InstalledHermes @('gateway', 'install', '--if-missing')
}

function Stage-Desktop {
    # External-caller contract: -Stage desktop stays dispatchable on its own
    # (see Invoke-StageByName). The work is the same completion call with the
    # desktop product selected. Voice and wake extras are not synced here: pm
    # lazy-installs them at first use (policy: Teknium, July 2026, #70509).
    Invoke-SourceCompletion $true
    Publish-UserCommand
    Confirm-DesktopArtifact
}

function Confirm-DesktopArtifact {
    # Probe the packaged artifact the completion just built -- the same
    # candidates hermes_cli/main_desktop._desktop_packaged_executable resolves.
    Push-Location $InstallDir
    try {
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
            Fail "desktop build produced no Hermes.exe under $desktopDir\release\*-unpacked"
        }
        Write-Ok "Desktop ready: $desktopExe"

        # Grant ALL APPLICATION PACKAGES (S-1-15-2-2) RX on the unpacked
        # app directory: Chromium's GPU/renderer sandboxes CHECK-fail with
        # 0x80000003 without this ACE beside orphan AppContainer SIDs under
        # %LOCALAPPDATA% (electron/electron#51761, hermes-agent#38216).
        # Best-effort -- never fail an otherwise-good install over ACL.
        try {
            $appDir = Split-Path -Parent $desktopExe
            Invoke-Native { & icacls $appDir /grant "*S-1-15-2-2:(OI)(CI)(RX)" /T /C /Q } | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Log "Granted AppContainer read access on $appDir"
            } else {
                Write-Warn "icacls AppContainer grant returned exit $LASTEXITCODE for $appDir"
            }
        } catch {
            Write-Warn "Could not grant AppContainer ACL: $($_.Exception.Message)"
        }
    } finally {
        Pop-Location
    }
    New-DesktopShortcuts -TargetExe $desktopExe
}

function Stage-Complete {
    $commit = $Commit
    if (-not $commit) {
        if (-not (Ensure-Git)) { Fail "no pinned Git artifact for this Windows architecture" }
        $commit = Invoke-Native { git -C $InstallDir rev-parse HEAD 2>$null }
    }
    if ($commit) {
        $marker = [ordered]@{
            schemaVersion = 1
            pinnedCommit = "$commit"
            pinnedBranch = $Branch
            completedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        $marker | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $InstallDir ".hermes-bootstrap-complete") -Encoding UTF8
        Write-Ok "Hermes Agent install complete (pinned $commit). Run: hermes"
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
                Write-Ok "Shortcut created: $lnkPath"
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
            Invoke-Native { & ie4uinit.exe -show 2>$null }
        } catch {
            # ie4uinit may be absent/renamed on some SKUs -- ignore.
        }
    } catch {
        Write-Warn "Skipping shortcut creation: $($_.Exception.Message)"
    }
}

function Invoke-StageByName([string]$name) {
    switch ($name) {
        "prerequisites" { Stage-Prerequisites }
        "repository" { Stage-Repository }
        "venv" { Stage-Venv }
        "python-deps" { Stage-PythonDeps }
        "products" { Stage-Products }
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

# Keep uv from discovering uv.toml / pyproject.toml config from whatever
# directory or user profile the installer runs under (mirrors install.sh).
$env:UV_NO_CONFIG = "1"
# Children that collapse their own output (windows-build-deps.ps1 under pm,
# when its stdout is still the console) stream too once -Verbose asked for it.
if ($VerbosePreference -ne 'SilentlyContinue') { $env:HERMES_INSTALL_VERBOSE = "1" }

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
    # directly dispatchable via -Stage even though it is never listed
    # (long-standing external-caller contract).
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
        Write-Err "$_"
        if ($Json) { Emit-Frame $false $Stage $false "$_" }
        exit 1
    }
}

# No -Stage: run the whole ladder — the same authoritative list the
# manifest prints, so -IncludeDesktop inserts desktop here too.
try {
    Write-Banner
    foreach ($s in $Stages) {
        Invoke-StageByName $s.name
    }
    Write-PathReloadHint
} catch {
    Write-Err "$_"
    if ($script:RunAsFile) { exit 1 }
    # Under iex: report failure without closing the user's window.
    $global:LASTEXITCODE = 1
}
