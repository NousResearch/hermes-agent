# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---
function Expand-HermesHomeValue {
    # The HOST os.path ladder, which is what pm runs: ntpath on Windows,
    # posixpath elsewhere. $Sep selects which ladder, so the Windows branch is
    # testable on any host. Quoted spans and the $/% escape rules follow the
    # platform's own _varpattern; an unset name stays literal.
    param(
        [string]$Value,
        [string]$Sep = [string][System.IO.Path]::DirectorySeparatorChar
    )
    $isWin = $Sep -eq '\'
    $envDict = [System.Environment]::GetEnvironmentVariables()
    $pattern = if ($isWin) { '''[^'']*''?|%(%|[^%]*%?)|\$(\$|[-\w]+|\{[^}]*\}?)' }
               else { '\$(\w+|\{[^}]*\}?)' }
    # Windows: group 1 holds the %VAR% form, group 2 the $VAR form. POSIX has
    # no %VAR%, so its single group is the $VAR form.
    $envGroup = if ($isWin) { 2 } else { 1 }
    $Value = [regex]::Replace($Value, $pattern, {
        param($m)
        if ($isWin -and $m.Groups[1].Success) {
            $name = $m.Groups[1].Value
            if ($name -eq '%') { return '%' }
            if (-not $name.EndsWith('%')) { return $m.Value }
            $name = $name.Substring(0, $name.Length - 1)
        } elseif ($m.Groups[$envGroup].Success) {
            $name = $m.Groups[$envGroup].Value
            if ($name -eq '$') { return '$' }
            if ($name.StartsWith('{')) {
                if (-not $name.EndsWith('}')) { return $m.Value }
                $name = $name.Substring(1, $name.Length - 2)
            }
        } else {
            return $m.Value
        }
        $v = [Environment]::GetEnvironmentVariable($name)
        if ($null -ne $v) { return $v }
        if ($envDict.Contains($name)) { return '' }
        return $m.Value
    })
    if ($Value.StartsWith('~')) {
        $seps = if ($isWin) { [char[]]@('/', '\') } else { [char[]]@('/') }
        $idx = $Value.IndexOfAny($seps, 1)
        if ($idx -lt 0) { $idx = $Value.Length }
        # Presence, not truthiness: ntpath tests membership in os.environ, so a
        # present-but-empty USERPROFILE still substitutes. On Windows the ladder
        # stops at the HOMEDRIVE+HOMEPATH pair — ntpath never consults HOME.
        if ($isWin) {
            if ($envDict.Contains('USERPROFILE')) { $userHome = [string]$envDict['USERPROFILE'] }
            elseif ($envDict.Contains('HOMEPATH')) {
                # ntpath joins the pair; a rooted HOMEPATH replaces the drive's
                # own separators rather than stacking a second one.
                $drive = [string]$envDict['HOMEDRIVE']
                $homePart = [string]$envDict['HOMEPATH']
                if ($homePart -match '^[\\/]') { $userHome = $drive.TrimEnd('\', '/') + $homePart }
                elseif ($homePart -match '^[A-Za-z]:') { $userHome = $homePart }
                else { $userHome = $drive + $homePart }
            } else { $userHome = $null }
        } else {
            # posixpath: HOME when SET (even empty), else the account database;
            # the answer is rstripped and empty folds to the separator.
            if ($envDict.Contains('HOME')) { $userHome = ([string]$envDict['HOME']).TrimEnd('/') }
            else { $userHome = ([string][System.Environment]::GetFolderPath('UserProfile')).TrimEnd('/') }
        }
        if ($null -ne $userHome) {
            if ($idx -ne 1) {
                # ntpath guesses another account's profile dir only when the
                # current account names it; posixpath needs pwd (unreachable
                # here), so a named user stays literal on POSIX.
                if (-not $isWin) { return $Value }
                $slash = $userHome.LastIndexOfAny([char[]]@('\', '/'))
                $leaf = if ($slash -ge 0) { $userHome.Substring($slash + 1) } else { $userHome }
                $currentUser = if ($envDict.Contains('USERNAME')) { [string]$envDict['USERNAME'] } else { $null }
                $targetUser = $Value.Substring(1, $idx - 1)
                if ($targetUser -cne $currentUser) {
                    if ($currentUser -cne $leaf) { return $Value }
                    $parent = if ($slash -ge 0) { $userHome.Substring(0, $slash) } else { '' }
                    $userHome = if ($parent) { $parent + $Sep + $targetUser } else { $targetUser }
                }
            }
            $Value = $userHome + $Value.Substring($idx)
            if (-not $Value -and -not $isWin) { return $Sep }
        }
    }
    return $Value
}
function _HermesNormPath {
    # Lexical Path normalization: drop "." and repeated separators; at most TWO
    # leading separators are kept (POSIX form / Windows UNC). ".." stays. A
    # drive-relative form ("C:rest" with NO separator after the colon) keeps
    # that shape: Windows anchors it at the drive's own current directory, and
    # rewriting it to "C:\rest" stages the store on the wrong directory.
    param([string]$Value, [string]$Sep)
    $isWin = $Sep -eq '\'
    $prefix = ''
    $driveRelative = $false
    $body = $Value
    if ($isWin) {
        if ($body -match '^([A-Za-z]:)([\\/]?)(.*)$') {
            $prefix = $matches[1]
            $driveRelative = -not [bool]$matches[2]
            $body = $matches[3]
        }
        $stripped = $body.TrimStart('\', '/')
        $leading = $body.Length - $stripped.Length
        if ($leading -ge 1 -and -not $prefix) {
            if ($leading -eq 2) { $prefix = '\\' } else { $prefix = '\' }
        }
        $body = $stripped
    } else {
        $stripped = $body.TrimStart('/')
        $leading = $body.Length - $stripped.Length
        if ($leading -ge 1) {
            if ($leading -eq 2) { $prefix = '//' } else { $prefix = '/' }
        }
        $body = $stripped
    }
    # POSIX does not treat a backslash as a separator -- only Windows does.
    $splitPattern = if ($isWin) { '[\\/]' } else { '/' }
    $parts = @($body -split $splitPattern | Where-Object { $_ -and $_ -ne '.' })
    if (-not $parts -or $parts.Count -eq 0) {
        # A bare drive-rooted input ('C:\') keeps its root separator; a bare
        # drive ('C:') is drive-relative and must not gain one.
        if (-not $driveRelative -and $prefix -match '^[A-Za-z]:$') { return $prefix + $Sep }
        return $prefix
    }
    $joined = $parts -join $Sep
    if ($prefix -and -not $prefix.EndsWith('\') -and -not $prefix.EndsWith('/')) {
        # A bare drive prefix joins the FIRST component without a separator for
        # a drive-relative input ('C:foo'), with one for a drive-rooted input.
        if ($driveRelative) { return $prefix + $joined }
        return $prefix + $Sep + $joined
    }
    return $prefix + $joined
}
function _HermesResolvePath {
    # Path.resolve(strict=False): follow every reparse point along the chain (not
    # only the longest existing prefix), keep missing components, pop ".." off the
    # resolved prefix (never the anchor).
    param([string]$Value, [string]$Sep, [int]$Depth = 0)
    $work = ($Value -replace '/', $Sep)
    # .NET calls a drive-relative form ("C:rest") unrooted, but prepending the
    # process cwd would turn it into "C:\cwd\..." — its real anchor is the
    # drive's own current directory. Keep it as its own anchor instead.
    $driveRelative = $Sep -eq '\' -and $work -match '^[A-Za-z]:[^\\/]'
    if (-not $driveRelative -and -not [System.IO.Path]::IsPathRooted($work)) {
        $work = (Join-Path (Get-Location).Path $work) -replace '/', $Sep
    }
    # Anchor: drive root (C:\), drive-relative (C:), UNC share (\\server\share)
    # or bare root. A share exists as a FORM even when unreachable, so ".."
    # must not rise above it.
    $anchor = ''
    $body = ''
    if ($Sep -eq '\') {
        if ($work -match '^([A-Za-z]:)[\\/](.*)$') {
            $anchor = $matches[1] + $Sep
            $body = $matches[2]
        } elseif ($work -match '^([A-Za-z]:)(.*)$') {
            # Drive-relative: the anchor carries NO separator after the colon.
            $anchor = $matches[1]
            $body = $matches[2]
        } elseif ($work -match '^\\\\+([^\\/]+)[\\/]+([^\\/]+)[\\/]*(.*)$') {
            $anchor = '\\' + $matches[1] + $Sep + $matches[2]
            $body = $matches[3]
        } elseif ($work -match '^[\\/]+(.*)$') {
            $anchor = $Sep
            $body = $matches[1]
        }
    } else {
        # Path.resolve("//foo") is "/foo" on POSIX: leading separators collapse.
        $anchor = '/'
        $body = $work.TrimStart('/')
    }
    $resolved = $anchor
    $rest = $body
    while ($rest) {
        $idx = $rest.IndexOf($Sep)
        if ($idx -lt 0) { $comp = $rest; $rest = '' }
        else { $comp = $rest.Substring(0, $idx); $rest = $rest.Substring($idx + 1) }
        if (-not $comp -or $comp -eq '.') { continue }
        if ($comp -eq '..') {
            # A one-element pop empties the chain (0..-1 would keep the element).
            if ($resolved.Length -gt $anchor.Length) {
                $cut = $resolved.LastIndexOf($Sep)
                if ($cut -lt $anchor.Length) { $cut = $anchor.Length }
                $resolved = $resolved.Substring(0, $cut)
            }
            continue
        }
        # The drive-relative anchor ('C:') joins its first component with no
        # separator; every later join is ordinary.
        if ($resolved.EndsWith($Sep)) { $cand = $resolved + $comp }
        elseif ($driveRelative -and $resolved -eq $anchor) { $cand = $resolved + $comp }
        else { $cand = $resolved + $Sep + $comp }
        $item = Get-Item -LiteralPath $cand -Force -ErrorAction SilentlyContinue
        if ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            # Follow the link (capped like realpath).
            if ($Depth -ge 40) { return _HermesNormPath $Value $Sep }
            $target = [string](@($item.Target)[0])
            if (-not $target) { $resolved = $cand; continue }
            $target = $target -replace '/', $Sep
            if ([System.IO.Path]::IsPathRooted($target)) {
                return _HermesResolvePath (($target + $Sep + $rest)) $Sep ($Depth + 1)
            }
            # Same join rule as the walk: a drive-relative anchor ('C:') adds
            # no separator, so the relative link stays drive-relative.
            if ($resolved.EndsWith($Sep) -or ($driveRelative -and $resolved -eq $anchor)) {
                $linkBase = $resolved
            } else {
                $linkBase = $resolved + $Sep
            }
            return _HermesResolvePath (($linkBase + $target + $Sep + $rest)) $Sep ($Depth + 1)
        }
        $resolved = $cand
    }
    return $resolved
}
function Get-HermesDefaultHome {
    # _get_platform_default_hermes_home(): LOCALAPPDATA first, else the platform
    # default -- AppData\Local on Windows, ~/.hermes on POSIX; the suffix is
    # appended LITERALLY. This assembled default must NOT go through
    # Expand-HermesHomeValue: a %/$ variable in the suffix would expand there
    # while pm treats the suffix as a fixed string.
    $suffix = if ($env:HERMES_DATA_DIR_SUFFIX) { $env:HERMES_DATA_DIR_SUFFIX } else { '' }
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA.Trim() } else { '' }
    $accountHome = if ($HOME) { [string]$HOME } else { '~' }
    $sep = [string][System.IO.Path]::DirectorySeparatorChar
    if ($localAppData) { return (Join-Path $localAppData "hermes$suffix") }
    if ($sep -eq '\') { return (Join-Path $accountHome "AppData/Local/hermes$suffix") }
    return (Join-Path $accountHome ".hermes$suffix")
}
function Get-HermesRoot {
    $sep = [string][System.IO.Path]::DirectorySeparatorChar
    # The default home carries the literal suffix (Get-HermesDefaultHome); only
    # the explicit HERMES_HOME below is expanded, matching _expand_hermes_home.
    $default = Get-HermesDefaultHome
    # Containment is decided on RESOLVED paths but the lexical form (keeping "..")
    # is returned, mirroring get_default_hermes_root().
    $defaultLex = _HermesNormPath $default $sep
    if ($defaultLex -notmatch '^[A-Za-z]:[\\/]$') { $defaultLex = $defaultLex.TrimEnd('/', '\') }
    $defaultRes = _HermesResolvePath $defaultLex $sep
    # pm strips the RAW env home before expanding it; whitespace-only folds to default.
    $rawHome = if ($env:HERMES_HOME) { $env:HERMES_HOME.Trim() } else { '' }
    $rootLex = if ($rawHome) { Expand-HermesHomeValue $rawHome } else { $defaultLex }
    $rootLex = _HermesNormPath $rootLex $sep
    # A bare drive root keeps its separator ('C:\'); every other path loses a
    # trailing one so a later join does not double up.
    if ($rootLex -notmatch '^[A-Za-z]:[\\/]$') { $rootLex = $rootLex.TrimEnd('/', '\') }
    # '.' like Path('.'): an empty root would land the store on the drive root.
    if (-not $rootLex) { $rootLex = '.' }
    $rootRes = _HermesResolvePath $rootLex $sep
    # Mirror pathlib's comparison: Windows accepts either separator and folds case,
    # POSIX does neither (so the fold flips by platform, as resolve()+normcase does).
    $root = $rootLex
    $default = $defaultLex
    if ($sep -eq '\') {
        $rootRes = $rootRes.Replace('/', '\')
        $defaultRes = $defaultRes.Replace('/', '\')
    }
    $cmp = if ($sep -eq '\') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if ($rootRes.Equals($defaultRes, $cmp) -or $rootRes.StartsWith($defaultRes + $sep, $cmp)) { return $default }
    $parent = Split-Path -Parent $root
    # `profiles` stays -ceq (hermes_constants compares the str name case-sensitively);
    # a lexical leaf, since Split-Path -Leaf resolves a trailing "..".
    $pleaf = ''
    if ($parent) {
        $pidx = $parent.LastIndexOf($sep)
        $pleaf = if ($pidx -ge 0) { $parent.Substring($pidx + 1) } else { $parent }
    }
    if ($pleaf -ceq 'profiles') {
        $profileRoot = Split-Path -Parent $parent
        if (-not $profileRoot) {
            $profileRoot = if ($root.StartsWith($sep)) { $sep } else { '.' }
        }
        return $profileRoot
    }
    return $root
}
function Test-HermesFullyQualifiedPath {
    # Whether the path needs no cwd to name one directory. [System.IO.Path]
    # ::IsPathFullyQualified is .NET Core-only (missing from Windows
    # PowerShell 5.1), and IsPathRooted is NOT its answer: a drive-relative
    # ``C:rest`` and a root-relative ``\rest`` are both rooted yet still
    # anchor at a current directory, so both must be bound first.
    param([string]$Value)
    if (-not [System.IO.Path]::IsPathRooted($Value)) { return $false }
    if ([string][System.IO.Path]::DirectorySeparatorChar -ne '\') { return $true }
    if ($Value -match '^[A-Za-z]:(?![\\/])') { return $false }
    if ($Value -match '^[\\/](?![\\/])') { return $false }
    return $true
}
# --- END canonical store-root resolver ---
