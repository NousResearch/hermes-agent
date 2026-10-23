# marker.ps1 -- the update marker (contract C1 v2, hand-off protocol 2 / A7)
# for windows.ps1, which dot-sources it as its first act. Pure PowerShell +
# CIM on purpose: the claim runs before the first Add-Type (CLM, #66753).
#
# Body (tests/fixtures/update_marker_corpus.json is authoritative):
#   line 1  owner pid, ASCII digits, fits u32 (else MALFORMED); pid 0 = dead
#   line 2  started_at unix seconds, ASCII digits (else MALFORMED)
#   line 3  ct:<digits[.digits]> owner creation time; anything else = v1
#   line 4+ 'delegate:<pid> ct:<ct>' (first well-formed wins) and
#           'run:<[A-Za-z0-9._-]{1,128}>' (first wins); others are ignored
# Per line: one leading BOM (line 1), one trailing CR, then surrounding
# spaces/tabs are dropped; whitespace inside a value makes that line malformed.
# Writers are canonical: "<pid>\n<started>\nct:<ct>\n[delegate:..\n][run:..\n]".
#
# Identity (A7 rule 4): a claim naming OUR pid is ours only at our exact
# creation time (5 ms); without a ct it is a previous incarnation (dead).
# Any other pid: live iff its creation time is within 2 s of the recorded
# one; no recorded ct (v1) or an unreadable one: live for 20 minutes from
# line 2 only (a reused pid must never park every reader).
#
# Mutation (A7 rule 1): every read -> judge -> mutate of the marker happens
# inside ONE hold of an exclusive kernel lock on the sidecar "<marker>.lock"
# ([IO.File]::Open with FileShare.None; the kernel drops it with the process,
# so there is no stale mutex). The wait is bounded (10 s); a busy lock fails
# closed (no claim). The sidecar is never deleted.

$script:MarkerCeilingSeconds = 1200
$script:MarkerCtTolerance = 2.0
$script:MarkerOwnCtEpsilon = 0.005
$script:MarkerLockTimeoutMs = 10000
$script:MarkerHeartbeatSeconds = 300
$script:MarkerReleaseWaitSeconds = 7200
$script:ProcessCtCache = @{}
$script:MarkerClaim = "none"    # claimed | adopted | refused
$script:MarkerBlocker = 0
$script:MarkerOwnCtText = $null
$script:MarkerLastHeartbeat = $null
$script:MarkerCustodian = $null     # the watcher process Start-MarkerCustodian started
$script:MarkerOwner = $null         # @{ Pid; Ct } of that watcher once line 1 names it instead of us
$script:MarkerReleaseWaited = $false   # the R6 wait ran (windows.ps1 re-stamps the result after it)
$script:StartedAt = $null
$script:MarkerRunPattern = '\A[A-Za-z0-9._-]{1,128}\z'

function ConvertTo-UnixCt([datetime]$Time) {
    return [DateTimeOffset]::new($Time.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
}

function Format-Ct([double]$Ct) { return $Ct.ToString('F3', [Globalization.CultureInfo]::InvariantCulture) }

function Get-UnixNow { return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }

function Get-ProcessCreationCt([int]$ProcessId) {
    # Win32_Process needs only limited query rights, so it reads SYSTEM,
    # elevated and other-user processes; Get-Process .StartTime is access
    # denied for those under Windows PowerShell 5.1.
    if ($ProcessId -eq $PID) { return ConvertTo-UnixCt ([Diagnostics.Process]::GetCurrentProcess().StartTime) }
    try {
        $row = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
        if ($row -and $row.CreationDate) { return ConvertTo-UnixCt $row.CreationDate }
    } catch {}
    return $null
}

function Get-ParentProcessId([int]$ProcessId) {
    # The RECORDED parent: Windows keeps no tree, so the pid may be gone or reused.
    try {
        $row = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
        if ($row) { return [int]$row.ParentProcessId }
    } catch {}
    return 0
}

function Get-LiveProcessCt([int64]$ProcessId, [switch]$Fresh) {
    # Alive + creation time (unix seconds, $null when unreadable). The time is
    # read once per pid and kept while that pid stays alive: a waiter polls
    # liveness, never one CIM query per poll. The cache only forgets a pid it
    # SAW dead, so a pid that exited and was reused between two polls keeps the
    # old incarnation's time: every marker decision passes -Fresh (re-read).
    if ($ProcessId -le 0 -or $ProcessId -gt [int]::MaxValue) { return @{ Alive = $false; Ct = $null } }
    $id = [int]$ProcessId
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    $alive = [bool]$p
    if ($alive) { try { $alive = -not $p.HasExited } catch {} }
    if (-not $alive) {
        $script:ProcessCtCache.Remove($id)
        return @{ Alive = $false; Ct = $null }
    }
    if ($Fresh -or -not $script:ProcessCtCache.ContainsKey($id)) {
        $script:ProcessCtCache[$id] = Get-ProcessCreationCt $id
    }
    return @{ Alive = $true; Ct = $script:ProcessCtCache[$id] }
}

function Get-ProcessIdentity([int64]$ProcessId, $RecordedCt) {
    # live | dead | unknown (alive, but a creation time is missing on a side).
    $probe = Get-LiveProcessCt $ProcessId
    if (-not $probe.Alive) { return 'dead' }
    if ($null -eq $RecordedCt -or $null -eq $probe.Ct) { return 'unknown' }
    if ([Math]::Abs($probe.Ct - [double]$RecordedCt) -le $script:MarkerCtTolerance) { return 'live' }
    return 'dead'
}

function Test-ProcessIdentityLive([int64]$ProcessId, $RecordedCt) {
    return (Get-ProcessIdentity $ProcessId $RecordedCt) -ne 'dead'
}

function Test-ProcessIdentityExact([int64]$ProcessId, $RecordedCt) {
    # 'match', never 'unknown': both creation times known and within 2 s.
    return (Get-ProcessIdentity $ProcessId $RecordedCt) -eq 'live'
}

# -- parsing ----------------------------------------------------------------

function Read-MarkerText {
    try { return [System.IO.File]::ReadAllText($MarkerPath, [System.Text.Encoding]::UTF8) } catch { return $null }
}

function Get-MarkerLineValue([string]$Line) {
    if ($Line.EndsWith("`r")) { $Line = $Line.Substring(0, $Line.Length - 1) }
    return $Line.Trim([char[]]@([char]32, [char]9))
}

function ConvertTo-MarkerPid([string]$Text) {
    # ASCII digits (any count, leading zeros allowed) that fit u32, else $null (a malformed pid).
    if ($Text -cnotmatch '\A[0-9]+\z') { return $null }
    $Text = $Text.TrimStart('0')
    if ($Text.Length -eq 0) { return [int64]0 }
    if ($Text.Length -gt 10) { return $null }
    $value = [uint64]$Text
    if ($value -gt 4294967295) { return $null }
    return [int64]$value
}

function ConvertTo-MarkerCt([string]$Text) {
    # A creation time past double's range is +Infinity -- it never matches a live process.
    # (Windows PowerShell's .NET Framework throws OverflowException there; .NET Core does not.)
    $value = 0.0
    if ([double]::TryParse($Text, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value)) { return $value }
    return [double]::PositiveInfinity
}

function ConvertFrom-MarkerText([string]$Text) {
    # $null = malformed, which every reader treats as dead.
    if ($null -eq $Text) { return $null }
    if ($Text.Length -gt 0 -and $Text[0] -eq [char]0xFEFF) { $Text = $Text.Substring(1) }
    $lines = @($Text -split "`n" | ForEach-Object { Get-MarkerLineValue $_ })
    if ($lines.Count -lt 2) { return $null }
    $ownerPid = ConvertTo-MarkerPid $lines[0]
    if ($null -eq $ownerPid -or $lines[1] -cnotmatch '\A[0-9]+\z') { return $null }
    # Line 2 fits u64 (any digit count), else malformed. StartedAt (arithmetic) clamps
    # past int64 to "far future"; StartedAtText keeps the exact digits for rewrites.
    $startedText = $lines[1].TrimStart('0')
    if ($startedText.Length -eq 0) { $startedText = '0' }
    if ($startedText.Length -gt 20 -or ($startedText.Length -eq 20 -and [decimal]$startedText -gt [decimal]'18446744073709551615')) { return $null }
    $started = if ($startedText.Length -le 18) { [int64]$startedText } else { [int64]::MaxValue }
    $info = @{
        Pid = $ownerPid; StartedAt = $started; StartedAtText = $startedText; Ct = $null; CtText = $null
        DelegatePid = 0; DelegateCt = $null; DelegateCtText = $null; Run = $null; Runs = @()
    }
    if ($lines.Count -ge 3 -and $lines[2] -cmatch '\Act:([0-9]+(\.[0-9]+)?)\z') {
        $info.CtText = $Matches[1]
        $info.Ct = ConvertTo-MarkerCt $Matches[1]
    }
    $runs = New-Object System.Collections.Generic.List[string]
    for ($i = 3; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ($info.DelegatePid -eq 0 -and $line -cmatch '\Adelegate:([0-9]+) ct:([0-9]+(\.[0-9]+)?)\z') {
            $delegatePid = ConvertTo-MarkerPid $Matches[1]
            if ($null -ne $delegatePid -and $delegatePid -gt 0) {
                $info.DelegatePid = $delegatePid
                $info.DelegateCtText = $Matches[2]
                $info.DelegateCt = ConvertTo-MarkerCt $Matches[2]
            }
        } elseif ($line -cmatch '\Arun:([A-Za-z0-9._-]{1,128})\z') {
            if ($null -eq $info.Run) { $info.Run = $Matches[1] }
            $runs.Add($line)
        }
    }
    $info.Runs = @($runs)
    return $info
}

function Format-MarkerBody([int64]$OwnerPid, [string]$StartedAt, $CtText, $DelegateLine, $Runs) {
    # Canonical, LF framing on purpose: the Rust/TS/Python readers split on "\n".
    $body = "$OwnerPid`n$StartedAt`n"
    if ($CtText) { $body += "ct:$CtText`n" }
    if ($DelegateLine) { $body += "$DelegateLine`n" }
    foreach ($run in @($Runs)) { if ($run) { $body += "$run`n" } }
    return $body
}

function Get-MarkerDelegateLine($Info) {
    if ($null -eq $Info -or $Info.DelegatePid -le 0) { return $null }
    return "delegate:$($Info.DelegatePid) ct:$($Info.DelegateCtText)"
}

# -- judging ----------------------------------------------------------------

function New-MarkerContext {
    # Who "we" are and how a pid is probed; the corpus test injects its own.
    # Fresh creation times: a judgement here decides a claim, a delegate or a
    # hand-over, which must never ride a reused pid's cached identity. Once line
    # 1 names our custodian (Start-MarkerCustodian), the claim is "ours" as that.
    $own = if ($script:MarkerOwner) { $script:MarkerOwner } else { @{ Pid = $PID; Ct = (Get-LiveProcessCt $PID).Ct } }
    return @{ OwnPid = [int64]$own.Pid; OwnCt = $own.Ct; Now = (Get-UnixNow); Probe = { param($p) Get-LiveProcessCt $p -Fresh } }
}

function Get-MarkerIdentityState([int64]$ProcessId, $RecordedCt, [int64]$StartedAt, $Ctx) {
    # ours | live | dead for one (pid, ct) identity of a parsed marker.
    if ($ProcessId -le 0) { return 'dead' }
    if ($ProcessId -eq $Ctx.OwnPid) {
        if ($null -ne $RecordedCt -and $null -ne $Ctx.OwnCt -and
            [Math]::Abs([double]$RecordedCt - [double]$Ctx.OwnCt) -le $script:MarkerOwnCtEpsilon) { return 'ours' }
        return 'dead'   # a previous incarnation of our pid, never live
    }
    $probe = & $Ctx.Probe $ProcessId
    if (-not $probe.Alive) { return 'dead' }
    if ($null -eq $RecordedCt -or $null -eq $probe.Ct) {
        if (($Ctx.Now - $StartedAt) -le $script:MarkerCeilingSeconds) { return 'live' }
        return 'dead'
    }
    if ([Math]::Abs([double]$RecordedCt - [double]$probe.Ct) -le $script:MarkerCtTolerance) { return 'live' }
    return 'dead'
}

function Get-MarkerJudgement($Info, $Ctx) {
    # verdict: malformed | dead | ours | live; owner: the owner identity if
    # live, else the delegate if live, else $null.
    if ($null -eq $Info) { return @{ Verdict = 'malformed'; Owner = $null; OwnerState = 'dead'; DelegateState = 'none'; Run = $null } }
    $ownerState = Get-MarkerIdentityState $Info.Pid $Info.Ct $Info.StartedAt $Ctx
    $delegateState = 'none'
    if ($Info.DelegatePid -gt 0) { $delegateState = Get-MarkerIdentityState $Info.DelegatePid $Info.DelegateCt $Info.StartedAt $Ctx }
    $owner = $null
    if ($ownerState -ne 'dead') { $owner = $Info.Pid } elseif ($delegateState -in @('ours', 'live')) { $owner = $Info.DelegatePid }
    $verdict = 'dead'
    if ($ownerState -eq 'ours' -or $delegateState -eq 'ours') { $verdict = 'ours' } elseif ($null -ne $owner) { $verdict = 'live' }
    return @{ Verdict = $verdict; Owner = $owner; OwnerState = $ownerState; DelegateState = $delegateState; Run = $Info.Run }
}

function Get-MarkerReleaseAction($Info, $Ctx) {
    # Corpus 'release': delete | rewrite (Text) | keep.
    if ($null -eq $Info) { return @{ Action = 'keep'; Text = $null } }
    $ownerState = Get-MarkerIdentityState $Info.Pid $Info.Ct $Info.StartedAt $Ctx
    $delegateState = 'none'
    if ($Info.DelegatePid -gt 0) { $delegateState = Get-MarkerIdentityState $Info.DelegatePid $Info.DelegateCt $Info.StartedAt $Ctx }
    if ($Info.Pid -eq $Ctx.OwnPid -and $ownerState -eq 'ours') {
        if ($Info.DelegatePid -gt 0 -and $Info.DelegatePid -ne $Ctx.OwnPid -and $delegateState -eq 'live') {
            # A7 rule 5: hand over to the live delegate, keeping started_at and runs.
            return @{ Action = 'rewrite'; Text = (Format-MarkerBody $Info.DelegatePid $Info.StartedAtText $Info.DelegateCtText $null $Info.Runs) }
        }
        return @{ Action = 'delete'; Text = $null }
    }
    if ($Info.DelegatePid -gt 0 -and $Info.DelegatePid -eq $Ctx.OwnPid -and $delegateState -eq 'ours') {
        if ($ownerState -ne 'dead') {
            return @{ Action = 'rewrite'; Text = (Format-MarkerBody $Info.Pid $Info.StartedAtText $Info.CtText $null $Info.Runs) }
        }
        return @{ Action = 'delete'; Text = $null }
    }
    return @{ Action = 'keep'; Text = $null }
}

# -- the A7 lock and in-lock writes -------------------------------------------

function Open-MarkerLock([int]$TimeoutMs = $script:MarkerLockTimeoutMs) {
    # The exclusive kernel lock (A7 rule 1): an open of the sidecar that
    # shares nothing. Returns the open stream (Dispose = release) or $null
    # when it stayed busy past $TimeoutMs. Never deletes the sidecar.
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        try {
            return [System.IO.File]::Open("$MarkerPath.lock", [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        } catch {}
        if ($clock.ElapsedMilliseconds -ge $TimeoutMs) { return $null }
        Start-Sleep -Milliseconds 25
    }
}

function Write-MarkerTemp([string]$Body) {
    $tmp = "$MarkerPath.$PID.tmp"
    [System.IO.File]::WriteAllText($tmp, $Body, (New-Object System.Text.UTF8Encoding $false))
    return $tmp
}

function Publish-MarkerNew([string]$Body) {
    # A3: the complete body goes to a tmp sibling, then an exclusive hard link
    # publishes it (a lockless reader never sees a half-written claim).
    # Returns published | exists | unwritable.
    try { $tmp = Write-MarkerTemp $Body } catch {
        Write-HandoffLog "WARNING: could not write update marker: $($_.Exception.Message)"
        return "unwritable"
    }
    try {
        try {
            New-Item -ItemType HardLink -Path $MarkerPath -Value $tmp -ErrorAction Stop | Out-Null
            return "published"
        } catch {
            if ([System.IO.File]::Exists($MarkerPath)) { return "exists" }
        }
        # No hard links on this volume: a no-replace rename of the full body.
        try { [System.IO.File]::Move($tmp, $MarkerPath); return "published" } catch {
            if ([System.IO.File]::Exists($MarkerPath)) { return "exists" }
            Write-HandoffLog "WARNING: could not publish update marker: $($_.Exception.Message)"
            return "unwritable"
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Set-MarkerBodyLocked([string]$Body) {
    # Caller holds the A7 lock. Atomic replace of the whole body; a lockless
    # reader holding the file open (no FILE_SHARE_DELETE) can make one attempt
    # fail, so retry briefly. A marker that vanished is re-published.
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not [System.IO.File]::Exists($MarkerPath)) {
            if ((Publish-MarkerNew $Body) -eq 'published') { return $true }
        } else {
            $tmp = $null
            try {
                $tmp = Write-MarkerTemp $Body
                [System.IO.File]::Replace($tmp, $MarkerPath, [NullString]::Value)
                return $true
            } catch {
                if ($tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
            }
        }
        Start-Sleep -Milliseconds 50
    }
    return $false
}

function Remove-MarkerLocked {
    # Caller holds the A7 lock and judged the marker inside that hold.
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { [System.IO.File]::Delete($MarkerPath); return $true } catch {}
        Start-Sleep -Milliseconds 50
    }
    return $false
}

function Test-MarkerFileYoung {
    # A3 fallback writers create then write: an empty marker younger than 5 s
    # is a claim in flight, not a dead one.
    try { return ([DateTime]::UtcNow - [System.IO.File]::GetLastWriteTimeUtc($MarkerPath)).TotalSeconds -lt 5 } catch { return $false }
}

function Read-MarkerLocked($Ctx) {
    # absent | busy (unreadable, or an empty claim in flight) | judged.
    if (-not [System.IO.File]::Exists($MarkerPath)) { return @{ State = 'absent'; Info = $null; Judgement = @{ Verdict = 'absent'; Owner = $null; OwnerState = 'dead'; DelegateState = 'none'; Run = $null } } }
    $text = Read-MarkerText
    if ($null -eq $text -or ($text.Length -eq 0 -and (Test-MarkerFileYoung))) { return @{ State = 'busy'; Info = $null; Judgement = $null } }
    $info = ConvertFrom-MarkerText $text
    return @{ State = 'judged'; Info = $info; Judgement = (Get-MarkerJudgement $info $Ctx) }
}

. (Join-Path $PSScriptRoot 'marker-claim.ps1')

function Get-CheckoutLockPath {
    # hermes_cli/update_lock.py::checkout_lock_path: <git common dir>/hermes-update.lock,
    # else <root>/.hermes-update.lock.
    $root = Get-Variable -Name InstallRoot -ValueOnly -ErrorAction SilentlyContinue
    if (-not $root) { return $null }
    $dot = Join-Path $root '.git'
    $gitDir = $null
    if ([System.IO.Directory]::Exists($dot)) {
        $gitDir = $dot
    } elseif ([System.IO.File]::Exists($dot)) {
        $first = (([System.IO.File]::ReadAllText($dot)).TrimStart([char]0xFEFF) -split "`n")[0].Trim()
        if ($first -like 'gitdir:*') {
            $gitDir = $first.Substring(7).Trim()
            if (-not [System.IO.Path]::IsPathRooted($gitDir)) { $gitDir = Join-Path $root $gitDir }
        }
    }
    if (-not $gitDir) { return (Join-Path $root '.hermes-update.lock') }
    $commonFile = Join-Path $gitDir 'commondir'
    if ([System.IO.File]::Exists($commonFile)) {
        $common = (([System.IO.File]::ReadAllText($commonFile)).TrimStart([char]0xFEFF) -split "`n")[0].Trim()
        if ($common) {
            $gitDir = if ([System.IO.Path]::IsPathRooted($common)) { $common } else { Join-Path $gitDir $common }
        }
    }
    return (Join-Path $gitDir 'hermes-update.lock')
}

function Test-CheckoutLockHeld {
    # R6: does some process hold the checkout lock right now? Python takes it with
    # msvcrt.locking on one byte at offset 1 MiB (LockFile), and every joiner the job
    # could not bind also locks one of the 16 lease bytes just past it (R5b): a leased
    # child outlives a killed owner and keeps the checkout busy. Probe all 1+16 bytes
    # in one LockFile (it fails if ANY of them is held) and give them straight back.
    # A lock file that exists but cannot be opened or probed (denied, opened without
    # sharing) counts as held, like marker.sh and the Desktop's own probe: reclaim
    # needs a provably free checkout.
    $path = Get-CheckoutLockPath
    if (-not $path -or -not [System.IO.File]::Exists($path)) { return $false }
    $fs = $null
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    } catch [System.IO.FileNotFoundException], [System.IO.DirectoryNotFoundException] { return $false } catch { return $true }
    try {
        try { $fs.Lock(1048576, 17) } catch { return $true }
        try { $fs.Unlock(1048576, 17) } catch {}
        return $false
    } finally {
        $fs.Dispose()
    }
}
