# marker.ps1 -- the update marker (contract C1 v2) for windows.ps1, which
# dot-sources it as its first act. Pure PowerShell + CIM on purpose: the claim
# runs before the first Add-Type.
#
# Body: "<pid>\n<started_at>\nct:<creation>\n" plus an optional line 4
# "delegate:<pid> ct:<creation>". Every reader (Python, Rust, Electron, bash,
# this file) parses it the same way (amendment A2): a leading BOM and CRLF are
# accepted; line 1 is the owner pid and line 2 an integer started_at, else the
# marker is malformed (dead); line 3 must be `ct:<n>` else the marker is v1;
# line 4 must be `delegate:<pid> ct:<n>` else it is ignored.
#
# Liveness (A1): a live pid whose creation time matches is live at any age; a
# live pid whose creation time cannot be read, or a v1 marker, is live for 20
# minutes from started_at only (a reused pid must never park every reader).

$script:MarkerCeilingSeconds = 1200
$script:ProcessCtCache = @{}
$script:MarkerBody = $null      # what we published; release compares lines 1-3
$script:MarkerClaim = "none"    # claimed | adopted | refused | unwritable
$script:MarkerBlocker = 0
$script:StartedAt = $null

function ConvertTo-UnixCt([datetime]$Time) {
    return [DateTimeOffset]::new($Time.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
}

function Format-Ct([double]$Ct) { return $Ct.ToString('F3', [Globalization.CultureInfo]::InvariantCulture) }

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

function Get-LiveProcessCt([int]$ProcessId) {
    # Alive + creation time (unix seconds, $null when unreadable). The time is
    # read once per pid and kept while that pid stays alive: a waiter polls
    # liveness, never one CIM query per poll.
    if ($ProcessId -le 0) { return @{ Alive = $false; Ct = $null } }
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    $alive = [bool]$p
    if ($alive) { try { $alive = -not $p.HasExited } catch {} }
    if (-not $alive) {
        $script:ProcessCtCache.Remove($ProcessId)
        return @{ Alive = $false; Ct = $null }
    }
    if (-not $script:ProcessCtCache.ContainsKey($ProcessId)) {
        $script:ProcessCtCache[$ProcessId] = Get-ProcessCreationCt $ProcessId
    }
    return @{ Alive = $true; Ct = $script:ProcessCtCache[$ProcessId] }
}

function Get-ProcessIdentity([int]$ProcessId, $RecordedCt) {
    # live | dead | unknown (alive, but a creation time is missing on a side).
    $probe = Get-LiveProcessCt $ProcessId
    if (-not $probe.Alive) { return 'dead' }
    if ($null -eq $RecordedCt -or $null -eq $probe.Ct) { return 'unknown' }
    if ([Math]::Abs($probe.Ct - [double]$RecordedCt) -le 2.0) { return 'live' }
    return 'dead'
}

function Test-ProcessIdentityLive([int]$ProcessId, $RecordedCt) {
    return (Get-ProcessIdentity $ProcessId $RecordedCt) -ne 'dead'
}

function Read-MarkerText {
    try { return [System.IO.File]::ReadAllText($MarkerPath, [System.Text.Encoding]::UTF8) } catch { return $null }
}

function Get-MarkerHead([string]$Text) {
    # Lines 1-3 (owner identity); line 4 is the delegate and may change.
    return (@($Text -split "`n") | Select-Object -First 3) -join "`n"
}

function ConvertFrom-MarkerText([string]$Text) {
    # Positional (A2). $null = malformed, which every reader treats as dead.
    $lines = @($Text.TrimStart([char]0xFEFF) -split "`n" | ForEach-Object { $_.TrimEnd("`r") })
    if ($lines.Count -lt 2 -or $lines[0].Trim() -cnotmatch '^[0-9]+$' -or $lines[1].Trim() -cnotmatch '^[0-9]+$') { return $null }
    $ownerPid = 0
    $started = 0L
    if (-not [int]::TryParse($lines[0].Trim(), [ref]$ownerPid) -or $ownerPid -le 0) { return $null }
    if (-not [int64]::TryParse($lines[1].Trim(), [ref]$started)) { return $null }
    $info = @{ Pid = $ownerPid; StartedAt = $started; Ct = $null; DelegatePid = 0; DelegateCt = $null }
    $invariant = [Globalization.CultureInfo]::InvariantCulture
    if ($lines.Count -ge 3 -and $lines[2] -cmatch '^ct:([0-9]+(\.[0-9]+)?)$') {
        $info.Ct = [double]::Parse($Matches[1], $invariant)
    }
    if ($lines.Count -ge 4 -and $lines[3] -cmatch '^delegate:([0-9]+) ct:([0-9]+(\.[0-9]+)?)$') {
        $delegatePid = 0
        if ([int]::TryParse($Matches[1], [ref]$delegatePid)) {
            $info.DelegatePid = $delegatePid
            $info.DelegateCt = [double]::Parse($Matches[2], $invariant)
        }
    }
    return $info
}

function Test-MarkerIdentityLive([int]$ProcessId, $RecordedCt, $Info) {
    switch (Get-ProcessIdentity $ProcessId $RecordedCt) {
        'live' { return $true }
        'unknown' { return ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $Info.StartedAt) -le $script:MarkerCeilingSeconds }
    }
    return $false
}

function Test-MarkerDelegateLive($Info) {
    if ($null -eq $Info -or $Info.DelegatePid -le 0 -or $Info.DelegatePid -eq $PID) { return $false }
    return Test-MarkerIdentityLive $Info.DelegatePid $Info.DelegateCt $Info
}

function Test-MarkerOwnerLive($Info) {
    if ($null -eq $Info -or $Info.Pid -eq $PID) { return $false }   # our own pid = a reused, stale claim
    return Test-MarkerIdentityLive $Info.Pid $Info.Ct $Info
}

function Write-MarkerTemp([string]$Body) {
    $tmp = "$MarkerPath.$PID.tmp"
    [System.IO.File]::WriteAllText($tmp, $Body, (New-Object System.Text.UTF8Encoding $false))
    return $tmp
}

function Publish-MarkerNew([string]$Body) {
    # A3: the complete body goes to a tmp sibling, then an exclusive hard link
    # publishes it. Never create-then-write: a reader saw the empty file as a
    # dead claim and deleted it. Returns published | exists | unwritable.
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

function Set-MarkerIfUnchanged([string]$Expected, [string]$Body) {
    # Compare-and-swap: replace only the exact bytes we judged.
    if ((Read-MarkerText) -cne $Expected) { return $false }
    $tmp = $null
    try {
        $tmp = Write-MarkerTemp $Body
        [System.IO.File]::Replace($tmp, $MarkerPath, [NullString]::Value)
        return $true
    } catch {
        if ($tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        return $false
    }
}

function Remove-MarkerIfUnchanged([string]$Expected) {
    # Compare-and-delete: never unlink by path on an older verdict.
    if ((Read-MarkerText) -cne $Expected) { return $false }
    try { [System.IO.File]::Delete($MarkerPath); return $true } catch { return $false }
}

function Test-MarkerFileYoung {
    # A3 fallback writers create then write: an empty marker younger than 5 s
    # is a claim in flight, not a dead one.
    try { return ([DateTime]::UtcNow - [System.IO.File]::GetLastWriteTimeUtc($MarkerPath)).TotalSeconds -lt 5 } catch { return $false }
}

function Invoke-MarkerClaim {
    # A hand-off started by the Desktop (-DesktopPid) only ADOPTS the
    # Desktop's bridge claim (A4): no bridge, or a bridge that is not that
    # Desktop's live claim, means the Desktop already gave up on this run.
    $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $startedAt = 0L
    $hasStartedAt = [int64]::TryParse($env:HERMES_UPDATE_STARTED_AT, [ref]$startedAt)
    if (-not $hasStartedAt -or $startedAt -gt $epoch -or ($epoch - $startedAt) -gt $script:MarkerCeilingSeconds) {
        $startedAt = $epoch
    }
    $script:StartedAt = $startedAt
    $own = Get-LiveProcessCt $PID
    $ctLine = if ($null -ne $own.Ct) { "ct:$(Format-Ct $own.Ct)`n" } else { "" }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        # LF framing on purpose: the Rust/TS/Python readers split on "\n".
        $body = "$PID`n$($script:StartedAt)`n$ctLine"
        if ($DesktopPid -le 0) {
            switch (Publish-MarkerNew $body) {
                "published" {
                    $script:MarkerBody = $body
                    Write-HandoffLog "claimed update marker (pid $PID)"
                    return "claimed"
                }
                "unwritable" { return "unwritable" }
            }
        }
        $seen = Read-MarkerText
        if ($null -eq $seen) {
            if ($DesktopPid -gt 0) {
                Write-HandoffLog "no update marker from the Desktop (pid $DesktopPid): it gave up on this hand-off; exiting without claiming"
                return "refused"
            }
            continue
        }
        $info = ConvertFrom-MarkerText $seen
        $ownerLive = Test-MarkerOwnerLive $info
        $delegateLive = Test-MarkerDelegateLive $info
        if ($seen -eq "" -and (Test-MarkerFileYoung)) { $ownerLive = $true }
        if ($DesktopPid -gt 0) {
            if ($ownerLive -and -not $delegateLive -and $info.Pid -eq $DesktopPid) {
                # One acquisition time for the whole chain: keep its started_at.
                if ($info.StartedAt -gt 0 -and $info.StartedAt -le $epoch) { $script:StartedAt = $info.StartedAt }
                $body = "$PID`n$($script:StartedAt)`n$ctLine"
                if (Set-MarkerIfUnchanged $seen $body) {
                    $script:MarkerBody = $body
                    Write-HandoffLog "adopted update marker from desktop pid $DesktopPid (pid $PID)"
                    return "adopted"
                }
                continue
            }
            Write-HandoffLog "update marker is not the live bridge claim of desktop pid ${DesktopPid}: it gave up on this hand-off; exiting without claiming"
            return "refused"
        }
        if ($ownerLive -or $delegateLive) {
            $script:MarkerBlocker = if ($delegateLive) { $info.DelegatePid } elseif ($info) { $info.Pid } else { 0 }
            Write-HandoffLog "update marker is held by live pid $($script:MarkerBlocker); refusing"
            return "refused"
        }
        $stalePid = if ($info) { $info.Pid } else { "?" }
        Write-HandoffLog "reclaiming stale update marker (owner pid $stalePid is not running)"
        [void](Remove-MarkerIfUnchanged $seen)
    }
    return "refused"
}

function Remove-MarkerIfOwned {
    # Compare-and-delete (C1 rule 5): only our own lines 1-3, and never while
    # a line-4 delegate (an update process running under our claim) lives.
    if ($NoMarkerCleanup -or $null -eq $script:MarkerBody) { return }
    try {
        $seen = Read-MarkerText
        if ($null -eq $seen) { return }
        if ((Get-MarkerHead $seen) -cne (Get-MarkerHead $script:MarkerBody)) {
            $firstLine = (@($seen -split "`n"))[0]
            Write-HandoffLog "leaving update marker: owned by pid '$firstLine', not us ($PID)"
            return
        }
        $info = ConvertFrom-MarkerText $seen
        if (Test-MarkerDelegateLive $info) {
            Write-HandoffLog "keeping update marker: delegate pid $($info.DelegatePid) is still running"
            return
        }
        if (Remove-MarkerIfUnchanged $seen) { Write-HandoffLog "removed update marker (owned)" }
    } catch {}
}

function Add-MarkerDelegate([int[]]$Candidates) {
    # The tree could not be quiesced: name a still-running member as the
    # marker's delegate so every reader keeps it LIVE exactly as long as that
    # process lives, instead of judging it dead with this script. A2 readers
    # ignore a delegate line without a creation time, so a member whose time
    # cannot be read is skipped.
    if ($null -eq $script:MarkerBody) { return }
    $seen = Read-MarkerText
    if ($null -eq $seen -or (Get-MarkerHead $seen) -cne (Get-MarkerHead $script:MarkerBody)) { return }
    if (Test-MarkerDelegateLive (ConvertFrom-MarkerText $seen)) { return }
    foreach ($candidate in @($Candidates)) {
        if ($candidate -le 0 -or $candidate -eq $PID) { continue }
        $probe = Get-LiveProcessCt $candidate
        if (-not $probe.Alive -or $null -eq $probe.Ct) { continue }
        $line = "delegate:$candidate ct:$(Format-Ct $probe.Ct)"
        if (Set-MarkerIfUnchanged $seen ((Get-MarkerHead $script:MarkerBody) + "`n" + $line + "`n")) {
            Write-HandoffLog "update marker now names surviving updater pid $candidate as its delegate"
        }
        return
    }
}
