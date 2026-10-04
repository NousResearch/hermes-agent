# marker-claim.ps1 -- claim / delegate / release / heartbeat / helper ops of
# the update marker, dot-sourced by marker.ps1 (still before any Add-Type).
# Every function here mutates the marker only inside one Open-MarkerLock hold
# (A7 rule 1) and judges it inside that same hold.
#
# Claim modes (hand-off protocol 2, see windows.ps1):
#   -HandoffRun R   new Desktop: ADOPT ONLY its bridge (line 1 = the Desktop's
#                   live incarnation, run:R, no live delegate), else refuse.
#   -DesktopPid P   old packaged Desktop (no run): it spawned a cmd.exe
#                   wrapper (our parent) and then OVERWROTE the marker with a
#                   v1 "<wrapper pid>\n<startedAt>\n". Accepted by LINEAGE,
#                   never by dropping the check -- see Invoke-MarkerAdoptLegacy.
#   neither         plain claim: reclaim a dead/malformed marker, refuse a live one.

function Write-MarkerOwnClaim([bool]$Exists, [int64]$Started, $Run, [string]$Verb) {
    # Caller holds the lock. Our canonical claim, published or rewritten.
    $runs = @()
    if ($Run) { $runs = @("run:$Run") }
    $body = Format-MarkerBody $PID $Started $script:MarkerOwnCtText $null $runs
    $ok = if ($Exists) { Set-MarkerBodyLocked $body } else { (Publish-MarkerNew $body) -eq 'published' }
    if (-not $ok) {
        Write-HandoffLog "could not write the update marker; refusing"
        return 'refused'
    }
    $script:StartedAt = $Started
    $script:MarkerLastHeartbeat = Get-Date
    return $Verb
}

function Get-MarkerKeptStartedAt($Info) {
    # One acquisition time for the whole chain: keep a sane line 2.
    if ($Info -and $Info.StartedAt -gt 0 -and $Info.StartedAt -le (Get-UnixNow)) { return [int64]$Info.StartedAt }
    return [int64]$script:StartedAt
}

function Invoke-MarkerAdoptRun($Read, [int]$Desktop, [string]$Run) {
    # Protocol 2 (SPEC 4a step 4): adopt-only of the Desktop bridge for run R.
    $info = $Read.Info
    $reason = $null
    if ($Desktop -le 0) { $reason = "-HandoffRun needs -DesktopPid" }
    elseif ($Run -cnotmatch $script:MarkerRunPattern) { $reason = "invalid run id" }
    elseif ($Read.State -eq 'absent') { $reason = "no bridge marker" }
    elseif ($null -eq $info) { $reason = "the marker is not a bridge claim" }
    elseif ($info.Pid -ne $Desktop) { $reason = "the marker names pid $($info.Pid)" }
    elseif ($info.Run -cne $Run) { $reason = "the marker carries run '$($info.Run)'" }
    elseif (-not (Test-ProcessIdentityExact $Desktop $info.Ct)) { $reason = "the bridge is not the desktop's live incarnation" }
    elseif ($Read.Judgement.DelegateState -in @('live', 'ours')) { $reason = "a delegate (pid $($info.DelegatePid)) is running" }
    if ($reason) {
        if ($Read.Judgement -and $Read.Judgement.Owner) { $script:MarkerBlocker = $Read.Judgement.Owner }
        Write-HandoffLog "update marker is not the bridge of desktop pid $Desktop for run ${Run}: $reason. The Desktop gave up on this hand-off; exiting without claiming"
        return 'refused'
    }
    $verdict = Write-MarkerOwnClaim $true (Get-MarkerKeptStartedAt $info) $Run 'adopted'
    if ($verdict -eq 'adopted') { Write-HandoffLog "adopted update marker from desktop pid $Desktop run $Run (pid $PID)" }
    return $verdict
}

function Get-MarkerLineage([int]$Desktop, $Info) {
    # X = our recorded parent: the old Desktop's cmd.exe wrapper.
    $x = Get-ParentProcessId $PID
    $xAlive = $false
    $xParent = 0
    if ($x -gt 0) {
        $xAlive = (Get-LiveProcessCt $x).Alive
        if ($xAlive) { $xParent = Get-ParentProcessId $x }
    }
    $envStarted = 0L
    $hasEnvStarted = [int64]::TryParse("$env:HERMES_UPDATE_STARTED_AT".Trim(), [ref]$envStarted)
    $namesX = $x -gt 0 -and $Info -and $Info.Pid -eq $x
    $xLineage = $false
    if ($namesX) {
        # Alive: it must still be the Desktop's child. Dead (start /b exits at
        # once): the marker must carry the startedAt the Desktop gave us.
        $xLineage = if ($xAlive) { $xParent -eq $Desktop } else { $hasEnvStarted -and $Info.StartedAt -eq $envStarted }
    }
    return @{
        Parent = $x; ParentAlive = $xAlive; Grandparent = $xParent; LauncherClaim = $xLineage
        Ancestor = ($x -gt 0 -and ($x -eq $Desktop -or ($xAlive -and $xParent -eq $Desktop)))
    }
}

function Invoke-MarkerAdoptLegacy($Read, [int]$Desktop) {
    # SPEC 4c: an OLD packaged Desktop (no run id) + this script.
    $info = $Read.Info
    $j = $Read.Judgement
    $lineage = Get-MarkerLineage $Desktop $info
    if ($lineage.LauncherClaim) {
        $verdict = Write-MarkerOwnClaim $true (Get-MarkerKeptStartedAt $info) $null 'adopted'
        if ($verdict -eq 'adopted') { Write-HandoffLog "adopted update marker from desktop pid $Desktop's launcher pid $($lineage.Parent) (pid $PID)" }
        return $verdict
    }
    if ($info -and $info.Pid -eq $Desktop -and $j.OwnerState -eq 'live' -and $j.DelegateState -notin @('live', 'ours')) {
        $verdict = Write-MarkerOwnClaim $true (Get-MarkerKeptStartedAt $info) $null 'adopted'
        if ($verdict -eq 'adopted') { Write-HandoffLog "adopted update marker from desktop pid $Desktop (pid $PID)" }
        return $verdict
    }
    if ($j.Verdict -in @('absent', 'dead', 'malformed') -and $lineage.Ancestor) {
        if ($Read.State -ne 'absent') { Write-HandoffLog "reclaiming stale update marker (owner pid $(if ($info) { $info.Pid } else { '?' }) is not running)" }
        $verdict = Write-MarkerOwnClaim ($Read.State -ne 'absent') $script:StartedAt $null 'claimed'
        if ($verdict -eq 'claimed') { Write-HandoffLog "claimed update marker (pid $PID) for desktop pid $Desktop" }
        return $verdict
    }
    if ($j.Owner) { $script:MarkerBlocker = $j.Owner }
    $named = if ($info) { "pid $($info.Pid)" } else { $j.Verdict }
    Write-HandoffLog "update marker ($named) is neither the live bridge of desktop pid $Desktop nor its launcher's (parent pid $($lineage.Parent)): it gave up on this hand-off; exiting without claiming"
    return 'refused'
}

function Invoke-MarkerClaim {
    # The FIRST act of windows.ps1. Returns claimed | adopted | refused.
    $epoch = Get-UnixNow
    $startedAt = 0L
    $hasStartedAt = [int64]::TryParse($env:HERMES_UPDATE_STARTED_AT, [ref]$startedAt)
    if (-not $hasStartedAt -or $startedAt -gt $epoch -or ($epoch - $startedAt) -gt $script:MarkerCeilingSeconds) {
        $startedAt = $epoch
    }
    $script:StartedAt = $startedAt
    $ctx = New-MarkerContext
    if ($null -ne $ctx.OwnCt) { $script:MarkerOwnCtText = Format-Ct $ctx.OwnCt }
    $lock = Open-MarkerLock
    if ($null -eq $lock) {
        Write-HandoffLog "update marker lock ($MarkerPath.lock) stayed busy; refusing"
        return 'refused'
    }
    try {
        $read = Read-MarkerLocked $ctx
        if ($read.State -eq 'busy') {
            Write-HandoffLog "update marker is being written by another process; refusing"
            return 'refused'
        }
        if ($HandoffRun) { return Invoke-MarkerAdoptRun $read $DesktopPid $HandoffRun }
        if ($DesktopPid -gt 0) { return Invoke-MarkerAdoptLegacy $read $DesktopPid }
        $j = $read.Judgement
        if ($j.Verdict -eq 'live') {
            $script:MarkerBlocker = $j.Owner
            Write-HandoffLog "update marker is held by live pid $($j.Owner); refusing"
            return 'refused'
        }
        if ($read.State -ne 'absent') {
            $stalePid = if ($read.Info) { $read.Info.Pid } else { "?" }
            Write-HandoffLog "reclaiming stale update marker (owner pid $stalePid is not running)"
        }
        $verdict = Write-MarkerOwnClaim ($read.State -ne 'absent') $script:StartedAt $null 'claimed'
        if ($verdict -eq 'claimed') { Write-HandoffLog "claimed update marker (pid $PID)" }
        return $verdict
    } finally {
        $lock.Dispose()
    }
}

function Add-MarkerDelegate([int[]]$Candidates) {
    # Name a running updater process (the `hermes update` child BEFORE it is
    # resumed, or a member of a tree that could not be quiesced) as the
    # delegate so every reader keeps the marker LIVE exactly as long as that
    # process lives. Only while line 1 is still our exact incarnation.
    # Returns published | kept (another live delegate stays) | skipped | lost.
    if ($script:MarkerClaim -notin @('claimed', 'adopted')) { return 'lost' }
    $lock = Open-MarkerLock
    if ($null -eq $lock) { Write-HandoffLog "update marker lock stayed busy; could not name a delegate"; return 'lost' }
    try {
        $ctx = New-MarkerContext
        $read = Read-MarkerLocked $ctx
        $info = $read.Info
        if ($null -eq $info -or $read.Judgement.OwnerState -ne 'ours') {
            Write-HandoffLog "update marker no longer names this hand-off (pid $PID); not naming a delegate"
            return 'lost'
        }
        if ($read.Judgement.DelegateState -eq 'live' -and $info.DelegatePid -notin @($Candidates)) {
            Write-HandoffLog "update marker keeps its running delegate pid $($info.DelegatePid)"
            return 'kept'
        }
        foreach ($candidate in @($Candidates)) {
            if ($candidate -le 0 -or $candidate -eq $PID) { continue }
            $probe = Get-LiveProcessCt $candidate
            if (-not $probe.Alive -or $null -eq $probe.Ct) { continue }
            $line = "delegate:$candidate ct:$(Format-Ct $probe.Ct)"
            if (-not (Set-MarkerBodyLocked (Format-MarkerBody $info.Pid $info.StartedAt $info.CtText $line $info.Runs))) { return 'lost' }
            Write-HandoffLog "update marker now names updater pid $candidate as its delegate"
            return 'published'
        }
        return 'skipped'
    } finally {
        $lock.Dispose()
    }
}

function Invoke-MarkerRelease {
    # Corpus 'release' under the lock: owner deletes, or hands the claim over
    # to a live delegate; anything that is not ours is kept.
    if ($NoMarkerCleanup -or $script:MarkerClaim -notin @('claimed', 'adopted')) { return }
    # R6: never while a survivor of the update still holds the checkout lock --
    # the marker would read free while that process still mutates the checkout.
    $waited = 0
    while (Test-CheckoutLockHeld) {
        if ($waited -eq 0) { Write-HandoffLog "a process still holds the checkout update lock; keeping the update marker until it exits" }
        if ($waited -ge $script:MarkerReleaseWaitSeconds) {
            Write-HandoffLog "checkout update lock still held after $($waited)s; leaving the update marker"
            return
        }
        Start-Sleep -Seconds 1
        $waited++
    }
    $lock = Open-MarkerLock
    if ($null -eq $lock) { Write-HandoffLog "update marker lock stayed busy; leaving the marker to identity-checking readers"; return }
    try {
        if (-not [System.IO.File]::Exists($MarkerPath)) { return }
        $info = ConvertFrom-MarkerText (Read-MarkerText)
        $release = Get-MarkerReleaseAction $info (New-MarkerContext)
        switch ($release.Action) {
            'delete' { if (Remove-MarkerLocked) { Write-HandoffLog "removed update marker (owned)" } }
            'rewrite' {
                if (Set-MarkerBodyLocked $release.Text) {
                    Write-HandoffLog "handed update marker over to running pid $((ConvertFrom-MarkerText $release.Text).Pid)"
                }
            }
            default {
                $owner = if ($info) { $info.Pid } else { '?' }
                Write-HandoffLog "leaving update marker: owned by pid '$owner', not us ($PID)"
            }
        }
    } catch {
        Write-HandoffLog "could not release the update marker: $($_.Exception.Message)"
    } finally {
        $lock.Dispose()
    }
}

function Update-MarkerHeartbeat {
    # Old packaged Electron readers age a marker on line 2 (20-minute
    # ceiling, then delete it). While a step runs, refresh line 2 every 5
    # minutes -- only while line 1 is still our exact incarnation.
    if ($script:MarkerClaim -notin @('claimed', 'adopted')) { return }
    $now = Get-Date
    if ($script:MarkerLastHeartbeat -and ($now - $script:MarkerLastHeartbeat).TotalSeconds -lt $script:MarkerHeartbeatSeconds) { return }
    $script:MarkerLastHeartbeat = $now
    $lock = Open-MarkerLock 500
    if ($null -eq $lock) { return }
    try {
        $ctx = New-MarkerContext
        $read = Read-MarkerLocked $ctx
        if ($null -eq $read.Info -or $read.Judgement.OwnerState -ne 'ours') { return }
        $info = $read.Info
        [void](Set-MarkerBodyLocked (Format-MarkerBody $info.Pid $ctx.Now $info.CtText (Get-MarkerDelegateLine $info) $info.Runs))
    } catch {} finally {
        $lock.Dispose()
    }
}

function Invoke-MarkerOp([string]$Op, [int]$Desktop, [string]$Run) {
    # SPEC 6 helper ops (Electron's only way to mutate). Returns the verdict
    # line: absent | reclaimed | held | live <pid> | busy | withdrawn | taken <pid> | foreign.
    $lock = Open-MarkerLock
    if ($null -eq $lock) { return 'busy' }
    try {
        $ctx = New-MarkerContext
        $read = Read-MarkerLocked $ctx
        if ($read.State -eq 'absent') { return 'absent' }
        if ($read.State -eq 'busy') { return 'busy' }
        $info = $read.Info
        $j = $read.Judgement
        if ($Op -eq 'reclaim') {
            if ($j.Verdict -in @('live', 'ours')) { return "live $($j.Owner)" }
            if (Test-CheckoutLockHeld) { return 'held' }   # R6: a survivor still mutates the checkout
            if (-not (Remove-MarkerLocked)) { return 'busy' }
            Write-HandoffLog "marker-op reclaim: removed a $($j.Verdict) update marker"
            return 'reclaimed'
        }
        # withdraw
        if ($null -eq $info -or $info.Run -cne $Run) { return 'foreign' }
        if ($info.Pid -eq $Desktop -and (Test-ProcessIdentityExact $Desktop $info.Ct)) {
            if (-not (Remove-MarkerLocked)) { return 'busy' }
            Write-HandoffLog "marker-op withdraw: removed the bridge of desktop pid $Desktop run $Run"
            return 'withdrawn'
        }
        if ($info.Pid -ne $Desktop -and $info.Pid -ne $PID -and (Test-ProcessIdentityExact $info.Pid $info.Ct)) {
            return "taken $($info.Pid)"
        }
        return 'foreign'
    } finally {
        $lock.Dispose()
    }
}

function Invoke-MarkerOpCli([string]$Op, [int]$Desktop, [string]$Run) {
    # Exactly one verdict line on stdout; 0 when one was printed, 64 on bad usage.
    if ($Op -cnotin @('reclaim', 'withdraw')) {
        [Console]::Error.WriteLine("unknown -MarkerOp '$Op' (reclaim|withdraw)")
        return 64
    }
    if ($Op -eq 'withdraw' -and ($Desktop -le 0 -or $Run -cnotmatch $script:MarkerRunPattern)) {
        [Console]::Error.WriteLine("-MarkerOp withdraw needs -DesktopPid and a valid -HandoffRun")
        return 64
    }
    [Console]::Out.WriteLine((Invoke-MarkerOp $Op $Desktop $Run))
    return 0
}
