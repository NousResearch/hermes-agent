# collect-logs - rebuilds D:\logs\ and D:\logs.zip (ledger snapshot + git state + manifest + completeness self-check). Run it any time by hand, and at the end of every agent task that touched the repo or the ledger.
<#
  scripts\collect-logs.ps1  [-OutDir <path>]  [-RepoRoot <path>]  [-Quiet]

  WHAT IT DOES
    Assembles the single review/handoff bundle for this checkout, then zips it:

      <OutDir>\ledger\             verbatim copy of <repo>\logs\ledger\  (the tracked project ledger)
      <OutDir>\repo-state.txt      git snapshot - HEAD, branch, ahead/behind origin & upstream,
                                   uncommitted files, recent commits
      <OutDir>\HANDOFF-INDEX.md    generated manifest + the completeness-check results
      <OutDir>\redaction-report.txt what the mandatory redaction pass touched (see below)
      <OutDir>\repo-runtime-logs\  <repo>\logs\*.log / *.jsonl  - EXCLUDED by default
                                   (highest-risk, lowest-value); pass -IncludeRuntimeLogs to include
      <OutDir>\*.md               LEFT UNTOUCHED by staging - the plain-language session reports
                                   humans drop here (but the redaction pass below still scans them)

    Before zipping, EVERY text file in <OutDir> is run through the agent's production
    credential redactor (scripts\redact_handoff.py -> agent.redact). If a likely secret
    SURVIVES redaction, the zip is NOT created: the run fails, names the file:line, and
    leaves any previous zip renamed to <OutDir>.zip.stale. Pattern-matching is a backstop,
    not a guarantee.

    Then compresses <OutDir> to <OutDir>.zip (default D:\logs.zip), overwriting any prior copy,
    writes <OutDir>.zip.sha256 next to it, and prints a PASS / WARN / FAIL completeness report.

    Legacy nested bundles (<OutDir>\north-forge-agent-logs-*.zip) are removed - the live
    ledger copy in <OutDir>\ledger\ supersedes them.

  WHEN TO RUN IT
    - Any time, by hand:  double-click scripts\collect-logs.cmd, or run this file in PowerShell.
    - Automatically:      at the end of every agent task (standing practice -
                          see logs/ledger/README.md, "Agent conduct").

  DEFAULTS
    -RepoRoot  parent folder of this script's folder      (...\north-forge-agent)
    -OutDir    <root of the repo's drive>\logs            (D:\north-forge-agent  ->  D:\logs)

  EXIT CODE   0 = no FAIL (WARN is allowed).   1 = at least one FAIL - the bundle is incomplete
              or the redaction gate blocked the zip.
  REQUIRES    PowerShell 5.1+.  git is used if on PATH; without it the git snapshot is skipped.
              A Python that can import agent.redact is REQUIRED for the redaction gate - the
              repo .venv, a sibling bootstrap venv, or python on PATH. No zip without it.
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [string]$RepoRoot,
    [switch]$IncludeRuntimeLogs,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

# --- resolve paths --------------------------------------------------------
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not $OutDir) { $OutDir = (Split-Path -Qualifier $RepoRoot) + '\logs' }

$LedgerSrc = Join-Path $RepoRoot 'logs\ledger'
if (-not (Test-Path -LiteralPath $LedgerSrc)) {
    Write-Error "No ledger at '$LedgerSrc' - is -RepoRoot correct? Cannot build a handoff without it."
    exit 1
}

# never let the staging dir sit inside the repo (it would recursively swallow itself)
$OutDirFull = [System.IO.Path]::GetFullPath($OutDir)
if ($OutDirFull.TrimEnd('\').ToLower().StartsWith($RepoRoot.TrimEnd('\').ToLower() + '\')) {
    Write-Error "-OutDir '$OutDirFull' is inside the repo. Pick a location outside '$RepoRoot' (default is D:\logs)."
    exit 1
}
if (-not (Test-Path -LiteralPath $OutDirFull)) { New-Item -ItemType Directory -Path $OutDirFull -Force | Out-Null }
$OutDir  = (Resolve-Path -LiteralPath $OutDirFull).Path
$OutLeaf = Split-Path -Leaf $OutDir
$Zip     = $OutDir.TrimEnd('\') + '.zip'

$stamp    = Get-Date
$stampIso = $stamp.ToString('yyyy-MM-dd HH:mm:ss zzz')

# --- check accumulator --------------------------------------------------
$checks = New-Object System.Collections.Generic.List[object]
function Add-Check([string]$level, [string]$text) { $checks.Add([pscustomobject]@{ Level = $level; Text = $text }) }

# write UTF-8, no BOM, LF line endings (matches the .sh mirror; clean for any reader)
function Write-TextFile([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, ($text -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
}

# --- 1. refresh the ledger copy --------------------------------------
$LedgerDst = Join-Path $OutDir 'ledger'
if (Test-Path -LiteralPath $LedgerDst) { Remove-Item -LiteralPath $LedgerDst -Recurse -Force }
Copy-Item -LiteralPath $LedgerSrc -Destination $LedgerDst -Recurse -Force

$srcFiles = @(Get-ChildItem -LiteralPath $LedgerSrc -Recurse -File)
$missing  = @()
foreach ($f in $srcFiles) {
    $rel = $f.FullName.Substring($LedgerSrc.Length).TrimStart('\')
    $dst = Join-Path $LedgerDst $rel
    if (-not (Test-Path -LiteralPath $dst) -or (Get-Item -LiteralPath $dst).Length -ne $f.Length) { $missing += $rel }
}
if ($missing.Count -eq 0) {
    Add-Check 'OK' ("ledger copied in full - {0} file(s) under {1}\ledger\" -f $srcFiles.Count, $OutLeaf)
} else {
    Add-Check 'FAIL' ("ledger copy incomplete - {0} file(s) missing or size-mismatched: {1}" -f $missing.Count, ($missing -join ', '))
}

# key ledger files present and non-empty
$need = @('README.md','INDEX.md','CHANGELOG.md','errors\ERROR-LOG.md','decisions\DECISION-LOG.md')
$bad  = @()
foreach ($n in $need) {
    $p = Join-Path $LedgerDst $n
    if (-not (Test-Path -LiteralPath $p) -or (Get-Item -LiteralPath $p).Length -eq 0) { $bad += $n }
}
$auditCount = @(Get-ChildItem -LiteralPath (Join-Path $LedgerDst 'audits') -Filter 'AUDIT-*.md' -ErrorAction SilentlyContinue).Count
$tmplCount  = @(Get-ChildItem -LiteralPath (Join-Path $LedgerDst 'templates') -Filter '*.md' -ErrorAction SilentlyContinue).Count
if ($bad.Count -gt 0) { Add-Check 'FAIL' ("ledger missing/empty: {0}" -f ($bad -join ', ')) }
if ($auditCount -lt 1) { Add-Check 'WARN' 'ledger has no AUDIT- file' } else { Add-Check 'OK' "ledger carries $auditCount audit(s), $tmplCount template(s)" }
if ($tmplCount -lt 4)  { Add-Check 'WARN' "ledger templates/ has only $tmplCount file(s) - expected 4 (audit/change/decision/error)" }

# --- 2. repo runtime logs (EXCLUDED by default) -------------------
$RtDst = Join-Path $OutDir 'repo-runtime-logs'
if (Test-Path -LiteralPath $RtDst) { Remove-Item -LiteralPath $RtDst -Recurse -Force }
$rtLogs = @(Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'logs') -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq '.log' -or $_.Extension -eq '.jsonl' })
if ($IncludeRuntimeLogs) {
    if ($rtLogs.Count -gt 0) {
        New-Item -ItemType Directory -Path $RtDst -Force | Out-Null
        $rtLogs | Copy-Item -Destination $RtDst -Force
        Add-Check 'WARN' ("-IncludeRuntimeLogs: copied {0} repo runtime log(s) - high-risk content, still redacted before zip" -f $rtLogs.Count)
    } else {
        Add-Check 'OK' '-IncludeRuntimeLogs set, but no repo runtime *.log / *.jsonl exist'
    }
} elseif ($rtLogs.Count -gt 0) {
    Add-Check 'OK' ("{0} repo runtime log(s) EXCLUDED (default) - pass -IncludeRuntimeLogs to include" -f $rtLogs.Count)
}

# --- 3. drop legacy nested bundles -------------------------------
$legacy = @(Get-ChildItem -LiteralPath $OutDir -Filter 'north-forge-agent-logs-*.zip' -ErrorAction SilentlyContinue)
foreach ($z in $legacy) {
    Remove-Item -LiteralPath $z.FullName -Force
    Add-Check 'OK' ("removed superseded nested bundle {0} (its content is now ledger\)" -f $z.Name)
}

# --- 4. git snapshot -> repo-state.txt --------------------------
$RepoState    = Join-Path $OutDir 'repo-state.txt'
$aheadOrigin  = $null
$behindOrigin = $null
$dirtyFiles   = @()
$gitOk        = $false
$headSha      = ''
$branch       = ''

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("repo-state.txt   generated $stampIso")
[void]$sb.AppendLine("host $env:COMPUTERNAME   by $env:USERNAME   via scripts\collect-logs.ps1")
[void]$sb.AppendLine("repo $RepoRoot")
[void]$sb.AppendLine(('=' * 72))

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    function Invoke-Git([string[]]$a) {
        try { return (& git -C $RepoRoot @a 2>&1 | Out-String).TrimEnd() }
        catch { return "<git $($a -join ' ') failed: $_>" }
    }
    $gitOk   = $true
    $headSha = Invoke-Git @('rev-parse','HEAD')
    $branch  = Invoke-Git @('rev-parse','--abbrev-ref','HEAD')
    [void]$sb.AppendLine("HEAD    $headSha")
    [void]$sb.AppendLine("branch  $branch")
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('--- git status -sb ---')
    [void]$sb.AppendLine((Invoke-Git @('status','-sb')))
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('--- vs origin/main  (left=local ahead, right=behind) ---')
    $ro = Invoke-Git @('rev-list','--left-right','--count','HEAD...origin/main')
    [void]$sb.AppendLine($ro)
    if ($ro -match '^\s*(\d+)\s+(\d+)\s*$') { $aheadOrigin = [int]$Matches[1]; $behindOrigin = [int]$Matches[2] }
    [void]$sb.AppendLine('--- vs upstream/main ---')
    [void]$sb.AppendLine((Invoke-Git @('rev-list','--left-right','--count','HEAD...upstream/main')))
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('--- git log --oneline -15 ---')
    [void]$sb.AppendLine((Invoke-Git @('log','--oneline','-15')))
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('--- remotes ---')
    [void]$sb.AppendLine((Invoke-Git @('remote','-v')))
    [void]$sb.AppendLine('')
    $porc = Invoke-Git @('status','--porcelain')
    if ($porc) { $dirtyFiles = @($porc -split "`n" | Where-Object { $_.Trim() }) }
    [void]$sb.AppendLine('--- uncommitted (git status --porcelain) ---')
    if ($dirtyFiles.Count -gt 0) { [void]$sb.AppendLine($porc) } else { [void]$sb.AppendLine('(working tree clean)') }
} else {
    [void]$sb.AppendLine('git not found on PATH - snapshot skipped.')
}
Write-TextFile $RepoState $sb.ToString()

if (-not $gitOk) {
    Add-Check 'WARN' 'git not on PATH - repo-state.txt has no snapshot; verify branch state by hand'
} else {
    if ($null -ne $behindOrigin -and $behindOrigin -gt 0) {
        Add-Check 'WARN' "local HEAD is $behindOrigin commit(s) BEHIND origin/main - checkout may be stale"
    }
    if ($null -ne $aheadOrigin) {
        if ($aheadOrigin -gt 0) { Add-Check 'OK' "$aheadOrigin commit(s) held on local main, not pushed (expected - review-before-push)" }
        else { Add-Check 'OK' 'local main is level with origin/main' }
    }
    if ($dirtyFiles.Count -gt 0) {
        $show = ($dirtyFiles | Select-Object -First 12) -join '; '
        $more = ''
        if ($dirtyFiles.Count -gt 12) { $more = ' ...' }
        Add-Check 'WARN' ("working tree has {0} uncommitted change(s): {1}{2} - commit intended edits before treating this bundle as final" -f $dirtyFiles.Count, $show, $more)
    } else {
        Add-Check 'OK' 'working tree clean'
    }
}

# --- 5. session-report freshness ------------------------------
$reports = @(Get-ChildItem -LiteralPath $OutDir -Filter '*.md' -File | Where-Object { $_.Name -ne 'HANDOFF-INDEX.md' })
function Get-MaxDate([string]$text) {
    # no \b anchors: dates here sit next to '_' (a word char), e.g. TOPIC_2026-09-06.md
    $m = [regex]::Matches($text, '(20\d\d-\d\d-\d\d)')
    if ($m.Count -eq 0) { return $null }
    return ($m | ForEach-Object { $_.Groups[1].Value } | Sort-Object | Select-Object -Last 1)
}
$reportMax = Get-MaxDate (($reports | ForEach-Object { $_.Name }) -join ' ')
$ledgerText = ((Get-ChildItem -LiteralPath $LedgerDst -Recurse -File | Get-Content -Raw) -join "`n")
$ledgerIds  = [regex]::Matches($ledgerText, '(?:CHG|ERR|DECISION|AUDIT|RUN)-(20\d\d-\d\d-\d\d)-\d{3}')
$ledgerMax  = $null
if ($ledgerIds.Count -gt 0) { $ledgerMax = ($ledgerIds | ForEach-Object { $_.Groups[1].Value } | Sort-Object | Select-Object -Last 1) }

if ($reports.Count -eq 0) {
    Add-Check 'WARN' "no session-report *.md in $OutDir - a handoff normally carries a plain-language write-up"
} else {
    Add-Check 'OK' ("{0} session report(s) present (newest dated {1})" -f $reports.Count, $reportMax)
}
if ($ledgerMax -and $reportMax -and ($ledgerMax -gt $reportMax)) {
    Add-Check 'WARN' "ledger has entries dated $ledgerMax but the newest session report is $reportMax - a run may not have left its handoff note"
}
if ($ledgerMax -and -not $reportMax) {
    Add-Check 'WARN' "ledger active ($ledgerMax) but no dated session report found"
}

# --- 5b. resolve a Python that can import agent.redact -----------
$RedactPy = Join-Path $RepoRoot 'scripts\redact_handoff.py'
$pyCandidates = @(
    (Join-Path $RepoRoot '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $RepoRoot) ((Split-Path -Leaf $RepoRoot) + '-venv\Scripts\python.exe')),
    'python', 'python3'
)
$py = $null
foreach ($cand in $pyCandidates) {
    $exe = $cand
    if ($cand -notmatch '[\\/]') {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $exe = $cmd.Source
    } elseif (-not (Test-Path -LiteralPath $exe)) { continue }
    & $exe -c "import sys; sys.path.insert(0, r'$RepoRoot'); import agent.redact" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $exe; break }
}

# --- 5c. build a throwaway STAGING COPY and redact IT (not D:\logs\) ---
# The durable session reports in <OutDir> are left byte-for-byte intact; only the
# copy that goes into the zip is redacted. Fails closed: no redactor, or a likely
# secret that survives redaction, => no zip.
$redactionFailed = $false
$StageParent = Join-Path ([System.IO.Path]::GetTempPath()) ("nf-handoff-" + [guid]::NewGuid().ToString('N').Substring(0, 10))
$StageDir    = Join-Path $StageParent $OutLeaf
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
Get-ChildItem -LiteralPath $OutDir -Force | Where-Object {
    $_.Name -notlike '*.zip' -and $_.Name -notlike '*.zip.*' -and $_.Name -ne 'HANDOFF-INDEX.md' -and $_.Name -ne 'redaction-report.txt'
} | Copy-Item -Destination $StageDir -Recurse -Force

if (-not $py) {
    Add-Check 'FAIL' 'handoff redaction could not run - no Python that imports agent.redact (repo .venv / sibling -venv / PATH). Run scripts\bootstrap-north-forge.ps1.'
    $redactionFailed = $true
} else {
    & $py $RedactPy $StageDir
    $rc = $LASTEXITCODE
    $rep = Join-Path $StageDir 'redaction-report.txt'
    $summary = ''
    if (Test-Path -LiteralPath $rep) {
        Copy-Item -LiteralPath $rep -Destination (Join-Path $OutDir 'redaction-report.txt') -Force  # keep a record in D:\logs\
        $rtxt = Get-Content -LiteralPath $rep -Raw
        $m1 = [regex]::Match($rtxt, 'files redacted\s*:\s*(\d+)')
        $m2 = [regex]::Match($rtxt, 'text files scanned\s*:\s*(\d+)')
        if ($m1.Success -and $m2.Success) { $summary = "$($m1.Groups[1].Value) of $($m2.Groups[1].Value) file(s) redacted" }
    }
    if ($rc -eq 0) {
        Add-Check 'OK' ("handoff redaction: {0}, 0 survivors (agent.redact; backstop only; D:\logs\ sources untouched)" -f ($(if ($summary) { $summary } else { 'done' })))
    } elseif ($rc -eq 3) {
        Add-Check 'FAIL' 'handoff redaction: likely secret(s) SURVIVED - see redaction-report.txt. Zip NOT created; fix the named file:line in D:\logs\.'
        $redactionFailed = $true
    } else {
        Add-Check 'FAIL' ("handoff redaction failed (exit {0}) - see stderr. Zip NOT created." -f $rc)
        $redactionFailed = $true
    }
}

# --- 6. write HANDOFF-INDEX.md into the staging copy ---------
$IndexPath = Join-Path $StageDir 'HANDOFF-INDEX.md'
$staged = @(Get-ChildItem -LiteralPath $StageDir -Recurse -File | Where-Object { $_.FullName -ne $IndexPath })
$idx = New-Object System.Text.StringBuilder
[void]$idx.AppendLine("# Handoff bundle - $($stamp.ToString('yyyy-MM-dd'))")
[void]$idx.AppendLine('')
[void]$idx.AppendLine("Generated $stampIso by ``scripts\collect-logs.ps1`` on $env:COMPUTERNAME.")
[void]$idx.AppendLine("This folder is zipped to ``$Zip`` - that zip is the single file to hand a reviewer.")
[void]$idx.AppendLine('')
[void]$idx.AppendLine('## Git coordinates')
[void]$idx.AppendLine('')
if ($gitOk) {
    [void]$idx.AppendLine("- HEAD ``$headSha`` on ``$branch``")
    if ($null -ne $aheadOrigin) { [void]$idx.AppendLine("- vs ``origin/main``: $aheadOrigin ahead / $behindOrigin behind") }
    [void]$idx.AppendLine("- uncommitted files: $($dirtyFiles.Count)")
} else {
    [void]$idx.AppendLine('- (git unavailable when this ran - see repo-state.txt)')
}
[void]$idx.AppendLine('')
[void]$idx.AppendLine('## Contents')
[void]$idx.AppendLine('')
foreach ($f in ($staged | Sort-Object FullName)) {
    $rel = $f.FullName.Substring($StageDir.Length).TrimStart('\').Replace('\','/')
    [void]$idx.AppendLine(("- ``{0}``  ({1:N0} bytes, {2:yyyy-MM-dd HH:mm})" -f $rel, $f.Length, $f.LastWriteTime))
}
[void]$idx.AppendLine('')
[void]$idx.AppendLine('## Redaction')
[void]$idx.AppendLine('')
[void]$idx.AppendLine('Every text file in this bundle was run through the agent''s production credential')
[void]$idx.AppendLine('redactor (`agent.redact`) on a throwaway copy before zipping - the source files in')
[void]$idx.AppendLine('D:\logs\ are left intact. See `redaction-report.txt` for what was touched.')
[void]$idx.AppendLine('Pattern-matching is a backstop, not a guarantee - treat the bundle as sensitive.')
[void]$idx.AppendLine('')
[void]$idx.AppendLine('## Completeness self-check')
[void]$idx.AppendLine('')
[void]$idx.AppendLine('_(zip-integrity checks run after this file is written - see the console output of the run.)_')
[void]$idx.AppendLine('')
foreach ($c in $checks) { [void]$idx.AppendLine(("- [{0}] {1}" -f $c.Level.PadRight(4), $c.Text)) }
[void]$idx.AppendLine('')
$failN = @($checks | Where-Object { $_.Level -eq 'FAIL' }).Count
$warnN = @($checks | Where-Object { $_.Level -eq 'WARN' }).Count
$okN   = @($checks | Where-Object { $_.Level -eq 'OK' }).Count
$state = 'COMPLETE'
if ($failN -gt 0) { $state = 'INCOMPLETE' }
[void]$idx.AppendLine(("**{0}** - {1} OK, {2} WARN, {3} FAIL" -f $state, $okN, $warnN, $failN))
Write-TextFile $IndexPath $idx.ToString()

# --- 7. zip the STAGING COPY -> <OutDir>.zip  (skipped if the redaction gate failed) ---
if ($redactionFailed) {
    if (Test-Path -LiteralPath $Zip) {
        Move-Item -LiteralPath $Zip -Destination ($Zip + '.stale') -Force
        if (Test-Path -LiteralPath ($Zip + '.sha256')) { Remove-Item -LiteralPath ($Zip + '.sha256') -Force }
        Add-Check 'FAIL' ("previous zip preserved as {0}.stale - it is NOT the current handoff" -f (Split-Path -Leaf $Zip))
    }
} else {
    if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
    if (Test-Path -LiteralPath ($Zip + '.stale')) { Remove-Item -LiteralPath ($Zip + '.stale') -Force }
    Compress-Archive -Path $StageDir -DestinationPath $Zip -Force
    $zipItem = Get-Item -LiteralPath $Zip
    $sha = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash
    Set-Content -LiteralPath ($Zip + '.sha256') -Value ("{0}  {1}" -f $sha, (Split-Path -Leaf $Zip)) -Encoding ASCII

    # --- 8. post-zip verification (against the staging copy) --
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipRel = @()
    $za = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        foreach ($e in $za.Entries) {
            if ($e.FullName.EndsWith('/')) { continue }
            $n = $e.FullName.Replace('\','/')
            $n = $n -replace ('^' + [regex]::Escape($OutLeaf) + '/'), ''
            $zipRel += $n
        }
    } finally { $za.Dispose() }
    $stageRel = @($staged | ForEach-Object { $_.FullName.Substring($StageDir.Length).TrimStart('\').Replace('\','/') })
    $stageRel += 'HANDOFF-INDEX.md'
    $onlyStage = @($stageRel | Where-Object { $zipRel -notcontains $_ })
    $onlyZip   = @($zipRel   | Where-Object { $stageRel -notcontains $_ })
    if ($onlyStage.Count -eq 0 -and $onlyZip.Count -eq 0) {
        Add-Check 'OK' ("{0} contains every staged file ({1} entries)" -f (Split-Path -Leaf $Zip), $zipRel.Count)
    } else {
        if ($onlyStage.Count -gt 0) { Add-Check 'FAIL' ("not in zip: {0}" -f ($onlyStage -join ', ')) }
        if ($onlyZip.Count -gt 0)   { Add-Check 'FAIL' ("in zip but not staged: {0}" -f ($onlyZip -join ', ')) }
    }
    Add-Check 'OK' ("{0} = {1:N0} bytes, SHA256 {2}..." -f (Split-Path -Leaf $Zip), $zipItem.Length, $sha.Substring(0,16))
}

# --- cleanup the throwaway staging copy -----------------
if (Test-Path -LiteralPath $StageParent) { Remove-Item -LiteralPath $StageParent -Recurse -Force -ErrorAction SilentlyContinue }

# --- 9. report -------------------------------------------
$failN = @($checks | Where-Object { $_.Level -eq 'FAIL' }).Count
$warnN = @($checks | Where-Object { $_.Level -eq 'WARN' }).Count
if (-not $Quiet) {
    Write-Host ''
    Write-Host "collect-logs - $stampIso" -ForegroundColor Cyan
    Write-Host "  repo : $RepoRoot"
    Write-Host "  out  : $OutDir"
    Write-Host "  zip  : $Zip"
    Write-Host ''
    foreach ($c in $checks) {
        $col = 'Gray'
        if ($c.Level -eq 'OK')   { $col = 'Green' }
        if ($c.Level -eq 'WARN') { $col = 'Yellow' }
        if ($c.Level -eq 'FAIL') { $col = 'Red' }
        Write-Host ("  [{0}] {1}" -f $c.Level.PadRight(4), $c.Text) -ForegroundColor $col
    }
    Write-Host ''
    if ($failN -gt 0) {
        Write-Host "  INCOMPLETE - $failN FAIL, $warnN WARN" -ForegroundColor Red
    } elseif ($warnN -gt 0) {
        Write-Host "  COMPLETE - 0 FAIL, $warnN WARN" -ForegroundColor Yellow
    } else {
        Write-Host "  COMPLETE - 0 FAIL, 0 WARN" -ForegroundColor Green
    }
    Write-Host ''
}
if ($failN -gt 0) { exit 1 } else { exit 0 }
