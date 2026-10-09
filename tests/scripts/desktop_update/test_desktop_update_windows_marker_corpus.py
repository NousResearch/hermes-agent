"""The shared update-marker corpus (A7 rule 7) on the PowerShell reader/releaser.

Every ``judge`` and ``release`` case of tests/fixtures/update_marker_corpus.json
runs through the production parser, judge and release decision of
scripts/desktop-update/marker.ps1 under Windows PowerShell, with the case's live
table, our pid/creation time and ``now`` injected. The marker text goes through
a real file and the production reader (Read-MarkerText) as well as straight
into the parser.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS = ROOT / 'tests/fixtures/update_marker_corpus.json'
MARKER_PS1 = ROOT / 'scripts/desktop-update/marker.ps1'

HARNESS = r"""
param([string]$Corpus, [string]$MarkerPs1, [string]$Work)
$ErrorActionPreference = 'Stop'
$MarkerPath = Join-Path $Work 'marker'
function Write-HandoffLog([string]$Message) {}
. $MarkerPs1
$c = Get-Content -LiteralPath $Corpus -Raw -Encoding UTF8 | ConvertFrom-Json
function New-CaseContext($Case, [int64]$OwnPid, $OwnCt) {
    $live = @{}
    if ($Case.live) { foreach ($p in $Case.live.PSObject.Properties) { $live[[int64]$p.Name] = $p.Value } }
    $probe = {
        param($p)
        if ($live.ContainsKey([int64]$p)) {
            $ct = $live[[int64]$p]
            if ($null -eq $ct) { return @{ Alive = $true; Ct = $null } }
            return @{ Alive = $true; Ct = [double]$ct }
        }
        return @{ Alive = $false; Ct = $null }
    }.GetNewClosure()
    return @{ OwnPid = $OwnPid; OwnCt = [double]$OwnCt; Now = [int64]$c.now; Probe = $probe }
}
function Read-CaseText([string]$Text) {
    [System.IO.File]::WriteAllBytes($MarkerPath, [System.Text.UTF8Encoding]::new($false).GetBytes($Text))
    return Read-MarkerText
}
$out = New-Object System.Collections.Generic.List[string]
foreach ($case in $c.judge) {
    $ownPid = if ($null -ne $case.our_pid) { $case.our_pid } else { $c.our_pid }
    $ownCt = if ($null -ne $case.our_ct) { $case.our_ct } else { $c.our_ct }
    $ctx = New-CaseContext $case $ownPid $ownCt
    $viaFile = Get-MarkerJudgement (ConvertFrom-MarkerText (Read-CaseText $case.text)) $ctx
    $direct = Get-MarkerJudgement (ConvertFrom-MarkerText $case.text) $ctx
    $out.Add((@{ kind = 'judge'; name = $case.name; verdict = $viaFile.Verdict; owner = $viaFile.Owner; run = $viaFile.Run;
                 direct = @($direct.Verdict, $direct.Owner, $direct.Run) } | ConvertTo-Json -Compress))
}
foreach ($case in $c.release) {
    $ctx = New-CaseContext $case $case.releaser_pid $case.releaser_ct
    $r = Get-MarkerReleaseAction (ConvertFrom-MarkerText (Read-CaseText $case.text)) $ctx
    $out.Add((@{ kind = 'release'; name = $case.name; action = $r.Action; text = $r.Text } | ConvertTo-Json -Compress))
}
[Console]::Out.Write(($out -join "`n") + "`n")
"""


def run_corpus(powershell: str, work: Path) -> dict[tuple[str, str], dict]:
    harness = work / 'corpus_harness.ps1'
    harness.write_text(HARNESS, encoding='utf-8')
    proc = subprocess.run(
        [powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
         '-Corpus', str(CORPUS), '-MarkerPs1', str(MARKER_PS1), '-Work', str(work)],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return {(row['kind'], row['name']): row for row in rows}


def _expected() -> list[tuple[str, dict]]:
    corpus = json.loads(CORPUS.read_text(encoding='utf-8-sig'))
    return [('judge', case) for case in corpus['judge']] + [('release', case) for case in corpus['release']]


def _mismatches(results: dict[tuple[str, str], dict]) -> list[str]:
    bad = []
    for kind, case in _expected():
        got = results.get((kind, case['name']))
        if got is None:
            bad.append(f'{kind} {case["name"]}: no result')
            continue
        want = case['expect']
        if kind == 'judge':
            seen = {'verdict': got['verdict'], 'owner': got['owner'], 'run': got['run']}
            if seen != want:
                bad.append(f'judge {case["name"]}: got {seen}, want {want}')
            if got['direct'] != [want['verdict'], want['owner'], want['run']]:
                bad.append(f'judge {case["name"]} (unread text): got {got["direct"]}, want {want}')
        else:
            seen = {'action': got['action']}
            if want['action'] == 'rewrite':
                seen['text'] = got['text']
            if seen != want:
                bad.append(f'release {case["name"]}: got {seen}, want {want}')
    return bad


@pytest.mark.platforms('windows')
def test_every_corpus_case_matches_on_windows_powershell(tmp_path: Path) -> None:
    powershell = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                              'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
    results = run_corpus(powershell, tmp_path)
    assert len(results) == len(_expected())
    assert _mismatches(results) == []
