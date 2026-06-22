# Externally-checkable evidence bundle (this round)

## 1. Reproducible diff-coverage proof (vs v0.16.0 3c231eb) — see DIFF-COVERAGE-PROOF.txt
Run `diff_coverage_proof.sh`: computes `git diff 3c231eb..HEAD` minus the union of
the 40 code PR diffs, classifies the remainder. PROVEN:
**delta(165) = in-PR(138) + 9 .bak + 12 .project-intel + 4 transcripts + 2 upstream-#29433, 0 orphans.**
(Correction this round: delta is 165 not 160; the 4 transcripts/ eval-capture files
were newly-appeared unclassified orphans, now documented as DISCARD capture artifacts.
DISCARD total = 25.)

## 2. GitHub-API head-SHA reconciliation (all 41) — see PINNED-SHAS-API.txt
Recently-fixed SHAs confirmed against the API: #50053=15c27997e ✓, #50111=c7cb84820 ✓
(both pushes landed despite the earlier disk-full condition, now cleared).

## 3. 41-vs-40 reconciliation
41 open PRs = **40 code-candidate PRs + 1 manifest (#50111, docs-only, draft)**. The
combined dry-run acts on the 40 code PRs; #50111 carries the evidence docs, not code.

## 4. PR-level CI status (all 41)
**0 checks reported on all 41** — fork-PR CI is maintainer-gated on this upstream
(non-collaborator fork PRs need maintainer approval to run CI). Uniform, documented
state; nothing red. Local CI-equivalent (ruff + ty + 285 tests) runs green
(CI-EQUIVALENT-RESULTS.txt).

## 5. Independent leak scan (all 41 tips) — 0 leaks
After scrubbing 58 real leaks found this campaign (private paths, engine names, phase
labels), the final independent scan across all 41 PR tips = **0 leaks**.
