# ORCH V4 — Upstream PR Prep (2026-08-06)

Local-only metadata. **NOT a code change.** No source files in
`hermes_cli/` or `scripts/` were modified for this PR prep.

## Provenance

- Base branch: `main`
- Base HEAD:   `aedd0382b feat(orch-v4): deeper multi-lane plan materialize + lane accept`
- Test status: 93 passed in 31.27s on the full `tests/hermes_cli/test_kanban_orch_*.py` suite
- Source root: `/home/claw/.hermes/hermes-agent`

## Scope of this PR

This branch contains the **full ORCH V4 candidate** (M1–M8 + S1–S3 + C-min
+ multi-lane min + multi-lane deep + digest UDF + dual-bind cutover).
The commit graph walks from `9a2565ec9` (M1–M3) through `aedd0382b`
(latest deep multi-lane plan+accept).

## What is NOT in this PR (intentional)

- No native `kanban.db` mutation. N1 hold.
- No live writer switch (no `~/.hermes/orch_v4_writer.json` flip).
- No gateway restart.
- No origin/main force-push.
- No live fork push — David approves the push URL separately.

## Local-only verification

- `python3 -m pytest tests/hermes_cli/test_kanban_orch_*.py -q` → **93 passed**
- Sidecar live: `/home/claw/.hermes/orch_v4.db` (27 orch tables, fk=1)
- Native live: `/home/claw/.hermes/kanban.db` (0 orch_* tables, N1 verified)

## PR-prep checklist (operator-only; David approves)

1. Push to fork: `git push crazyief orch-v4/upstream-pr-20260806` (BLOCKED until David)
2. Open upstream PR from fork branch via `gh pr create` (BLOCKED until David)
3. Maintainer ping: 1 max per David go (sealed in `evidence/orch-v4-pr-maintainer-ping-…`)
4. Local cutover does **not** depend on upstream merge — dual-bind ON continues to run.

## Evidence roots

- `/home/claw/Desktop/evidence/orch-v4-candidate-review-20260805T062100Z/`
- `/home/claw/Desktop/evidence/orch-v4-cmin-20260805T104500Z/`
- `/home/claw/Desktop/evidence/orch-v4-cmin-tick-multilane-20260805T105500Z/`
- `/home/claw/Desktop/evidence/orch-v4-real-fanout-e2e-20260805T113000Z/`
