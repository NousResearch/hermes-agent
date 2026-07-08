# Hermes Upgrade Playbook — read this BEFORE every upgrade

Purpose: carry hard-won context across upgrades so we don't re-make the same
assumptive mistakes. Each upgrade is a large 3-way merge of our carried commits
onto a new upstream; the traps below are the ones that actually bit us. Append a
new dated section each upgrade. First entry: **0.17.0 → 0.18.0 (2026-07-07)**.

Companion docs: `docs/lfdm-v018-dropped-commits.md` (what we dropped + how to
recover it). Backup of pre-upgrade state + rollback anchors are recorded in the
per-upgrade section below.

---

## Reusable lessons (version-agnostic — check these every time)

### 1. A `no_agent` cron job is defined by its `script`, NOT its `command`
**The mistake (0.18):** we saw cron jobs with `command`, `action`, `kind` all
`null` and flagged them "malformed / non-functional placeholders" — then
quarantined them. **Wrong.** They were valid `no_agent` **script** jobs: for that
job type `command`/`action`/`kind` are *legitimately* null and the work lives in
a separate `script` field (a wrapper under `~/.hermes/scripts/`). We'd only
printed a subset of keys and never looked at `script`/`no_agent`.

**Rule:** before ever calling a cron job malformed / dead / removable, dump the
**full** record and check `no_agent` + `script`. If `no_agent: true` and `script`
points at a real file, it is a working job — `command: null` is correct.
Two live examples that tripped this: `n8n-mcp-reaper` (reaps orphaned n8n-mcp
Docker containers) and `wiki-autodraft-weekly` (drafts wiki pages to `_drafts/`).

### 2. The cron job schema tightens across versions — validate `jobs.json`
0.17→0.18 made the scheduler tick require bookkeeping fields (`id`,
`created_at`, `next_run_at`, `state`, `repeat`, …). **Hand-authored** job records
missing those fields ran fine on 0.17 but crash the 0.18 tick (the no-`next_run_at`
recovery branch does `job.get("name", job["id"])`, an eager-eval that KeyErrors on
a null/absent `id`). **Worse: one malformed job aborts the ENTIRE tick** — all
jobs stop firing (total cron outage), not just the bad one.

**Rule:** before AND after an upgrade, validate every record in
`~/.hermes/cron/jobs.json` has an `id` and full schema. Never hand-write job
records — always `hermes cron create …` (it generates canonical full-schema
records and coordinates with the running gateway's store lock). Quick check:
```python
raw=json.load(open("~/.hermes/cron/jobs.json")); jobs=raw.get("jobs",raw)
assert all(j.get("id") for j in jobs), "job(s) missing id will crash the 0.18+ tick"
```
Consider hardening the tick upstream so a single bad job can't take down all cron.

### 3. Differential testing pins regressions — and "environmental" is a trap
Method that worked: run the affected suite against BOTH pristine-vNEW (a detached
worktree at the release tag) and our merged branch on the *same* venv; the
failures that appear **only on our branch** are the real regressions. Everything
failing on pristine too is upstream/environmental — not ours to fix.

**But:** do NOT wave a failure off as "environmental" without proving the
mechanism. The `KeyError: 'id'` we dismissed as a bare-venv test artifact was the
*same* bug that later caused a live cron outage on deploy (see Lesson 1/2). If you
can't explain *why* it's environmental, treat it as real.

### 4. "Superseded / redundant / removable" claims need proof, not plausibility
Two audit verdicts in 0.18 were plausible but wrong on inspection:
- "Drop our manual cron fallback loop, 0.18's `get_fallback_chain` replaces it" —
  wrong: `get_fallback_chain()` only *builds* the chain; nothing iterates it, so
  the manual loop is load-bearing. Deleting it would kill cron fallback.
- "These null-command cron jobs are dead placeholders" — wrong (Lesson 1).

**Rule:** before deleting/skipping carried work, verify the replacement actually
covers it end-to-end. When you drop something, log an evidence entry with the
exact recovery command (see the dropped-commits ledger).

### 5. Watch for precedence inversions when grafting our config onto rewritten code
Our cron model/provider routing, re-applied onto 0.18's rewritten `run_job`,
silently **inverted** model precedence (`HERMES_MODEL` env began overriding
config.yaml `model:`), breaking `${VAR}` expansion and the spend-safety
model-drift guard. Grafting KEEP-config onto a rewritten hotspot is exactly where
semantics drift — diff the resolution order against pristine and test the guards.

---

## Upgrade log

### 0.17.0 → 0.18.0 "The Judgment Release" (2026-07-07)
- Live install `~/.hermes/hermes-agent` (venv inside, gitignored, editable install
  pointing at that dir). Deploy = stop `hermes-gateway`+`hermes-dashboard` →
  `git checkout <branch>` in place → clear `__pycache__` → smoke test → restart.
  Cron scheduler runs IN the gateway process. Leave `openclaw-gateway` alone.
- Dep note: 0.18 pins `cryptography==46.0.7` (CVE floor) but the live venv had
  49.0.0 (newer, satisfies it) — a code-only editable swap keeps installed pkgs;
  don't downgrade. Smoke-test crypto (PyJWT roundtrip) under the actual live venv.
- Merge = 18 carried commits on v0.18.0; 10 dropped (see dropped-commits ledger).
  Hotspots resolved as "take 0.18 structure, graft our gap-fillers":
  `cron/scheduler.py`, `hermes_state.py`, `tools/kanban_tools.py`.
- Rollback anchors: pre-upgrade branch `lfdm-main` @ `9636d88`; 9G backup at
  `~/.hermes/backups/pre-v018-update-20260707T175806/` (git bundle + config/code
  tar + sqlite online backups + systemd units + MANIFEST).
- Post-deploy surprise: cron outage from the two hand-authored `no_agent` jobs
  (Lessons 1+2); fixed by recreating them via `hermes cron create`.
