# Upstream Parity Merge — Implementation Plan

> **For Apollo:** Execute task-by-task in an **isolated worktree** (never the live shared checkout).
> This is a MERGE, not greenfield code — so the TDD cycle maps to **marker-present → resolved →
> gate-green**, and "RED" proofs are revert-to-RED mutation checks on existing fork tests. PAUSE
> for Ace before Phase 5/6 (deploy). Source spec:
> `docs/sync/2026-06-29-upstream-parity-merge-PRD.md` (v0.3, Opus 2-pass APPROVE-WITH-CHANGES).

**Goal:** Merge `929dd9c0d` (upstream NousResearch/main) into `fork/main`, preserving all 220
fork-only commits, resolve 35 conflict files / 83 hunks, prove green, deploy staged with Ace gate.

**Architecture:** Two-stage spine — Stage 1 (resolution, tree non-importable, marker-free +
import-smoke checkpoints only); Stage 2 (verification, tree importable, hermetic pytest gates).
Risk-ordered resolution: mem0 (27 hunks) → architectural ports → bulk → tests/lockfiles.

**Tech Stack:** git merge, `scripts/run_tests.sh` (hermetic wrapper), `hermes update` (deploy
vector — NOT `fleet.sh`), Opus-reviewed gates.

**Frozen target:** `929dd9c0d`. **Conflict baseline:** 35 files / 83 hunks (re-measure in Task 1).

---

## STAGE 0 — Setup & audit (tree still clean)

### Task 1: Park live tree + re-measure drift
**Objective:** Triage the 4 uncommitted live-tree edits and confirm the spec's numbers still hold.
**Files:** live checkout `~/.hermes/hermes-agent` (read), no merge yet.

**Step 1:** Inspect + classify the parked changes:
```bash
cd ~/.hermes/hermes-agent && git status -sb
git diff --stat toolsets.py tests/plugins/memory/test_mem0_remember.py
# _bgr_mem0_proof.py is untracked scratch — confirm it's throwaway
```
**Step 2:** Decide each (OQ-4): superseded-by-landed-commit → discard; genuine un-landed → commit
to a fork branch first. Record disposition in `docs/sync/review/live-tree-disposition.md`.
**Step 3:** Re-measure drift (it grows daily):
```bash
git fetch origin --quiet && git fetch fork --quiet
echo "ahead: $(git rev-list --count origin/main..fork/main)  behind: $(git rev-list --count fork/main..origin/main)"
git merge-base fork/main 929dd9c0d
```
Expected: ahead ≈220, behind ≈1734+, merge-base `c6b0eb4`. **If behind grew materially**, note it
and re-run the test-merge conflict count before proceeding.
**Step 4:** Commit the disposition doc.
**Verify:** live tree clean (parked changes committed-to-branch or discarded); disposition recorded.

### Task 2: Stand up isolated worktree
**Objective:** A throwaway-safe build tree that proves `__file__` isolation.
**Step 1:**
```bash
cd ~/.hermes/hermes-agent
git worktree add ~/.hermes/worktrees/upstream-parity-20260629 -b sync/upstream-2026-06-29 fork/main
cd ~/.hermes/worktrees/upstream-parity-20260629
```
**Step 2:** Prove isolation:
```bash
PYTHONPATH=$PWD venv/bin/python -c "import run_agent; print(run_agent.__file__)" 2>/dev/null \
  || python3 -c "import sys; print([p for p in sys.path][:2])"
```
Expected: resolves to the **worktree** path, not `~/.hermes/hermes-agent`.
**Verify:** `git rev-parse --abbrev-ref HEAD` = `sync/upstream-2026-06-29`; `__file__` in worktree.

### Task 3: Verify-then-cite — confirm every INV-1 gate test EXISTS (Phase 0a)
**Objective:** No gate cites a nonexistent test (Blocker #4).
**Step 1:** Existence check (all should print the blob, none error):
```bash
for t in \
  tests/agent/test_compaction_announce_lcm.py \
  tests/agent/test_compaction_stats_reconcile.py \
  tests/plugins/memory/test_mem0_v2.py \
  tests/gateway/test_restart_cascade.py \
  tests/tools/test_send_message_origin.py \
  tests/plugins/blackbox/test_cost_perclass.py \
  tests/hermes_cli/test_kanban_model_override.py \
  tests/gateway/test_telegram_network_reconnect.py ; do
  git cat-file -e fork/main:"$t" 2>/dev/null && echo "OK  $t" || echo "MISSING  $t"
done
```
Expected: all `OK` (already spot-verified). Any `MISSING` → write the test or downgrade its INV.
**Step 2:** Record the system→test→tier map in `docs/sync/review/inv1-test-map.md` (4 mutation-tier
systems: mem0, F2 breaker, send_message, LCM-reconcile; 3 existence-tier: blackbox, kanban,
fallback).
**Verify:** the map artifact exists; zero MISSING.

### Task 4: Upstream-delta behavior audit (Phase 0b, INV-9)
**Objective:** Know what inherited behavior the merge ships unreviewed (Blocker #5).
**Step 1:** Code-surface scan (NOT commit-message grep):
```bash
git diff c6b0eb4..929dd9c0d -- '*.py' \
  | grep -nE '^\+.*(os\.getenv\("HERMES_|os\.environ\[|requests\.|httpx\.|urllib|socket\.)' \
  | tee /tmp/delta-env-net.txt | head -40
# tool-registration surface (named sites, not just toolsets.py)
git diff c6b0eb4..929dd9c0d -- toolsets.py tools/__init__.py acp_adapter/tools.py | grep -E '^\+' | grep -iE 'name=|register|schema' | head
# major dep bumps
git diff c6b0eb4..929dd9c0d -- uv.lock | grep -E '^\+name = |^\+version = ' | head -40
```
**Step 2:** Write `docs/sync/review/upstream-delta-audit.md` — per category: findings or "none
found"; flag any AGENTS.md violation (un-gated telemetry / non-secret `HERMES_*` / new always-on
core tool) for Ace's knowing-deploy call.
**Verify:** the audit artifact exists with per-category findings.

---

## STAGE 1 — Resolution (tree NOT importable: marker-free + import-smoke only)

### Task 5: Begin the merge, capture the conflict set
**Step 1:**
```bash
git merge --no-commit --no-ff 929dd9c0d > /tmp/mergeout.txt 2>&1; echo "exit $?"
git diff --name-only --diff-filter=U | tee /tmp/conflicts.txt | wc -l
```
Expected: ~35 conflicted files. **Do NOT run pytest** — tree is non-importable.
**Verify:** conflict list captured to `/tmp/conflicts.txt`.

### Task 6: Resolve mem0 (the 27-hunk monster) FIRST
**Objective:** Preserve Wave-2 hybrid retrieval AND upstream lazy-install (Blocker risk #1, D-3).
**Files:** `plugins/memory/mem0/__init__.py`, decide `tests/plugins/memory/test_mem0_v2.py` (UD).
**Step 1:** Read both sides since base:
```bash
git log -p c6b0eb4..929dd9c0d -- plugins/memory/mem0/__init__.py | head -200   # upstream intent
git log -p c6b0eb4..fork/main -- plugins/memory/mem0/__init__.py | head -200    # fork intent
```
**Step 2:** Resolve by hand — keep fork's param-drop fix, rerank gate, temporal weighting; keep
upstream's lazy-install. Decide `test_mem0_v2.py`: keep-and-port if it covers fork hybrid
retrieval; accept-delete only if coverage genuinely relocated. Record the call in the merge-notes.
**Step 3:** Marker-free check for this file:
```bash
git diff --check plugins/memory/mem0/__init__.py
grep -nE '^(<<<<<<<|=======|>>>>>>>)' plugins/memory/mem0/__init__.py || echo "CLEAN"
git grep -n 'rerank\|hybrid\|temporal' -- plugins/memory/mem0/__init__.py | head   # fork symbols present
```
**Verify:** no markers in the file; fork hybrid symbols present.

### Task 7: Architectural ports (Class A) — telegram + any extraction-relocations
**Objective:** Re-thread fork behavior onto upstream's relocated structure (INV-2), don't line-merge.
**Step 1:** Telegram ladder → new adapter path:
```bash
git log -p c6b0eb4..fork/main -- gateway/platforms/telegram.py | grep -A3 -iE 'reconnect|ladder|backoff' | head
# port the configurable reconnect ladder into plugins/platforms/telegram/adapter.py
```
**Step 2:** Inspect gateway/run.py + conversation_loop.py for extraction-relocations (the
06-16 slash_commands/turn_finalizer pattern); port any fork feature whose host was extracted.
**Step 3:** Per port, grep new-path-present + old-path-absent:
```bash
git grep -n '<fork-reconnect-symbol>' -- plugins/platforms/telegram/adapter.py   # present
git ls-files gateway/platforms/telegram.py 2>/dev/null && echo "OLD STILL TRACKED — check" || echo "old gone"
```
**Verify:** fork symbol on new path; old path resolved (gone or intentionally retained).

### Task 8: Resolve remaining core (Class B/C/D) + test conflicts + DU/UD
**Objective:** Clear the other ~19 core + 13 test conflicts by class.
**Step 1:** For each remaining conflicted file, read both diffs, resolve by class (B keep-both /
C union / D take-upstream). DU (`test_run_agent.py`) = keep upstream's mod unless our delete was
deliberate.
**Step 2:** Tree-wide marker check (all three types):
```bash
git grep -nE '^(<<<<<<<|=======|>>>>>>>)' && echo "MARKERS REMAIN" || echo "ZERO MARKERS"
git diff --check
```
**Verify:** zero markers tree-wide; `git diff --check` clean.

### Task 9: Clean-merge + import-smoke audit (Phase 1b, INV-10) — still Stage 1
**Objective:** Catch rename-miss dead-duplicates + semantic breaks the conflict set is blind to
(Blocker #1).
**Step 1:** Walk BOTH renames and deletes ∩ fork-touched (pass-2 fix):
```bash
comm -12 <(git diff -M --diff-filter=R --name-only c6b0eb4..929dd9c0d | sort) \
         <(git diff --name-only c6b0eb4..fork/main | sort)
comm -12 <(git diff --diff-filter=D --name-only c6b0eb4..929dd9c0d | sort) \
         <(git diff --name-only c6b0eb4..fork/main | sort)
```
For each hit: confirm ported to new home OR clean upstream-only relocation.
**Step 2:** Import smoke over every INV-1 module:
```bash
for m in plugins.memory.mem0 gateway.restart agent.context_compressor plugins.blackbox tools.send_message_tool hermes_cli.kanban_db; do
  venv/bin/python -c "import $m" 2>&1 | head -1 && echo "import-ok $m" || echo "IMPORT-FAIL $m"
done
```
Expected: all import-ok. **A fail here = a clean-merge semantic break — fix before Stage 2.**
**Verify:** both R∩fork and D∩fork checklists complete; all INV-1 modules import.

---

## STAGE 2 — Verification (tree importable: hermetic pytest gates)

### Task 10: Per-subsystem gates with revert-to-RED (Phase 2)
**Objective:** Each INV-1/INV-2 system green via the **hermetic wrapper**; 4 high-risk get mutation
proof.
**Step 1:** Run each subsystem (hermetic — NOT bare pytest):
```bash
scripts/run_tests.sh tests/plugins/memory
scripts/run_tests.sh tests/agent tests/gateway tests/tools tests/cron tests/hermes_cli
scripts/run_tests.sh tests/plugins/blackbox tests/hermes_cli/test_kanban_model_override.py
```
Expected: all pass.
**Step 2:** Revert-to-RED for the 4 mutation-tier systems (prove teeth):
```bash
# example: mem0 reranked path — temporarily revert the fork hunk, run the named test, expect RED, restore
# (clear __pycache__ after restore — stale .pyc can mask the restore, per prd-plan pitfall)
find . -name __pycache__ -path '*memory*' -exec rm -rf {} + 2>/dev/null
```
Do this for: mem0 hybrid, F2 breaker, send_message in-turn, LCM-reconcile. Record RED→GREEN per
system in `docs/sync/review/revert-to-red-proof.md`.
**Verify:** every subsystem green; 4 mutation proofs recorded; 3 existence-tier green-pass.

### Task 11: Lockfiles + LCM provenance + merge commit (Phase 3)
**Step 1:** Regenerate lockfiles (don't hand-merge):
```bash
venv/bin/uv lock 2>&1 | tail -3
# package-lock.json if touched: npm install --package-lock-only
git diff --stat uv.lock package-lock.json
```
**Step 2:** LCM provenance:
```bash
venv/bin/python scripts/check_lcm_upstream_drift.py 2>&1 | tail -5   # offline mode passes
```
**Step 3:** Write the merge commit (model on `d8c2836f` — commits-behind, conflict count, each
architectural port named, mechanical resolutions block):
```bash
git commit --no-edit 2>/dev/null || git commit -F docs/sync/review/merge-commit-msg.txt
git rev-list --parents -n1 HEAD | awk '{print NF-1" parents"}'   # expect 2 parents
```
**Verify:** lockfile + LCM-drift checks pass; merge commit has 2 parents.

### Task 12: Full hermetic suite (Phase 3 close) — no `-x`
**Step 1:**
```bash
scripts/run_tests.sh 2>&1 | tee /tmp/fullsuite.log | tail -20
```
Expected: complete pass set (no first-failure abort — a huge merge needs the full failure list).
**Step 2:** Confirm no fork-only test silently dropped:
```bash
git diff --diff-filter=D --name-only fork/main..HEAD -- 'tests/**' | tee /tmp/deleted-tests.txt
```
Review each deletion against INV-6.
**Verify:** full suite green; deleted-test list reviewed and justified.

### Task 13: Push, open PR, drive CI green (Phase 4)
**Step 1:**
```bash
git push fork sync/upstream-2026-06-29
gh pr create --repo Kyzcreig/hermes-agent --base main --head sync/upstream-2026-06-29 \
  --title "Merge upstream NousResearch/main into fork/main (2026-06-29 parity sync)" \
  --body-file docs/sync/review/merge-commit-msg.txt
```
**Step 2:** Drive CI; address Greptile threads on merit (per `merging-a-green-fork-pr.md`):
```bash
gh pr checks <N> --repo Kyzcreig/hermes-agent --watch
```
Expected: all required workflows green, 0 pending. `fleet-secret-scan` + `fleet-sast` specifically
green (large surface).
**Verify:** `gh pr view <N> --json statusCheckRollup` → every required check SUCCESS; no `--admin`
over red; Greptile threads resolved.

---

## STAGE 3 — Deploy (GATED — PAUSE for Ace)

### Task 14: Config-migration dry-run vs REAL fleet configs (Phase 5, INV-5) — pre-deploy gate
**Objective:** Prove the 29→32 migration preserves every fork-only key (Blocker #3). CI can't see this.
**Step 1:** Per hermes-agent host, copy its live `config.yaml` to a **writable** scratch HERMES_HOME:
```bash
for host in <hermes-agent hosts>; do
  mkdir -p /tmp/cfgdry/$host && scp $host:~/.hermes/config.yaml /tmp/cfgdry/$host/config.yaml
  HERMES_HOME=/tmp/cfgdry/$host venv/bin/hermes config migrate --quiet 2>&1 | tail -2
  # assert _config_version 29 -> 32 and fork-only keys survive
done
```
**Step 2:** Diff pre/post key set; assert `kanban`, `fallback_model`/reasoning_effort, F1/F2
gateway settings, `compression.*` all present + value-preserved. Inject a synthetic fork-only key →
must NOT be stripped.
**Verify:** per-host migration report; all pass. **No host restarts until its snapshot passes.**

### Task 15: PAUSE → present to Ace → staged deploy with soak
**Objective:** Ace go-ahead, then blast-radius-1 rollout (Phase 6).
**Step 1:** Present green PR + per-system reconciliation summary + upstream-delta audit. **WAIT for
explicit go-ahead.**
**Step 2:** On approval, merge: `gh pr merge <N> --repo Kyzcreig/hermes-agent --merge --admin`
(only after Greptile resolved — `--admin` does NOT bypass conversation-resolution).
**Step 3:** Deploy vector = **`hermes update`** per host (NOT `fleet.sh deploy` — that's
bridge/proxy only). Host-1 = a representative gateway host that is **NOT this session's host** (or
drive its relaunch via `safe-gateway-restart` detached — no synchronous self-restart). Record each
host's pre-merge SHA.
**Step 4:** Soak host-1: multiple real turns + ≥1 compaction (no `COMPACTION_STATS_RECONCILE_FAILED`)
+ ≥1 mem0 recall (reranked path) + ≥1 cron tick. Regression FLOOR (not exhaustive) → rollback host-1
(`git reset --hard <pre-sha>` + `hermes update`) before any fan-out.
**Step 5:** Fan out to remaining hosts only after host-1 soaks clean; smoke each.
**Verify:** all hermes-agent hosts on merged SHA; host-1 soak clean; per-host rollback SHA recorded
+ rollback path tested on host-1.

---

## Smoke test (the end-to-end row, per host)
After each host's `hermes update`: `hermes -p apollo -q "ping"` returns a real answer + a real turn
fires compaction/mem0 cleanly. One evidence row per host: `host-1 → ok, compaction reconciled,
mem0 reranked, cron ✓`. A host without a green smoke row is not deployed.

## Cleanup
After fan-out is green and stable: `git worktree remove ~/.hermes/worktrees/upstream-parity-20260629`;
the live `~/.hermes/hermes-agent` is updated via its own `hermes update` like any host.

---

## Roadmap (post-merge, per spec §Roadmap)
- **v0.2:** recurring upstream-sync cron (weekly/biweekly) so drift never re-walls. (D-8)
- **v0.3:** optional — automate the Class A/B/C/D conflict pre-classifier if v0.2 shows it's worth it.
