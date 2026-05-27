# Hermes Monolith Split — Plan v2 (post upstream-sync)

**Date**: 2026-05-20
**Author**: minsu (jbn7660-hash)
**Versions**:
- v0 = initial draft (superseded; 3 monolith focus, false 5K LOC promise)
- v1 = post plan-gate Critic + Codex review (superseded; same monolith set)
- **v2 = THIS revision** — `run_agent.py` phase **dropped** after discovering upstream already split it on 2026-05-16 (HEAD `5e743559e`). Sibling work: fork sync completed on `minsu/launchd-status-fallback` (HEAD now `25c09ec0c`), uncommitted work preserved in stash `stash@{0}` + diff `/tmp/uncommitted-2026-05-20-snapshot-tracked.diff` + branch ref `backup/minsu-launchd-status-fallback-2026-05-20-pre-sync`. PR #1 (`chore/docs-context-guard`) updated with `4c73eb4a1` reflecting new upstream layout.

**Trigger**: GPT-5.5 (256k) compact loops on hermes-agent fork builds, caused by ~14-18k LOC single files.

---

## What upstream already did (May 2026, do NOT redo)

Upstream NousResearch extracted run_agent.py from ~16,500 → 4,137 LOC across these commits between fork merge-base `f3a4af9cf` and origin HEAD `5e743559e`:

```
9f408989c refactor(run_agent): extract __init__ (1,381 LOC) to agent/agent_init.py
94c3e0ab8 refactor(run_agent): extract 10 more helpers to agent/agent_runtime_helpers.py
053025238 refactor(run_agent): extract run_conversation to agent/conversation_loop.py
47823790b refactor(run_agent): review fixes — keyword-forward __init__, drop dead code, tighten guards
b07524e53 feat(xai-oauth): port to extracted modules
27df24956 feat(nvidia): port to extracted modules
fe4c87eb2 fix(agent): port to extracted modules
…
```

New `agent/*` modules in origin/main:

| Module | LOC | Role |
|---|---|---|
| `run_agent.py` | 4,137 | `AIAgent` class shell + `main()` |
| `agent/agent_init.py` | 1,510 | constructor body |
| `agent/conversation_loop.py` | 4,099 | `run_conversation` |
| `agent/agent_runtime_helpers.py` | 2,158 | 10 helpers |
| `agent/chat_completion_helpers.py` | 2,078 | per-provider chat call helpers |
| `agent/conversation_compression.py` | 605 | summarize-and-trim |
| `agent/codex_runtime.py` | 448 | Codex Responses adapter |
| `agent/iteration_budget.py` | 62 | `IterationBudget` (exactly what v1's plan called `agent/_helpers/budget.py`) |

**Phase 2 of plan v0/v1 (run_agent.py split) is fully discharged by upstream.** Removed from this plan.

## Remaining monolith targets (still need our work)

origin/main HEAD `5e743559e`:

| File | LOC | 256k % | Notes |
|---|---|---|---|
| `gateway/run.py` | 18,205 | 89% | grew slightly from v1 era; still single `GatewayRunner` god class |
| `cli.py` | 14,518 | 71% | unchanged in structure |
| `hermes_cli/main.py` | 13,233 | 65% | grew; argparse + CLI bootstrap |

These three remain in scope. Upstream has NOT split them.

## Honest framing — what each phase actually achieves

| Phase | Target | LOC delta (realistic) | Note |
|---|---|---|---|
| Phase 0 | Prereq scaffolding | n/a | audit + parity tests, no monolith edit |
| Phase 1 | `gateway/run.py` | ~700-900 LOC extracted | preparation for eventual god-class mixin split |
| Phase 2 | `cli.py` | ~1,500-1,800 LOC extracted | cli.py is mostly function-level; largest single-phase delta |
| Phase 3 | `hermes_cli/main.py` | ~600-1,000 LOC extracted | argparse + setup wizard mostly modular already |
| Phase 4 | God-class mixin split (3 classes) | deferred to separate plan | only after Phases 1-3 stable for 14+ days |

**Phase 1-3 reduce each monolith by ~5-12%.** That is NOT the compact-loop solution by itself — the compact-loop fix already landed via PR #1 docs split + the "NEVER full-read" guards in AGENTS.md. Phases 1-3 are prep for Phase 4.

If you accept that framing, continue. If you want a 70%+ LOC reduction now, only Phase 4 delivers it and Phase 4 is high-risk on god-class extraction.

---

## Goal

Reduce each remaining monolith by extracting module-level helper functions into focused `_helpers/` sub-packages, without behavioral change, without prompt-cache invalidation, and with mandatory re-export shims so external consumers (tests, plugins, tui_gateway, cron, tools) keep working.

## Non-goals

- No behavior change. No new features.
- No prompt-cache key change.
- No directory restructure outside the 3 monolith files' own packages.
- `run_agent.py` / `agent/*` — already handled by upstream, out of scope.
- Plugin / skill / website modification.

## Mandatory prerequisites (BLOCKING — Phase 1 cannot start until done)

Same five from v1 (P0 module-global audit, P1 re-export shim, P2 cross-monolith import audit, P3 prompt-cache parity test, P4 module-init snapshot test, P5 phantom doc references), refocused on the 3 remaining monoliths:

| Item | Status | Notes |
|---|---|---|
| `scripts/audit_module_globals.py` | DONE (Phase 0 partial) | check works on `gateway/run.py`, `cli.py`, `hermes_cli/main.py` |
| `scripts/audit_external_imports.py` | DONE (Phase 0 partial) | re-run against new origin/main consumer base |
| `scripts/lint_monolith_growth.py` | DONE (Phase 0 partial) | baseline `.monolith_baseline.json` needs regeneration against origin/main (the existing baseline has `run_agent.py: 4 functions` from outdated worktree — correct for upstream main but confirm) |
| `tests/agent/test_prompt_cache_stability.py` | DONE (Phase 0 partial) | verify it runs against the new agent/* layout (may need import path updates) |
| `tests/refactor/test_module_init_snapshot.py` | **MISSING** — executor dropped this | finish in this session |

## Phase 0 — finish prerequisite scaffolding

| Deliverable | Status |
|---|---|
| `scripts/audit_module_globals.py` | ✓ |
| `scripts/audit_external_imports.py` | ✓ |
| `scripts/lint_monolith_growth.py` + baseline.json | ✓ (baseline matches origin/main) |
| `tests/agent/test_prompt_cache_stability.py` | ✓ (needs verification against new agent/* layout) |
| `tests/refactor/test_module_init_snapshot.py` | ✗ MISSING |
| `tests/refactor/__init__.py` | ✓ |
| `tests/refactor/fixtures/.gitkeep` | ✓ |
| Commit & push | not yet |

**Action**: write `test_module_init_snapshot.py` per the spec in this plan's Phase 0 section, run both parity tests with `--update-fixtures`, then re-run them clean. Commit. Push to fork. Open PR #2.

## Phase 1 — `gateway/run.py` (~18,200 LOC)

Same shape as v1 Phase 1, plus re-audit because origin/main is at HEAD `5e743559e`, ~1,100 LOC larger than v1's reference snapshot:

1. Re-run `scripts/audit_module_globals.py gateway/run.py` to enumerate every cross-boundary module global.
2. Re-run `scripts/audit_external_imports.py gateway.run` to enumerate every consumer importing from `gateway.run` directly. Mandatory re-export list.
3. Build candidate file list (`gateway/_helpers/<topic>.py` per topic — text / time / replay / ssl / env / skills / session / cfg, exact mapping per audit output).
4. Two-commit PR (copy + delete+wire).

## Phase 2 — `cli.py` (~14,500 LOC)

Same audit-then-extract pattern. cli.py is mostly module-level functions already, so it has the largest extraction potential. Already known sub-targets (subject to audit):

- text/strip helpers (`_strip_reasoning_tags`, etc.)
- `load_cli_config` (~425 LOC; imported by `tui_gateway/server.py`)
- worktree helpers
- color/skin helpers
- output history
- `_run_cleanup` — **stays in cli.py** (per v1 C2: entangled with `_cleanup_done` + `_active_agent_ref`).

## Phase 3 — `hermes_cli/main.py` (~13,200 LOC)

This was Non-goals in v1; promoted to Phase 3 in v2 because (a) the 3 remaining monoliths are now the entire workload, (b) Phase 4 god-class split needs main.py reduced too.

Largest extraction candidates (per a quick scan of v1's analysis):
- subcommand handler functions (argparse-tree entries)
- profile resolution helpers
- setup wizard sub-routines

Audit-then-extract, same as Phases 1 and 2.

## Phase 4 — God-class mixin split (DEFERRED)

`GatewayRunner` / `HermesCLI` (and the now-empty-ish `AIAgent` shell on origin/main) → mixin split. Plan separately after Phases 1-3 stable for 14+ days.

---

## Per-PR mechanics (unchanged from v1)

- Two commits per PR: (1) copy moved code into new file, monolith untouched; (2) delete from monolith + import + re-export shim.
- Re-export shims are mandatory and live for the lifetime of at least one minor version.
- Each PR description links the `audit_module_globals.py` + `audit_external_imports.py` output captured before extraction.

## Acceptance (per phase)

Unchanged from v1. The `<= 5,000 LOC` per-file target is reserved for Phase 4 — Phases 1-3 acceptance is "audit + parity tests green, helpers extracted with re-export shims intact, two-commit reviewability proven."

## Risks (unchanged from v1, plus one new entry)

| Risk | Mitigation |
|---|---|
| (v1's 9 risks — module globals, re-exports, prompt cache, init order, circular imports, squash hazards, plugin monkey-patches, naming bikeshed, growth lint) | (v1's mitigations) |
| **NEW: fork divergence from upstream** | Pin upstream sync as a precondition at the start of each phase. Run `git fetch origin main && git merge-base --is-ancestor HEAD origin/main` — if false, abort the phase, do the sync first, regenerate baselines. |
| **NEW: stashed user work `stash@{0}` may conflict with phase changes** | User to drain `stash@{0}` (apply + resolve conflicts against new run_agent.py / gateway/run.py / etc.) BEFORE Phase 1 PR opens. If stash is irrelevant to phase scope, can defer indefinitely. Diff captured at `/tmp/uncommitted-2026-05-20-snapshot-tracked.diff`. |

## Orchestration

- Worktree per phase, branched off latest origin/main, rebased onto origin/main before opening PR.
- One PR per phase, two commits per PR, sequential merging with ≥ one full CI cycle + 48h dogfood between phases.
- Rollback procedure: revert merge, do NOT cherry-pick fixups.

## Codex 5.5 external gate

- Plan-gate: this v2 should be re-fired through `codex-consult` because v1 was reviewed against a different target set (run_agent.py included; that's now removed). Cost is small; v2 is mostly a delta over v1.
- Diff-gate: each phase PR before merge.

## Timeline (revised honest)

| Phase | Effort | Wall time |
|---|---|---|
| Phase 0 finish (`test_module_init_snapshot.py` + commits + PR) | 1-2h | same day |
| Plan-gate Codex revisit on v2 (optional) | 1h | same day |
| Phase 1 (gateway helpers) | 5-7h | 2-3 days |
| Phase 2 (cli helpers) | 7-10h | 3-4 days |
| Phase 3 (hermes_cli/main.py helpers) | 5-8h | 2-3 days |
| Phase 4 (god-class mixin split) | 15-25h | separate sprint, plan again |

**Phases 0-3 total**: ~20-27h focused work, ~10-15 calendar days incl. review + 48h dogfood per phase.

---

## User work parking (stash)

Before fork upstream sync, the following was preserved:

- `stash@{0}` "pre-upstream-sync-2026-05-20: 8 modified + 4 new tool/test files + img/dir artifacts"
  - 8 modified: `agent/context_compressor.py`, `gateway/run.py` (26 LOC), `hermes_cli/config.py`, `run_agent.py` (130 LOC), `tests/agent/test_context_compressor.py`, `tests/run_agent/test_run_agent.py`, `toolsets.py`, `website/docs/user-guide/configuration.md`
  - 4 new: `agent/context_rollover.py`, `tests/agent/test_context_rollover.py`, `tools/external_llm_tools.py`, `tests/tools/test_external_llm_tools.py`
  - Plus dir artifacts: `.omc/`, `.playwright-mcp/`, `memory/`, `stitch_mcid/`, several `mcid_*.png`
- Branch ref backup: `backup/minsu-launchd-status-fallback-2026-05-20-pre-sync`
- Diff snapshot file: `/tmp/uncommitted-2026-05-20-snapshot-tracked.diff` (518 lines)
- Status snapshot file: `/tmp/uncommitted-2026-05-20-status.txt`

**Resolution path** (user discretion):
- The `run_agent.py` 130-LOC stashed change is now against a file that is ~12k LOC smaller. The stashed diff almost certainly references hunks that no longer exist or have moved into `agent/*`. Likely outcomes: (a) the work was already done upstream — drop the stash, (b) the work was unique — recreate on top of the new layout, (c) hybrid — partial cherry-pick.
- Recommended: `git stash show -p stash@{0} > /tmp/stash-2026-05-20-snapshot.diff`, then for each modified file, eyeball the hunks against current origin/main to decide. Don't attempt `git stash pop` blindly — conflicts on `run_agent.py` will be severe.

---

## Status

- [x] v0 → v1 (Critic + Codex plan-gate revisions)
- [x] v1 → v2 (post upstream-sync discovery; run_agent.py phase dropped; user fork synced to origin/main; PR #1 docs commit `4c73eb4a1` reflects new layout)
- [ ] Phase 0 — finish `test_module_init_snapshot.py`
- [ ] Phase 0 — commit + push + PR #2
- [ ] Plan-gate Codex revisit on v2 (optional)
- [ ] User: drain or accept `stash@{0}` 
- [ ] Phase 1 PR opened (gateway/run.py)
- [ ] Phase 1 merged + 48h dogfood
- [ ] Phase 2 PR opened (cli.py)
- [ ] Phase 2 merged + 48h dogfood
- [ ] Phase 3 PR opened (hermes_cli/main.py)
- [ ] Phase 3 merged + 48h dogfood
- [ ] Phase 4 plan v0 written

## v1 → v2 change ledger

| Change | Reason |
|---|---|
| Dropped Phase 2 (run_agent.py extraction) | Upstream did it 2026-05-16; we'd be duplicating ~12k LOC of refactoring |
| Promoted `hermes_cli/main.py` from Non-goals to Phase 3 | Remaining monolith set shrank from 4 to 3; main.py is the natural third target |
| Phase numbering | v1 Phase 1 (gateway) stays Phase 1; v1 Phase 3 (cli) becomes Phase 2; new Phase 3 (main.py) |
| New prerequisite: fork must be synced to upstream before each phase starts | Fork was 589 commits behind on 2026-05-20; this is now a real failure mode |
| Added stash management section | User uncommitted work needs explicit resolution path before phase work begins |
| LOC table updated to origin/main HEAD `5e743559e` snapshot | v1 numbers were against the user's old HEAD |
| Phase 0 partial completion acknowledged | Executor dropped `test_module_init_snapshot.py`; track it explicitly |
| Codex/Critic plan-gate findings preserved | All v1 P0/P1/M1-M5 remain valid for the 3 remaining monoliths |

---

## v2 → v3 addendum (Codex 5.5 v2 plan-gate revisit, GO with conditions)

Plan-gate Codex 5.5 ran a second time against v2 (`/tmp/codex-plan-hermes-monolith-split-v2-2026-05-20.md`). **Verdict: GO** — v1 → v2 delta is sound, dropping `run_agent.py` phase is justified. **One P0 blocker + five P1 + one P2** to fold in before Phase 1 PR opens.

### CDX-V2-P0 — Stash hunks intersect Phase 1 target (`gateway/run.py`)

The preserved user uncommitted work touches `gateway/run.py` (Phase 1's target). Auditing each hunk is **mandatory** before Phase 1.

**Status: AUDITED 2026-05-20.** Pure stash hunks isolated against the wip-branch base:

- `gateway/run.py` (50 diff lines, two distinct intents):
  1. **launchd service detection** at `GatewayRunner` ~L8956 — extends `_under_service` from systemd-only (`INVOCATION_ID`) to also recognize launchd (`LAUNCH_JOBKEY_LABEL`, `XPC_SERVICE_NAME`). User's macOS gateway fix.
  2. **`last_prompt_tokens` hydration** at `GatewayRunner` ~L15689 — pre-loop seeding of `context_compressor.last_prompt_tokens` from persisted `SessionEntry` on fresh agent construction. User's rollover-preflight correctness fix.
- `run_agent.py` (164 diff lines, two distinct intents):
  1. **`ContextRolloverConfig` load** at `AIAgent.__init__` ~L2368 — wires the new `agent/context_rollover.py` module's config object into the agent. **After upstream split, this hunk belongs in `agent/agent_init.py`, not `run_agent.py`.** Replay needs manual placement.
  2. **Compression-summary warning text** at ~L10713 — `"Inserted a fallback context marker."` → `"Inserted a deterministic extractive fallback summary."`. **After upstream split, this hunk belongs in `agent/conversation_loop.py` or `agent/conversation_compression.py`** (which one — check during replay).

Plus net-new files (untracked, no replay conflict): `agent/context_rollover.py`, `tests/agent/test_context_rollover.py`, `tools/external_llm_tools.py`, `tests/tools/test_external_llm_tools.py`.

Plus modified files that replay clean against new origin/main (no upstream split): `agent/context_compressor.py`, `hermes_cli/config.py`, `tests/agent/test_context_compressor.py`, `tests/run_agent/test_run_agent.py`, `toolsets.py`, `website/docs/user-guide/configuration.md`.

**Phase 1 unblock condition**: user must replay the `gateway/run.py` launchd-detect + hydration hunks onto current origin/main (line numbers will shift; intent transfers cleanly), commit them on `minsu/launchd-status-fallback` or a successor branch. Replay is **user-driven** — automation would risk dropping intent.

Snapshots preserved at:
- `/tmp/wip-stash-gateway-run-pure-2026-05-20.diff` (50 lines, gateway hunks only)
- `/tmp/wip-stash-run-agent-pure-2026-05-20.diff` (164 lines, run_agent hunks only)
- `/tmp/uncommitted-2026-05-20-snapshot-tracked.diff` (518 lines, all 8 modified files)
- Branch `wip/stash-2026-05-20-uncommitted` (working tree carries the full applied stash)

### CDX-V2-P1-1 — Retire stale `run_agent.py` gates

Plan v2 still references `run_agent.py` in Risk + Acceptance + Per-PR mechanics. Sweep the body so no audit step blocks on a file that's already split upstream.

**Status**: tracked; plan body sweep deferred (informational drift, not execution drift).

### CDX-V2-P1-2 — Expand init-snapshot to new `agent/*` modules

After the upstream split, the behaviorally-loaded imports moved to `agent/agent_init.py`, `agent/conversation_loop.py`, `agent/agent_runtime_helpers.py`. Snapshot those too.

**Status: DONE** — `MONOLITHS` list in `tests/refactor/test_module_init_snapshot.py` extended; baseline regenerated and verified.

### CDX-V2-P1-3 — Pre-Phase-3 mini-audit for `hermes_cli/main.py`

Phase 3 acceptance must require the `audit_module_globals.py` + `audit_external_imports.py` output for `hermes_cli/main.py` to be attached to the Phase 3 PR description.

### CDX-V2-P1-4 — Regenerate baselines post-sync

**Status: DONE** — fixtures captured against post-sync layout (HEAD `5e743559e`), `5/5 passed` on no-flag re-run.

### CDX-V2-P1-5 — Replace calendar-based Phase 4 readiness gate

Plan v2 says "tests stay green for 2 weeks on main." Replace with concrete evidence: zero revert-tagged commits on `gateway/` / `agent/` / `hermes_cli/` paths since the prior phase merge, OR ≥ N consecutive green CI runs touching the extracted helpers.

### CDX-V2-P2 — Move durable recovery state from `/tmp` into repo

Move `/tmp/uncommitted-2026-05-20-*.diff` and recovery snapshots into `.omc/recovery/<topic>/<date>.diff` (or repo-relative equivalent) so they survive reboot. Non-blocking for Phase 1; do before Phase 3 ships.

---

## Action queue (post v3 addendum)

1. **User**: replay `gateway/run.py` launchd-detect + hydration hunks onto current `origin/main`, commit on `minsu/launchd-status-fallback`. Unblocks Phase 1 audit.
2. **User**: replay `run_agent.py` `ContextRolloverConfig` hunk onto `agent/agent_init.py`. Replay compression-summary warning hunk onto upstream's relocated emit site (likely `agent/conversation_loop.py` per upstream commit `053025238` — verify).
3. **User**: replay 6 non-conflicting modified files (`context_compressor.py`, `hermes_cli/config.py`, `tests/agent/test_context_compressor.py`, `tests/run_agent/test_run_agent.py`, `toolsets.py`, `website/docs/user-guide/configuration.md`). Add 4 untracked new files (`agent/context_rollover.py`, `tests/agent/test_context_rollover.py`, `tools/external_llm_tools.py`, `tests/tools/test_external_llm_tools.py`). Commit.
4. **Automation**: re-run `scripts/lint_monolith_growth.py` after user replay to confirm no new top-level funcs slipped into the monoliths.
5. **Automation**: open Phase 1 PR (`gateway/run.py`).
6. **Plan body cleanup** (CDX-V2-P1-1): sweep stale `run_agent.py` mentions into the "already-split" sub-block.
