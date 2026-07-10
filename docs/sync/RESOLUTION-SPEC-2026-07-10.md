# Upstream parity merge — conflict resolution spec (2026-07-10)

You are resolving a large `git merge` of upstream `NousResearch/hermes-agent` into our long-lived
hard fork `Kyzcreig/hermes-agent` (`fork/main`). The merge is ALREADY STAGED in this worktree
(`git status` shows 56 unmerged files). **DO NOT run `git merge` or `git merge --abort`.** Your job is
to resolve the conflicts, leave the tree marker-free and importable, and STOP (do not commit — the
orchestrator commits + verifies).

## Frozen facts
- Worktree: `/Users/alexgierczyk/.hermes/worktrees/parity-2026-07-10`, branch `sync/upstream-2026-07-10`.
- Upstream target SHA (frozen): `caf557be5b4c9ae75b3a7566d65d3df2c701c5df`.
- Merge-base: `852c9b3cb2ce2f00a2403434a84fdbd7ebf95fda`.
- Fork is ~386 ahead / ~1216 behind upstream. This is the biggest ingest in the fork's life.
- Python for import-smoke: `PYTHONPATH=$PWD /Users/alexgierczyk/.hermes/runtime/hermes-agent/venv/bin/python3`

## The prime directive
**The fork's 386 lead-commits are LIVE on Ace's production fleet.** A blind side-pick that drops a
fork feature is a production outage. For EVERY both-sides conflict, read BOTH sides' history before
deciding:
```
git log -p 852c9b3cb2ce2f00a2403434a84fdbd7ebf95fda..caf557be -- <file>   # what upstream did
git log -p 852c9b3cb2ce2f00a2403434a84fdbd7ebf95fda..fork/main -- <file>  # what the fork did
```
Resolution shapes (pick per-hunk on merit, never by size):
- **keep-both / union** — independent additions on each side → include both.
- **interleave** — both restructured the same block with complementary guards → merge the logic so
  both intents survive.
- **architectural port** — upstream relocated/extracted the host file → re-thread the fork's behavior
  onto upstream's NEW structure. NEVER paste stale fork bodies back into a moved module.
- **take-fork** — a fleet-tuned value/feature upstream lacks.
- **take-upstream** — a real correctness fix the fork lacks.

## Known semantic traps (READ BOTH SIDES — do not blind-merge these)
1. **`plugins/memory/mem0/`** — the big one. Fork has a self-contained ~1653-line `__init__.py`
   client. Upstream refactored into `__init__.py` (574 lines) delegating to `_backend.py` / `_setup.py`
   (both are `DU` = fork-deleted / upstream-modified). **Take the fork's superset wholesale:**
   `git checkout --ours plugins/memory/mem0/__init__.py`, then `git rm` upstream's orphaned
   `_backend.py` `_setup.py` and their tests `tests/plugins/memory/test_mem0_backend.py`
   `test_mem0_setup.py` `test_mem0_v3.py` (they import symbols the fork doesn't export). Confirm the
   fork `__init__.py` still imports clean. Record the call in `docs/sync/review/mem0-resolution.md`.
   BUT: if upstream added a genuinely NEW capability in `_backend`/`_setup` that the fork lacks, note
   it in the review file for the orchestrator — do not silently lose a real upstream feature.
2. **`toolsets.py` is NOT in conflict but is a landmine** — the fork has custom `messaging`
   (`send_message`) and `moa` (`mixture_of_agents`) toolsets upstream lacks (documented in-file around
   line 344). If ANY conflict resolution touches toolset wiring, those MUST survive. The orchestrator
   runs a config-migration dry-run to catch a drop — but don't cause one.
3. **`agent/auxiliary_client.py` — `_resolve_task_provider_model` api_mode.** Upstream's contract:
   returns `api_mode=None` for first-class providers (transport corrected downstream). Fork commit
   eagerly reads `ProviderProfile.api_mode` so PLUGIN providers on a plain base_url get the right wire.
   RECONCILE (satisfies both): gate the fork's eager profile-read to plugin providers only — skip when
   provider is `""`/`auto`/`custom`/`custom:*` OR first-class. First-class + custom → `None`; plugins →
   fork fix.
4. **subagent `code_execution`** — fork deliberately UN-blocks `code_execution` for subagents;
   upstream's blocklist-derived strip re-blocks it. Keep the fork's exemption (hybrid:
   derive-from-blocklist + explicit exemption).
5. **`agent/chat_completion_helpers.py`** — fork has the relay pool header stamping
   (`x-hermes-session`, `x-hermes-lane`, `x-hermes-lane-src`, background-lane classifier). MUST survive.
6. **`gateway/session.py` async session DB** — upstream migrated to `AsyncSessionDB(SessionDB())`
   whose `__getattr__` wraps callables as `asyncio.to_thread`. Any `self._session_db.<m>(` call on the
   gateway loop MUST be a literal `await`. An AST guard test enforces this.
7. **`get_messages_as_conversation` / timestamp gate** — fork keeps `include_timestamp` opt-in
   (default no timestamp), confirmed by prod callers passing `include_timestamp=True`. Upstream added
   timestamps unconditionally. Keep the fork's gate.
8. **`agent/model_metadata.py`, `cron/scheduler.py`, `gateway/run.py`, `tui_gateway/server.py`,
   `run_agent.py`, `hermes_state.py`** — heavy fork customization (LCM/compaction, reset-weighted
   router, model-switch announce, undo/redo half-turn, denorm session list). Read both sides per hunk.

## Test files in conflict
Tests encode contracts. When a test conflicts, resolve to the MERGED contract:
- fork deliberately changed a behavior → keep the fork's test assertion, update/drop upstream's.
- upstream added a real new test for a real new behavior the fork keeps → keep it.
- two unioned tests encode OPPOSITE contracts → pick the fleet's contract, drop/adjust the other.
Note any test you can't cleanly adjudicate in `docs/sync/review/test-conflicts.md` for the orchestrator.

## Desktop app (`apps/desktop/*`) conflicts
TypeScript/Electron. Same rule: read both sides, keep-both where independent, port where upstream
moved things. These are lower fleet-risk than the Python runtime but must still compile.

## Definition of done (your stop condition)
1. ZERO conflict markers anywhere — verify ALL THREE marker types:
   `git grep -nE '^(<<<<<<<|=======|>>>>>>>)'` returns nothing.
2. Every resolved `.py` passes `python -m py_compile`.
3. Import-smoke passes for the fleet-critical modules:
   `PYTHONPATH=$PWD <venv> -c "import run_agent, gateway.run, agent.conversation_loop, agent.auxiliary_client, agent.chat_completion_helpers, cron.scheduler, hermes_state, tui_gateway.server; import importlib; importlib.import_module('plugins.memory.mem0'); print('IMPORT-SMOKE OK')"`
4. `git status` shows no remaining `UU`/`DU`/`AA` unmerged entries.
5. Write `docs/sync/review/resolution-decisions.md` documenting every architectural-port and
   semantic-reconciliation decision (which shape, why), plus anything you flagged for the orchestrator.
6. **DO NOT commit. DO NOT push. STOP** — the orchestrator commits, runs the pytest gates + CI-gate
   traps + config-migration dry-run, and lands the PR.

Work in RISK ORDER (biggest/most-fleet-critical first): mem0 → gateway/run.py → tui_gateway/server.py →
cron/scheduler.py → hermes_state.py → agent/* → gateway/* → tools/* → tests → desktop.
