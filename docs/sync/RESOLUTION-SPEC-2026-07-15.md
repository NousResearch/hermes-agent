# Upstream parity merge — conflict resolution spec (2026-07-15)

You are resolving a large `git merge` of upstream `NousResearch/hermes-agent` into our long-lived
hard fork `Kyzcreig/hermes-agent` (`fork/main`). The merge is ALREADY STAGED in this worktree
(`git status` shows 81 unmerged files). **DO NOT run `git merge` or `git merge --abort`.** Your job is
to resolve the conflicts, leave the tree marker-free and importable, and STOP (do not commit — the
orchestrator commits + verifies).

## Frozen facts
- Worktree: `/Users/alexgierczyk/.hermes/worktrees/parity-2026-07-15`, branch `sync/upstream-2026-07-15`.
- Upstream target SHA (frozen): `2ea39daeb1f675d72e5c21c9400f2d58d7e6d71a`.
- Merge-base: `caf557be5b4c9ae75b3a7566d65d3df2c701c5df` (= the 2026-07-10 sync's frozen target).
- Fork is ~529 ahead / ~581 behind upstream.
- Python for import-smoke: `PYTHONPATH=$PWD /Users/alexgierczyk/.hermes/runtime/hermes-agent/venv/bin/python3`

## The prime directive
**The fork's 529 lead-commits are LIVE on Ace's production fleet.** A blind side-pick that drops a
fork feature is a production outage. For EVERY both-sides conflict, read BOTH sides' history before
deciding:
```
git log -p caf557be5b4c9ae75b3a7566d65d3df2c701c5df..2ea39daeb -- <file>   # what upstream did
git log -p caf557be5b4c9ae75b3a7566d65d3df2c701c5df..fork/main -- <file>   # what the fork did
```
Resolution shapes (pick per-hunk on merit, never by size):
- **keep-both / union** — independent additions on each side → include both.
- **interleave** — both restructured the same block with complementary guards → merge the logic so
  both intents survive.
- **architectural port** — upstream relocated/extracted the host file → re-thread the fork's behavior
  onto upstream's NEW structure. NEVER paste stale fork bodies back into a moved module.
- **take-fork** — a fleet-tuned value/feature upstream lacks.
- **take-upstream** — a real correctness fix the fork lacks.

## Registered fork features that MUST survive (docs/sync/fork-features.json)
Each has a CI test that goes red if dropped. Verify each survives your resolution:
1. cron-subagent approval gate reads ContextVar not raw env.
2. systemd restart exits 0, darwin/launchd exits 75.
3. hygiene compaction announces in-chat on success.
4. `messaging` + `moa` fork toolsets present (toolsets.py — a landmine even when not in conflict).
5. telegram polling never drops pending updates on recovery.

## Known semantic traps (READ BOTH SIDES — do not blind-merge these)
1. **`agent/chat_completion_helpers.py` (6 hunks)** — fork has the relay pool header stamping
   (`x-hermes-session`, `x-hermes-lane`, `x-hermes-lane-src`, background-lane classifier). MUST survive.
2. **`agent/auxiliary_client.py`** — fork's `_resolve_task_provider_model` gates the eager
   `ProviderProfile.api_mode` read to PLUGIN providers only (first-class/custom → `None`). Keep that
   reconciliation; don't let upstream's version regress it.
3. **`agent/tool_executor.py` (6 hunks)** — fork has the shared block-seam chain
   (`_ts_scope_block` → pre-tool-call block → guardrails) in BOTH concurrent and sequential paths,
   plus the subagent `code_execution` exemption. All must survive.
4. **`agent/context_compressor.py` (9 hunks) / `agent/conversation_compression.py`** — heavy fork
   LCM/compaction customization (per-model thresholds, in-place compaction, hygiene compaction
   announce). Read both sides per hunk; upstream correctness fixes are welcome but fork contracts win
   where they collide (the fork's tests encode them).
5. **`gateway/run.py` (27 hunks)** — the biggest file. Fork: model-switch announce (symmetric
   to/from + effort/context deltas), undo/redo half-turn, restart/resume machinery, denorm session
   list, config→env bridge map. Upstream target commit `2ea39daeb` adds shared relay adapter in
   multiplex mode — that's a real upstream feature, keep it.
6. **`hermes_state.py` (24 hunks)** — fork: undo/redo stacks, denorm session columns, AsyncSessionDB
   literal-await contract (AST guard test enforces `await` on every gateway-loop `_session_db` call).
7. **`cron/scheduler.py` (12 hunks)** — fork: cron_mode deny, script timeout 7200, no_agent lane.
8. **`get_messages_as_conversation` timestamp gate** — fork keeps `include_timestamp` opt-in.
9. **`.github/workflows/ci.yml` + `.github/actions/detect-changes/action.yml` +
   `scripts/ci/classify_changes.py`** — fork CI has extra gates (contributor-check AUTHOR_MAP,
   gitleaks pin, config-migration dry-run). Union carefully; don't drop fork gates.
10. **`scripts/release.py`** — the conflict is almost certainly in AUTHOR_MAP; union both sides'
    entries (fork has fleet identities like apollo@daemonarchy.local that MUST stay).

## Pre-adjudicated ARCH-SPLIT files (researched by the orchestrator — follow these)
- **`apps/desktop/electron/windows-child-process.test.ts` (UD)** — upstream `727025a2f` DELETED it,
  replacing the regex tests with real tests in `windows-hermes-path.{ts,test.ts}` (+ extracted logic
  from `main.ts`). Take upstream's deletion (`git rm`); confirm `windows-hermes-path.test.ts` exists
  post-merge. If the fork's `1171388d5` TS-harness fixes touched this file, check whether the same
  fix class is needed in upstream's replacement test file.
- **`apps/desktop/src/app/desktop-controller.tsx` + `desktop-controller-utils.ts` (UD)** — upstream
  `369d0eeef` RETIRED desktop-controller for the contribution surface (`0f922002e` adds the
  contribution controller). This is an architectural port: the fork's behaviors in those files —
  server-side pinned sessions (#186), `desktop.reset_model_on_new_session` flag, cross-machine
  session-list sync (sidebar refresh), session.changes poll using the STORED session id — must be
  RE-THREADED onto the new contribution controller/surfaces, not pasted back. Read the fork commits
  (`0df54d76f`, `77b9fc1e9`, `d9ce140b4`, `830855809`) and find where each behavior now belongs in
  upstream's new structure. If a behavior already exists upstream in the new shape, take upstream.
  Document every port decision.
- **`apps/desktop/src/components/pane-shell/pane-shell.test.tsx` (UD)** — upstream `63a9bde77`
  (layout-tree renderer) deleted/replaced it. Take upstream's structure; port the fork's ambient
  test-harness fixes only if the replacement suite needs them.
- **`tests/run_agent/test_run_agent.py` (DU)** — the FORK deleted it (`8f7c231c7` split the monolith
  into per-issue files under `tests/run_agent/`); upstream updated the monolith. Keep the fork's
  split (git rm the monolith), then PORT any NEW/CHANGED upstream tests from
  `git diff caf557be..2ea39daeb -- tests/run_agent/test_run_agent.py` into the matching split files
  (or a new appropriately-named split file). Do not lose upstream's new test coverage.

## Test files in conflict
Tests encode contracts. When a test conflicts, resolve to the MERGED contract:
- fork deliberately changed a behavior → keep the fork's test assertion, update/drop upstream's.
- upstream added a real new test for a real new behavior the fork keeps → keep it.
- two unioned tests encode OPPOSITE contracts → pick the fleet's contract, drop/adjust the other.
Note any test you can't cleanly adjudicate in `docs/sync/review/test-conflicts.md` for the orchestrator.

## Desktop app (`apps/desktop/*`) + web (`web/src/*`) conflicts
TypeScript. Same rules. After resolving, run `cd apps/desktop && npm run typecheck` — the desktop
TS surface is a second verification tier the Python gates never touch. i18n files (`ja.ts`,
`zh.ts`, `zh-hant.ts`): union both sides' keys; keep key order consistent with `en`.

## Definition of done (your stop condition)
1. ZERO conflict markers anywhere — verify ALL THREE marker types:
   `git grep -nE '^(<<<<<<<|=======|>>>>>>>)'` returns nothing.
2. Every resolved `.py` passes `python -m py_compile`.
3. Import-smoke passes for the fleet-critical modules:
   `PYTHONPATH=$PWD <venv> -c "import run_agent, gateway.run, agent.conversation_loop, agent.auxiliary_client, agent.chat_completion_helpers, agent.tool_executor, agent.context_compressor, cron.scheduler, hermes_state, tui_gateway.server; print('IMPORT-SMOKE OK')"`
4. `git status` shows no remaining `UU`/`DU`/`UD`/`AA` unmerged entries (resolved deletions staged
   via `git rm`, resolved keeps via `git add`).
5. `cd apps/desktop && npm run typecheck` passes (or document exactly what fails and why it's
   pre-existing).
6. Write `docs/sync/review/resolution-decisions-2026-07-15.md` documenting every architectural-port
   and semantic-reconciliation decision (which shape, why), plus anything flagged for the orchestrator.
7. **DO NOT commit. DO NOT push. STOP** — the orchestrator commits, runs the pytest gates + CI-gate
   traps + config-migration dry-run, and lands the PR.

Work in RISK ORDER (biggest/most-fleet-critical first): gateway/run.py → hermes_state.py →
agent/* → cron/scheduler.py → tui_gateway/server.py → gateway/* → tools/* → hermes_cli/* →
CI workflow files → tests → desktop/web → i18n/docs/skills.
