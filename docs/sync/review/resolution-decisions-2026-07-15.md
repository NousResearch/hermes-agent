# Resolution decisions - 2026-07-15 upstream parity merge

## Architectural ports

- Desktop retired `desktop-controller*`: kept upstream's contribution-controller architecture and removed the retired controller files. Verified the fork behaviors are represented in the new surface: server-side pinned sessions through `store/layout.ts` + contribution wiring, `desktop.reset_model_on_new_session` through `use-hermes-config`, sidebar/session refresh through `session.changes` polling in contribution wiring, and stored-session resume paths through session action hooks.
- Desktop Windows child-process test: removed `apps/desktop/electron/windows-child-process.test.ts` per upstream replacement. The upstream `windows-hermes-path.ts` and test remain present.
- Desktop pane-shell test: removed the retired `pane-shell.test.tsx` and kept upstream layout-tree replacement structure.
- `tests/run_agent/test_run_agent.py`: kept fork deletion of the monolith. Existing split tests under `tests/run_agent/` were preserved; upstream watchdog/fallback coverage that conflicted elsewhere was retained in the split files.

## Semantic reconciliations

- `gateway/run.py`: interleaved fork safe-restart active-agent keys, model-switch/restart durability, async-delegation outbox acknowledgement states, and synthetic-event safeguards with upstream active-work accounting, shared relay adapter routing, sanitized shared-user prefixes, and completion-delivery de-duplication.
- `hermes_state.py`: unioned fork undo/redo, desktop resume markers, effective-last-active denorm/backfill, and skew history with upstream schema v21 additions: profile routing, model usage rows, fallback compression streak, WAL/session fixes, and async delegation durable delivery.
- `agent/chat_completion_helpers.py`: kept fork relay pool headers and surrogate repair while accepting upstream Anthropic zero-event stream retry normalization.
- `agent/auxiliary_client.py`: kept fork route-scoped Anthropic timeout/no-retry behavior and provider/model resolver safeguards while accepting upstream reasoning and `extra_body` passthrough.
- `agent/tool_executor.py`: kept fork tool-search scope block, pre-tool block, guardrail chain, and session-context mismatch warning in both execution paths while accepting upstream segmented mixed-tool `finalize` support.
- `agent/context_compressor.py` and `agent/conversation_compression.py`: kept fork skew calibration persistence, summary-model reset on route switch, persist-failure signaling, and compaction announce contracts while accepting upstream fallback streak loading, fallback budget accounting, media stripping, and completed-boundary effectiveness verification.
- `cron/scheduler.py`: preserved fork cron ContextVar/cron-mode behavior, per-job reasoning override, and long script timeout behavior; merged upstream run-claim heartbeat and shared per-model reasoning resolver.
- CI workflow/action/classifier: unioned fork `test_scope`, repo-meta/built-asset skips, contributor/fleet gates, and upstream `ci_review`, npm lock diff, JS checks, and CI-sensitive review outputs.
- Desktop/web TypeScript: upstream contribution-controller files were preferred, then fork session search and server-side pinning exports were restored where current components/tests depend on them. Desktop typecheck passes.
- i18n/docs/skills: low-risk docs and skill conflicts were resolved by taking upstream wording where no fork contract was involved; zh desktop locale was repaired to match the merged English key shape.

## Flags for orchestrator

- No unresolved test conflict is intentionally left for follow-up.
- `npm install` was run because the worktree had no `node_modules`; this was required for the desktop typecheck gate.
