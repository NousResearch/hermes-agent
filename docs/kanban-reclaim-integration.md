# Kanban reclaim integration for #84258

This integrates the systemd-scope work with current main and strengthens the
ownership checks around every attempt. It is an integration candidate, not a
claim that the complete execution-ownership issue is closed.

## Source and contributor history

The integration preserves these commits as parents, with their original history:

| Source | Exact commit | Role |
|---|---|---|
| Dan Ashburn, PR #101911 | `b75c4a6f899dbb09f08e384c76a31d15c6998a49` | Scope isolation, registration, stop service, restart adoption and handoffs; 65 contributor commits retained |
| Current main | `de2d6a1b93508463c31434c1ae067e204af81238` | Current schema, module topology, shared restart-safe launch, secure user-bus environment, configuration recovery and acceptance behavior |
| KoNit-K, PR #107723 | `351850023f248a321c3efd7860285f5f905e9571` | Expected-run termination API and dashboard precondition, integrated with stronger reservation checks |

The original common base for the scope integration is
`63279301bcbdc185c1b07b98a9312eb0c862f26d`. Integration is appended to contributor
history; contributor commits are not recreated under the integrator's name.

Related work was evaluated by responsibility, beyond the issue comment:
#104125 (namespace identity), #95106 (process groups), #91162 (survivor policy),
#91554 (progress detection), #71259 and its predecessor #63073 (restart-safe
same-user launch), #62668 (restart-safe completion), and #108162 (authority
history). These are context, not additional imported implementations. Main's
shared launch implementation remains the authority. Existing extraction PRs
#79893, #79895, #79896 and #79897 overlap the module decomposition here; this
candidate does not independently close or replace their discussions.

## Behavior changes

- Heartbeat, registration, dispatch publication, reclaim and delayed handoffs
  check the immutable run ID, not a reusable claim string alone.
- A reclaim reserves a complete execution snapshot before signaling outside the
  SQLite write lock. Final settlement checks the same reservation and identity;
  failed stops retain ownership. Retry accounting commits with settlement.
- Nested transactions defer scope stops until durable outer commit. Rollback
  discards stop intents. Raw caller transactions cannot enqueue irreversible
  scope stops without an observable commit boundary.
- Scope names include canonical board-path identity. Recursive cgroup population
  is required for quiescence; a missing systemd unit alone does not prove death.
- Restart adoption distinguishes live reservations, expired controller ownership,
  pending handoffs and stop intents. Parked scopes prevent fresh claims until a
  verified sweep clears them.
- Board export and import strip local worker ownership from both tasks and runs.
  CLI, dashboard, tool heartbeat, gateway and cron integrations use the defining
  modules. Public compatibility entrypoints remain available.
- Large database, dispatch, process and test modules are split by responsibility;
  changed Python files stay below 2,000 lines.

The ownership map and transaction contract are in
`website/docs/developer-guide/cli-internals.md`.

## Validation

Independent witnesses reproduced stale-heartbeat successor mutation and manual
reclaim run/PID/heartbeat races on main `d595e636c83aa0b9606d4e914e1140ae9c796897`.
The subsequent main commit changes configuration recovery, not these paths.
A separate witness reproduced premature scope-stop effects inside a raw outer
transaction on the unmodified scope carrier. The integration passes those
regressions. Board-transfer and recursive/missing-unit cgroup witnesses also
failed before their corresponding fixes and passed afterward.

The canonical runner completed the focused integration set: **315 passed,
0 failed, 7 platform skips across 19 files**. Public compatibility tests were
then rerun unchanged from main: **2 passed**. Repository blocking Ruff checks,
the 2,119-entry internal compatibility dependency check, independent cold imports
of the changed owners, and whitespace checks passed.

Reproduce the focused set with the repository's development environment:

```sh
scripts/run_tests.sh \
  tests/hermes_cli/test_kanban_reclaim_generation.py \
  tests/hermes_cli/test_kanban_reclaim_transaction.py \
  tests/hermes_cli/test_kanban_dispatch_generation.py \
  tests/hermes_cli/test_kanban_adoption_reservations.py \
  tests/hermes_cli/test_kanban_parked_scope_transitions.py \
  tests/hermes_cli/test_kanban_transfer.py \
  tests/hermes_cli/test_kanban_db.py \
  tests/hermes_cli/test_kanban_review_lifecycle.py \
  tests/hermes_cli/test_kanban_review_lifecycle_complete.py \
  tests/hermes_cli/test_kanban_parent_reopen_invalidation.py \
  tests/hermes_cli/test_kanban_gateway_restart_handoff.py \
  tests/plugins/test_kanban_transition_generation.py \
  tests/plugins/test_kanban_worker_runs.py \
  tests/tools/test_kanban_generation_surfaces.py \
  tests/tools/test_kanban_tools.py \
  tests/cron/test_cron_kanban_env_isolation.py \
  tests/cron/test_restart_safe_worker.py \
  tests/tools/test_process_registry.py \
  tests/test_compat_manifest_targets.py -j4 --tb=short
```

Additional targeted runs cover dashboard, gateway, transfer, scope probes and
spawn/restart behavior. These are not a full-suite or native-systemd CI claim.
The managed validation runtime cannot expose subprocess PIDs normally and
rejects Unix-domain sockets. Broader process tests hit the same PID-visibility
guard failures reproduced on main, and carrier cancellation tests hit that
restriction too. The live-system guard was not disabled. All 106 carrier scope
tests were retained across the six `test_kanban_scope_*.py` files and shared
support; their native process behavior still requires a suitable runner.

Before readiness, run the scope files, process lifecycle/scope files, remaining
Kanban neighbors and platform lanes through `scripts/run_tests.sh` on hosts with
normal process visibility, including a working systemd user manager on Linux.

## Remaining execution-ownership boundaries

The existing non-systemd/PID-only fallback remains supported. A dead root PID
does not prove that descendants stopped; a detached-child witness confirms that
limitation. Recursive scope quiescence strengthens the scoped path only.
Persisted boot, manager and namespace provenance are not implemented here.
Filesystem, VCS, publication and other external effects still need their own
reconciliation contract. Terminal transitions retain scope history on runs while
asynchronous cleanup completes. None of those facts should be translated into
an unconditional "safe to retry" or a full closure of #84258.
