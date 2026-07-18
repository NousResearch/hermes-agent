# Kanban test isolation inventory

Every mutation-capable Kanban test must fail **closed**: before any code
resolves a Kanban path, all inherited Kanban pins are cleared, a fresh
per-test temporary home/root is set, and the run aborts unless BOTH the
resolved Kanban home and the resolved `kanban.db` (plus the workspaces,
attachments, and worker-logs roots) live beneath that temp root. This stops
a stray `HERMES_KANBAN_DB` / `HERMES_KANBAN_HOME` / `HERMES_KANBAN_BOARD` /
`HERMES_KANBAN_ATTACHMENTS_ROOT` … pin — leaked from a developer shell or a
dispatched worker — from routing a unit-test mutation at a live board.

## The shared guard

Two cooperating layers, both in `tests/conftest.py`:

1. **Layer 1 — `_hermetic_environment` (global autouse).** Blanks every
   inherited Kanban pin (path pins *and* the behavioral `HERMES_KANBAN_GOAL_MODE`
   / `HERMES_KANBAN_GOAL_MAX_TURNS` pins the dispatcher injects) and points
   `HERMES_HOME` at a per-test tempdir. Runs before any test under both
   direct `pytest` and `scripts/run_tests.sh` (which additionally uses
   `env -i`).
2. **Layer 2 — `_isolate_kanban_root`**, exposed as the **`isolate_kanban_root`**
   fixture (a callable `isolate_kanban_root(tmp_path, monkeypatch, *, home=None)`).
   The fixture-local, defense-in-depth layer: re-clears every inherited Kanban
   pin, sets `HERMES_HOME`/`Path.home` under `tmp_path`, resets the per-process
   init cache, then asserts containment of home + `kanban.db` + workspaces +
   attachments + worker-logs roots — raising `AssertionError` (fail closed)
   otherwise.

Contract test: `tests/hermes_cli/test_kanban_isolation_guard.py` (asserts
behavior — resolved paths / raising — never source text).

## Mutation-capable modules wired to `isolate_kanban_root`

Each module's shared mutation setup fixture (for example, `kanban_home`,
`fresh_home`, `worker_env`, or `isolated_kanban_home`) routes through the
guard where it owns a common mutation path. Modules with deliberately inline
temporary setup are called out below rather than being incorrectly described
as using one shared fixture for every test.

| Category | Module | Setup fixture |
| --- | --- | --- |
| CLI | `tests/hermes_cli/test_kanban_cli.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_cli_dispatch_passthrough.py` | `isolated_kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_core_functionality.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_boards.py` | `fresh_home` |
| CLI | `tests/hermes_cli/test_kanban_db.py` | `kanban_home` + poison fixture |
| CLI | `tests/hermes_cli/test_kanban_decompose.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_decompose_db.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_default_assignee.py` | `isolated_kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_per_profile_cap.py` | `isolated_kanban_home_with_profiles` |
| CLI | `tests/hermes_cli/test_kanban_diagnostics.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_notify.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_promote.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_specify.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_specify_db.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_block_kinds.py` | `kanban_home` |
| CLI | `tests/hermes_cli/test_kanban_worker_image_extraction.py` | `kanban_home` |
| Dispatch lock | `tests/hermes_cli/test_kanban_dispatch_lock.py` | `kanban_home` |
| Blocked sticky | `tests/hermes_cli/test_kanban_blocked_sticky.py` | `kanban_home` |
| Goal mode | `tests/hermes_cli/test_kanban_goal_mode.py` | `kanban_home` + legacy-migration test |
| Lifecycle | `tests/hermes_cli/test_kanban_lifecycle_hooks.py` | `kanban_home` |
| Reclaim/claim lock | `tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py` | `kanban_home` |
| Stress (init lock) | `tests/hermes_cli/test_kanban_init_lock_bounded.py` | `kanban_home` |
| Plugins | `tests/plugins/test_kanban_attachments.py` | `kanban_home` |
| Plugins | `tests/plugins/test_kanban_dashboard_plugin.py` | `kanban_home` |
| Plugins | `tests/plugins/test_kanban_worker_runs.py` | `kanban_home` |
| Tools | `tests/tools/test_kanban_tools.py` | `worker_env`; `orchestrator_env`; `_make_goal_mode_worker_env`; direct goal-mode setup calls; `_orchestrator_env_with_leaked_attachments_pin` verifies attachment-root containment before it triggers `orchestrator_env` |
| Tools | `tests/tools/test_kanban_redaction.py` | `worker_env` |

## Covered by Layer 1 only (intentionally not wired to the fixture guard)

These paths are still fail-safe — Layer 1 blanks every inherited Kanban pin
before they run — but they are **not** routed through the `isolate_kanban_root`
fixture, by design:

- **Resolution-behavior tests** in `test_kanban_db.py` / `test_kanban_boards.py`
  that deliberately set `HERMES_KANBAN_HOME` / `HERMES_KANBAN_DB` /
  `HERMES_KANBAN_WORKSPACES_ROOT` / `HERMES_KANBAN_BOARD` to exercise the
  resolver. They opt out of the strict containment assertion on purpose and
  pin their own temp locations.
- **Gateway notifier** `tests/gateway/test_kanban_notifier.py`: each test pins
  `HERMES_KANBAN_DB` to a per-test file under `tmp_path` inline; wiring the
  fixture guard would fight that explicit per-test pin. Safe via the tmp_path
  pin + Layer 1.
- **Inline per-test homes** in `test_kanban_db_init.py`, `test_kanban_worker_spawn_toolsets.py`,
  `test_kanban_worker_terminal_cwd.py`, and a handful of one-off setups inside
  `test_kanban_core_functionality.py` set `HERMES_HOME` under their own
  `tmp_path` directly (no shared fixture to wire).
- **Tools module resolver/schema and board-routing setups** in
  `test_kanban_tools.py`: `multi_board_env`, the schema-visibility tests, and
  small inline board/task setups intentionally exercise explicit board or task
  resolution. They use temporary paths or Layer 1 and are not represented as
  shared fixture-guard coverage. `allow_private_urls` and `default_url_guard`
  only control URL safety and do not create Kanban storage.
- **`test_kanban_write_txn_busy_retry.py`**: uses in-memory `_FakeConn`
  objects, never touches an on-disk `kanban.db`; the guard is not applicable.
