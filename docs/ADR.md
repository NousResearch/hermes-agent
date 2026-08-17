# Architecture Decision Records

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.

## 2026-08-17: Keep human attention receipts separate from Kanban workflow

Status: Accepted

Context:
Kanban status answers what the workflow is doing (`running`, `blocked`,
`review`, `done`). It does not answer whether a human has seen an item or wants
it hidden until later. Encoding acknowledgement as `done`, or snooze as
`scheduled`, would corrupt dependency gating, dispatch, review, and completion.
Renderer-local state is also insufficient: Desktop and the dashboard can be
open on different devices, expiry must survive restart/offline, and new task
activity must deterministically resurface an acknowledged item.

Decision:
- Store a versioned `attention_receipts` row in the canonical board database,
  keyed by `(subject_kind, subject_id)`. The first supported subject kind is
  `kanban_task`; unknown kinds and unknown task ids fail closed. Standalone
  sessions are deferred until their gateway exposes an equally stable,
  backend-owned event sequence.
- Receipts are a reversible overlay with typed actions: `settle`, `snooze`,
  and `wake`. They contain `state`, `wake_at`, the task-event cursor observed by
  the action, actor/source, revision, and timestamps. Every accepted action is
  appended once to canonical `task_events` as an `attention_*` event; clients
  never submit arbitrary receipt blobs. Replay hashes live in a normalized,
  FK-indexed `attention_idempotency` cache bounded to 64 keys per task and are
  deleted with the task, so replay protection cannot become a second audit log.
- The pure projection policy compares a receipt with canonical time and the
  task's latest event id. Snooze expires at `wake_at`. Any consequential task
  event after the observed cursor resurfaces settled/snoozed attention. Receipt
  events themselves are stored separately and therefore cannot recursively
  trigger resurfacing. Invalid/corrupt state projects active (visible), never
  hidden.
- Writes use optimistic revision checks and an idempotency key. A repeated key
  with identical action content is a no-op; conflicting reuse is rejected.
  Attention endpoints contain no workflow fields and never update `tasks`.
- Board/task responses carry the projected receipt. The existing task-events
  WebSocket invalidates both Desktop React Query and dashboard state, avoiding
  a new polling loop. A bounded client timer only wakes at the nearest
  `wake_at`; restart/offline recovery comes from projection against server time.

UX acceptance contract:
- Active cards expose **Settle** and **Snooze** (1 hour, tomorrow morning, one
  week, or an accessible date/time input). Snoozed cards leave the active view.
- Settled cards recede into a collapsed **Settled** tail, newest first, with a
  visible **Wake** control. This is history, not another workflow column.
- Expiry or consequential activity returns the card to active attention and
  announces the wake in an accessible status region. Every action is available
  by keyboard and touch; no gesture is required.
- Workflow labels and status controls remain unchanged. In particular,
  `review` remains review and `done` remains completion.

Current call graph:
`task_events` / `tasks` (`hermes_cli/kanban_db.py`) → projection and typed action
routes (`plugins/kanban/dashboard/plugin_api.py`) → plugin REST + existing
`/events` invalidation (`apps/desktop/src/plugins/kanban/api.ts`) → board model
and cards (`types.ts`, `board.tsx`). The standalone dashboard follows the same
backend routes from `plugins/kanban/dashboard/dist/index.js`; both clients send
receipts back through the typed action endpoint and receive device-consistent
state on the next event invalidation/refetch.

Consequences:
- Attention can be acknowledged and reversed without ever completing,
  reopening, cancelling, scheduling, or reviewing a task.
- Board DB backup/restore rolls workflow and its receipts together. Rollback of
  this feature is additive: older code ignores the two new tables; dropping
  them removes only attention history, not task data.
- Session attention remains unchanged rather than guessing across runtime,
  stored, and lineage identities.
