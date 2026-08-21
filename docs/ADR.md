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
## 2026-08-20: Strict-readonly Kanban worker capability (workspace confinement)

Status: Accepted

### Context

Kanban workers can be dispatched autonomously to act on a task. Some
workflows require an autonomous worker that is **explicitly confined
to its own task workspace** — a worker that can write files but cannot
escape that workspace, modify the Hermes repository, reach the user's
profile, or interact with arbitrary external systems.

Without a strict capability, an autonomous worker that runs in a
workspace shell could mutate the repository under itself, escalate by
writing profile config, traverse symlinks to escape the workspace,
or open a shell that bypasses the file-tool gate entirely. The
narrow solution is a strict-readonly capability that activates only
when the dispatcher explicitly opts in.

### Decision

Persist an explicit `strict_readonly` capability on the Kanban task
row (`INTEGER NOT NULL DEFAULT 0`). Provenance is NOT capability:
`created_by='agent'` does NOT imply strict mode.

The capability travels via two channels:

1. **Database column** `tasks.strict_readonly`. Read by
   `hermes_cli/kanban_db._default_spawn` when constructing the
   worker environment.
2. **Environment variable** `HERMES_KANBAN_STRICT_READONLY=1`,
   observed by `tools.file_tools._is_strict_readonly_active`. Set
   ONLY by the dispatcher when the task row opts in.

When the task opts in, the dispatcher (`_default_spawn`):

- Pins `HERMES_KANBAN_STRICT_READONLY=1`.
- Pins `HERMES_KANBAN_TASK` to the Kanban task id.
- Pins `HERMES_KANBAN_WORKSPACE` to the canonical workspace path
  for the task.
- Pins `HERMES_KANBAN_DB` and `HERMES_KANBAN_BOARD` so the worker
  opens the same authoritative board DB the dispatcher used.
- Removes `terminal` and `code_execution` from the worker CLI
  `--toolsets` allowlist at the resolved-toolsets level (the
  production CLI accepts a comma-separated `--toolsets` allowlist
  but does NOT accept `--disabled-toolsets`).
- Preserves the `kanban` toolset on the worker (re-appended by
  `model_tools` because `HERMES_KANBAN_TASK` is in the spawned env),
  so the worker can still self-complete via `kanban_complete`,
  `kanban_attach`, etc.

The strict gate in `tools/file_tools.py` enforces, in order:

1. `HERMES_KANBAN_STRICT_READONLY` is set (`_is_strict_readonly_active`).
2. `HERMES_KANBAN_TASK` and `HERMES_KANBAN_WORKSPACE` are present
   and non-sentinel.
3. The dispatcher-pinned workspace authenticates against the
   authoritative task row via
   `hermes_cli.kanban_db.expected_workspace_for_task(conn, task_id,
   board=board)` — a pure read-only resolver — and must canonicalise
   equal to the pinned workspace.
4. The target path resolves inside the workspace via
   `Path.resolve(strict=False).is_relative_to(workspace)`.

Identity model:

- `HERMES_KANBAN_TASK` is the Kanban task id (e.g. `t_…`).
- The `task_id` argument on file tools is the Hermes session id
  (e.g. `20260819_…`). These are **separate identities**; the gate
  never compares them.
- `HERMES_KANBAN_WORKSPACE` alone is NOT authorization; the gate
  cross-checks it against the persisted task row before allowing
  any write.

Failure modes are fail-closed: missing DB pin, missing board pin,
import failure, DB connect failure, missing task row, mismatched
workspace, symlink chain escape, `..` traversal, malformed
workspace env — all return a structured `tool_error` JSON and the
write is denied. No soft fallback, no cwd fallback, no `$HOME`
fallback.

Trusted artifact promotion: completion artifacts are promoted by
`hermes_cli/kanban_db._persist_scratch_completion_artifacts` and
`_insert_completion_attachment` (the byte-preserving path used by
`kanban_complete(artifacts=[...])`). The LLM does NOT re-emit file
contents; production code reads bytes from the existing workspace
file via `open("rb")` and writes them to the attachment directory
via `open("xb")`. This is the property that guarantees byte-exact
promotion regardless of the strict posture.

CLI surface: `hermes kanban create --strict-readonly` /
`--strict_readonly` flag at `hermes_cli/kanban.py` (`p_create`
argparse). The internal Python destination is `strict_readonly`,
which is forwarded to `hermes_cli.kanban_db.create_task`.

Autonomy propagation: `agent/autonomy/initiator.py` accepts
`objective_spec['strict_readonly']` (default `False`) and forwards
it to `kb.create_task`. The capability remains explicit per task;
no autonomous origin alone infers strict mode.

### Consequences

- Strict workers can mutate only their authorized task workspace
  through `write_file_tool` and `patch_tool`. The repository,
  profile, configuration, reports, skills, other-task workspaces,
  and external paths are not writable by the strict worker.
- Ordinary Kanban and non-Kanban behavior remains outside this
  capability. Non-strict workers see no effect.
- The dispatcher, the Kanban DB, and the trusted completion path
  remain trusted components. The file-tool gate is downstream of
  those three.
- The strict capability is opt-in per task. Operators must
  explicitly set `strict_readonly=True` (or `--strict-readonly` on
  the CLI) to activate it.
- Provider and model are NOT pinned by the capability. The
  dispatcher passes `--model`/`--provider` only when
  `task.model_override` / `task.provider_override` are explicitly
  set on the task row; the capability is independent of model
  routing.
- Strict-readonly does NOT mean zero writes. It confines the
  worker's file-tool mutation surface to the authoritative task
  workspace; the trusted completion/attachment promotion path
  remains available.

### References

- `docs/security/strict-readonly-kanban-workers.md` — full
  architectural description.
- `hermes_cli/kanban_db.py` — `class Task` (strict_readonly
  field), `create_task` (kwarg + INSERT), `_migrate_add_optional_columns`
  (column migration), `_default_spawn` (env export + toolsets
  filter), `expected_workspace_for_task` and
  `KanbanWorkspaceLookupError` (S14 binding resolver).
- `tools/file_tools.py` — `_is_strict_readonly_active`,
  `_verify_strict_readonly_task_workspace_binding`,
  `_resolve_strict_readonly_pinned_workspace`, `_strict_readonly_gate`.
- `hermes_cli/kanban.py` — `--strict-readonly` CLI flag
  (`p_create.add_argument`) and forward into `kb.create_task`.
- `agent/autonomy/initiator.py` — `objective_spec['strict_readonly']`
  propagation into `kb.create_task`.