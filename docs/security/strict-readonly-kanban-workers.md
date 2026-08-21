# Strict-Readonly Kanban Workers

**Status:** Accepted
**Audience:** Hermes operators, contributors, and security reviewers
**Source files:** `hermes_cli/kanban_db.py`, `tools/file_tools.py`,
`hermes_cli/kanban.py`, `agent/autonomy/initiator.py`

## Overview

Hermes can dispatch autonomous Kanban workers via
`agent/autonomy/initiator.py`. Some workflows need an autonomous worker
that is **explicitly confined to its own task workspace** — a worker
that can write files but cannot escape that workspace, modify the
repository, reach the user's profile, or interact with arbitrary
external systems.

**Strict-readonly** is the narrow capability that satisfies that
requirement. It is an explicit, persisted task property
(`strict_readonly=True`), opt-in per task, and never inferred from
provenance. A normal autonomous task does NOT become strict merely
because it was autonomously created:

```
AUTONOMOUS ORIGIN  ≠  STRICT-READONLY CAPABILITY
```

Provenance (who created the task) is not capability (what the worker
may do). The strict mode is a separate, explicit field on the Kanban
task row.

**Strict-readonly does NOT mean zero writes.** It means:

- the worker has no `terminal` and no `code_execution` toolset;
- file mutation is confined to the authoritative task workspace;
- the trusted artifact completion / promotion path remains available;
- ordinary (non-strict) workers and non-Kanban sessions are unchanged.

## Threat Model

Without a strict capability, an autonomous worker that runs in a
workspace shell could:

- `write_file` into the Hermes repository tree (`run_agent.py`,
  `tools/file_tools.py`, etc.) and replace the gate under itself.
- `write_file` into the user's profile (`~/.hermes/profiles/...`),
  skills, `.usage.json`, or `~/.hermes/.env` and escalate privilege.
- `write_file` into another task's workspace and corrupt cross-task
  data.
- Traverse symlinks to escape the workspace.
- Open a shell (`terminal`) or run Python (`code_execution`) and
  bypass the file-tool gate entirely.

The strict capability collapses the attack surface to: the worker can
write only inside its own canonical task workspace, and only through
the file tools, only when the dispatcher-pinned workspace agrees with
the persisted task's canonical workspace.

## Security Invariants

The strict capability enforces, in order:

1. **Hermes session_id is NOT the Kanban task_id.** The value passed
   as the `task_id` argument to `write_file_tool` / `patch_tool` is
   the Hermes session ID (the format `YYYYMMDD_HHMMSS_<short>`). The
   Kanban task identity is `HERMES_KANBAN_TASK` (the format
   `t_<short>`). These are distinct identifiers and the gate never
   compares them.

2. **`HERMES_KANBAN_WORKSPACE` alone is NOT authorization.** The env
   variable is set by the dispatcher; an attacker (or a misconfigured
   dispatcher) could lie about it. The gate authenticates the workspace
   against the persisted task.

3. **Authoritative boundary is derived from persisted Kanban state.**
   `hermes_cli.kanban_db.expected_workspace_for_task(conn, task_id,
   board)` — a pure read-only resolver — returns the canonical
   workspace recorded for the task. The gate requires this to
   canonicalise equal to the pinned workspace.

4. **Pinned workspace must equal authoritative expected workspace.**
   `Path.resolve(strict=False)` of both is compared. Any mismatch —
   wrong directory, divergent symlink resolution, missing task row —
   is denied.

5. **Targets must remain canonically contained within the workspace.**
   `Path(target).resolve(strict=False).is_relative_to(workspace)` is
   the final containment check. The workspace root itself is not a
   writable target.

6. **Missing, malformed, or mismatched authority fails closed.** No
   fallback to the process cwd, no warning-and-continue, no
   degraded-but-permissive mode.

7. **Symlink and traversal escape fails closed.** `Path.resolve`
   walks the symlink chain; `..` traversal is caught by
   `is_relative_to`.

8. **Repository, profile, reports, skills, configuration, other-task
   workspaces are not writable.** They are all outside the canonical
   workspace and are denied by the containment check.

9. **`terminal` and `code_execution` are absent from the strict worker
   toolset.** The dispatcher subtracts these two from the worker's
   resolved `--toolsets` allowlist before emitting the CLI argv. The
   production CLI accepts a comma-separated `--toolsets` allowlist but
   does NOT accept `--disabled-toolsets`; emitting an unsupported
   flag is a hard argparse failure, so the strict filter operates at
   the resolved-toolsets level.

10. **Kanban lifecycle capability remains available.** `model_tools`
    re-appends the `kanban` toolset because `HERMES_KANBAN_TASK` is
    set in the spawned env, so the worker can self-complete via
    `kanban_complete`, attach via `kanban_attach`, comment via
    `kanban_comment`, etc.

11. **Completion artifacts are promoted by the trusted completion
    path.** `kanban_complete(artifacts=[...])` validates that
    `artifacts` is a top-level array of paths, then
    `hermes_cli/kanban_db._persist_scratch_completion_artifacts`
    reads each source via `open("rb")` and writes to the attachment
    directory via `open("xb")`. The LLM does not re-emit file
    contents — bytes are read by production code.

12. **Automatic background review is isolated.** The worker CLI's
    `_effective_skip_background_review` mixin detects the
    `HERMES_KANBAN_TASK` / `HERMES_SESSION_SOURCE=="kanban"` worker
    markers and propagates `skip_background_review=True` onto the
    AIAgent constructor, so no post-turn review fork fires after the
    worker's own turns. This is the current upstream behavior used
    for all Kanban workers; the strict capability does not introduce
    a separate review-suppression patch.

## Architecture

```
objective / task
        │
        ▼
strict_readonly  (persisted task capability, opt-in)
        │
        ▼
Kanban persistence (kanban.db: tasks.strict_readonly column)
        │
        ▼
dispatcher (hermes_cli/kanban_db._default_spawn)
        ├── strict env: HERMES_KANBAN_STRICT_READONLY=1
        ├── Kanban task identity: HERMES_KANBAN_TASK=t_<id>
        ├── authoritative workspace: HERMES_KANBAN_WORKSPACE=…/workspaces/t_<id>
        ├── DB pin: HERMES_KANBAN_DB=<board root>/kanban.db
        ├── board slug: HERMES_KANBAN_BOARD=<slug>
        ├── session source tag: HERMES_SESSION_SOURCE=kanban
        ├── toolsets allowlist = resolved - {terminal, code_execution}
        └── Kanban toolset still attached by model_tools
        │
        ▼
worker subprocess
  (provider / model resolved by existing task.profile / override
   machinery — strict_readonly adds confinement capability only;
   no provider / model is required by the capability)
        │
        ▼
write_file_tool / patch_tool → strict gate (tools/file_tools)
        │
        ▼
HERMES_KANBAN_TASK + HERMES_KANBAN_WORKSPACE + HERMES_KANBAN_DB + HERMES_KANBAN_BOARD
        │
        ▼
kanban_db.expected_workspace_for_task(task_id, board)
        │
        ▼
expected_workspace == pinned_workspace   (canonicalised equality)
        │
        ▼
target.resolve(strict=False).is_relative_to(workspace)
        │
        ├── ALLOW
        ▼
kanban_complete(artifacts=[...])
        │
        ▼
trusted byte-preserving artifact promotion
  (open("rb") → open("xb") → uploaded_by=kanban_complete)
        │
        ▼
task done
```

## Capability Propagation

The strict capability travels via two channels:

1. **Database column.** `tasks.strict_readonly` (`INTEGER NOT NULL
   DEFAULT 0`). The dispatcher reads this column when constructing the
   worker environment.

2. **Environment variable.** `HERMES_KANBAN_STRICT_READONLY=1`. The
   `_is_strict_readonly_active(task_id)` check in `tools/file_tools`
   reads this env. The variable is set ONLY by the dispatcher path
   that originated from a task with `strict_readonly=True`.

The CLI surface (`hermes kanban create --strict-readonly` /
`--strict_readonly` flag at `hermes_cli/kanban.py`) sets the database
column; the dispatcher path reads the column and exports the env.

Autonomy propagation: `agent/autonomy/initiator.py` accepts
`objective_spec['strict_readonly']` and forwards it to
`hermes_cli.kanban_db.create_task`. The capability is opt-in per
objective; absence / falsy values leave the task writable.

## Dispatcher Boundary

The dispatcher (`hermes_cli/kanban_db._default_spawn`) is the TCB for
strict capability. It is responsible for:

- Pinning `HERMES_KANBAN_STRICT_READONLY=1` only when
  `task.strict_readonly` is true.
- Pinning `HERMES_KANBAN_TASK` to the task id.
- Pinning `HERMES_KANBAN_WORKSPACE` to the canonical workspace path
  for the task.
- Pinning `HERMES_KANBAN_DB` and `HERMES_KANBAN_BOARD` so the worker
  opens the same board the dispatcher used.
- Removing `terminal` and `code_execution` from the `--toolsets`
  allowlist on the worker CLI argv.
- Tagging `HERMES_SESSION_SOURCE=kanban` so worker-side mechanisms
  (CLI mixin, session DB, sidebar filters) recognise the worker.
- Pinning `TERMINAL_CWD` to the workspace so worker file tools and
  context-file loading anchor on the task workspace rather than the
  dispatching gateway's cwd.

Provider / model are NOT pinned by the capability. `--model` /
`--provider` are emitted only when `task.model_override` /
`task.provider_override` are explicitly set on the task row; the
capability is independent of model routing.

The dispatcher is also responsible for the byte-preserving artifact
promotion path: `kanban_complete` in the worker triggers
`_persist_scratch_completion_artifacts` and `_insert_completion_attachment`
in the dispatcher's own kanban database connection, with
`uploaded_by='kanban_complete'`.

## Authoritative Task-to-Workspace Binding

The task-to-workspace binding check is the security-critical step:

```
os.environ["HERMES_KANBAN_TASK"]            → env_task
os.environ["HERMES_KANBAN_WORKSPACE"]       → env_workspace
os.environ["HERMES_KANBAN_DB"]              → db_pin
os.environ["HERMES_KANBAN_BOARD"]           → board_pin

conn = kanban_db.connect()
expected = kanban_db.expected_workspace_for_task(conn, env_task, board=board_pin)

Path(env_workspace).resolve(strict=False) == expected.resolve(strict=False)
```

`expected_workspace_for_task` is a pure read-only resolver. It does
not create directories, does not branch workspaces, does not write to
the DB. It only reads the canonical workspace recorded for the task.

If `HERMES_KANBAN_DB` or `HERMES_KANBAN_BOARD` are unset, the gate
denies — strict mode without dispatcher authority is refused up-front.

If `kanban_db` cannot be imported, the DB cannot be opened, the task
row is missing, or `expected_workspace_for_task` raises
`KanbanWorkspaceLookupError`, the gate denies.

## Workspace Enforcement

Containment is the last step:

```
target = Path(resolved_path).resolve(strict=False)
workspace = Path(env_workspace).resolve(strict=False)

target != workspace                                   (root not writable)
target.is_relative_to(workspace)                      (containment)
```

`Path.resolve(strict=False)` walks the symlink chain. Symlinks whose
chain leaves the workspace are caught by the containment check. `..`
traversal is caught by `is_relative_to`. Both failures return a
structured `tool_error` JSON; the worker sees the denial as a normal
tool result, not a hard agent crash.

## Toolset Confinement

The strict worker receives a `--toolsets` allowlist that contains:

- All toolsets the assignee profile would normally receive,
- **minus** `terminal`,
- **minus** `code_execution`,
- **plus** `kanban` (re-appended by `model_tools` because
  `HERMES_KANBAN_TASK` is in the env).

The dispatcher applies the filter at the resolved-toolsets level
because the Hermes CLI accepts a comma-separated `--toolsets` allowlist
but does not accept `--disabled-toolsets`. Filtering at the
resolved-toolsets level stays inside the existing CLI transport
contract.

The `terminal` and `code_execution` exclusion is the single most
important confinement: without it, the worker could shell out and
bypass the file-tool gate entirely.

## Kanban Lifecycle

The `kanban` toolset is preserved so the worker can self-complete:

- `kanban_complete(task_id, summary, result, artifacts=[...])` —
  closes the task and, when `artifacts` is supplied, persists the
  declared workspace files as task attachments via the byte-preserving
  path.
- `kanban_attach(task_id, filename, data_b64)` / `kanban_attach_url`
  / `kanban_attachments` — attach a workspace file as a task
  attachment, attach by URL, or list current attachments.
- `kanban_comment`, `kanban_heartbeat`, `kanban_show`,
  `kanban_create`, `kanban_link`, `kanban_block`,
  `kanban_request_changes`, `kanban_request_review`,
  `kanban_unblock` — observability and progress surfaces.

The `artifacts` argument to `kanban_complete` MUST be a list of
absolute paths. The dispatcher's own `kanban_db.complete_task`
validates the shape.

## Trusted Artifact Promotion

When the worker calls `kanban_complete(artifacts=[...])`:

1. `tools/kanban_tools.py::_handle_kanban_complete` validates the
   `artifacts` argument is a top-level list and copies the values
   into `metadata["artifacts"]`.
2. `hermes_cli/kanban_db.complete_task(...)` invokes
   `_persist_scratch_completion_artifacts(conn, task_id, metadata)`.
3. The promotion path validates each declared path is inside the
   canonical workspace, opens each source via `open("rb")`, and writes
   to the attachment directory via `open("xb")` (chunk-by-chunk).
4. `_insert_completion_attachment(...)` records each artifact as a
   `task_attachments` row with `uploaded_by='kanban_complete'`.

The LLM does not re-emit file contents in this path. Production code
reads bytes from the existing workspace file and writes them to the
attachment store. This is the property that guarantees byte-exact
promotion.

When the worker uses `kanban_attach` directly, the underlying write
goes through `hermes_cli.kanban_db.store_attachment_bytes`, which is
the trusted in-DB attachment write path used by the dashboard, the
CLI, and the kanban toolset. None of these trusted paths are
intercepted by the strict file-tool gate because they are not
model-tool mutations of arbitrary file paths — they are explicit
promotion calls into the kanban subsystem.

## Background-Review Isolation

A normal worker session can fire `_spawn_background_review` after a
turn, which can invoke `skill_view` / `skill_manage` and trigger
`curator` / `auxiliary_client` events. For a strict worker this would
be a silent second-tick attack surface.

The current upstream behavior uses the worker CLI's
`_effective_skip_background_review()` mixin
(`hermes_cli/cli_agent_setup_mixin.py`), which detects
`HERMES_KANBAN_TASK` or `HERMES_SESSION_SOURCE=="kanban"` in the
worker subprocess env and propagates `skip_background_review=True`
onto the AIAgent constructor. The dispatcher sets
`HERMES_SESSION_SOURCE=kanban` for every Kanban worker, so this
mechanism applies to both strict and non-strict workers; the strict
capability does not introduce a separate review-suppression patch.

The worker session log shows exactly the API calls for the worker's
own turns and no skill-management events after the worker exits.

## Failure and Fail-Closed Semantics

Every failure mode returns a structured `tool_error` JSON and the
write is denied. There is no soft fallback, no return
`None`-indicating-allowed-with-warning, no falling back to the
process cwd or `$HOME`.

Failure modes (non-exhaustive):

- `HERMES_KANBAN_STRICT_READONLY` not set → gate is a no-op (this is
  the non-strict path; gate does not fire).
- `HERMES_KANBAN_TASK` empty → deny.
- `HERMES_KANBAN_WORKSPACE` empty, sentinel, non-absolute, or not an
  existing directory → deny.
- `HERMES_KANBAN_DB` or `HERMES_KANBAN_BOARD` empty → deny.
- `kanban_db` import or DB connect failure → deny.
- `expected_workspace_for_task` raises `KanbanWorkspaceLookupError`
  → deny.
- Pinned workspace ≠ authoritative expected workspace → deny.
- Target equals workspace root → deny.
- Target not `is_relative_to(workspace)` → deny.
- Path traversal check raises `OSError` → deny.
- Path canonicalisation raises → deny.

## Validation

The contract is exercised by reproducible repository tests:

- `tests/hermes_cli/test_kanban_strict_readonly_field.py` —
  schema, migration, round-trip persistence.
- `tests/hermes_cli/test_kanban_worker_strict_dispatch.py` —
  dispatcher env export, toolsets filter, lifecycle preservation,
  model / provider / reasoning propagation, trusted artifact
  promotion.
- `tests/tools/test_file_tools_strict_workspace_gate.py` —
  containment, traversal, symlink escape, missing / malformed /
  mismatched / unknown-task denial, S14 binding matrix.
- `tests/hermes_cli/test_kanban_initiator_strict_propagation.py` —
  autonomy objective → task row propagation, no origin-based
  inference.
- `tests/cron/test_cron_kanban_env_isolation.py` —
  `HERMES_KANBAN_STRICT_READONLY` is in the identity-gated set.

Background-review isolation is exercised by
`tests/agent/test_skip_background_review.py` and
`tests/agent/test_skip_background_review_cli_propagation.py`.

Provider / model preservation is exercised by
`tests/hermes_cli/test_kanban_worker_strict_dispatch.py`
(`test_strict_worker_propagates_model_provider_reasoning`).

## Limitations

- Strict-readonly applies to the explicit Kanban strict capability. It
  is not a global process sandbox, not a kernel-level isolation, and
  not a substitute for an OS-level sandbox.
- It does not make every Hermes session read-only. Ordinary CLI,
  non-strict Kanban, and non-Kanban sessions are unaffected.
- It does not make autonomous origin itself a security authority.
  Provenance is not capability.
- It does not eliminate the trusted dispatcher, the trusted Kanban DB,
  or the trusted completion path from the TCB. All three are
  upstream of the file-tool gate and remain attacker-relevant if
  compromised.
- It does not authorize arbitrary external filesystem writes. It
  confines the worker's file-tool mutation surface, nothing more.
- This document describes the accepted current implementation. It is
  not a future-design document; proposed-but-not-accepted changes are
  not documented here.

## Related

- `docs/ADR.md` — `## 2026-08-20: Strict-readonly Kanban worker
  capability (workspace confinement)` decision entry.
- `hermes_cli/kanban_db.py` — `expected_workspace_for_task`,
  `KanbanWorkspaceLookupError`, `_default_spawn`,
  `_persist_scratch_completion_artifacts`, `_insert_completion_attachment`,
  `store_attachment_bytes`.
- `tools/file_tools.py` — `_is_strict_readonly_active`,
  `_verify_strict_readonly_task_workspace_binding`,
  `_resolve_strict_readonly_pinned_workspace`, `_strict_readonly_gate`.
- `hermes_cli/kanban.py` — `--strict-readonly` CLI flag.
- `agent/autonomy/initiator.py` — `strict_readonly` objective field.
- `hermes_cli/cli_agent_setup_mixin.py` — `_effective_skip_background_review`
  worker marker detection.
- `docs/security/network-egress-isolation.md` — adjacent isolation
  pattern (network egress, complementary concept).
- `SECURITY.md` — Hermes trust model and vulnerability reporting.