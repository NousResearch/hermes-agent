# Per-Workspace Venv Isolation for Kanban Worktrees

Status: design, not yet implemented
Author: braintrusteng (kanban worker, task t_0d621c12)
Audience: Sahil
Scope: hermes-agent kanban dispatcher + worker

## Problem

Worktrees give kanban workers git isolation. They do **not** give Python
isolation. Every worker — and the orchestrator and the live `hermes` CLI —
imports `hermes_agent` from the same site-packages under
`~/.hermes/hermes-agent/venv/`. The package is installed editable, so the
import path is pinned by a `.pth` file plus an `__editable___…_finder.py`
that holds the repo path as a literal string.

Today's incident: a worker ran `pip install -e .` from inside its worktree.
pip rewrote `__editable___hermes_agent_0_14_0_finder.py` to point at the
worktree path. When the worktree was later garbage-collected, every other
`hermes` subprocess on the box crashed with `ModuleNotFoundError:
hermes_cli`. The dispatcher cheerfully respawned workers into the broken
venv. Sahil's recovery was a manual `sed` on the finder file.

The shared-venv blast radius is everything mutating site-packages — not
just editable installs:

- `pip install <pkg>` upgrades / downgrades a transitive of hermes-agent.
- `pip uninstall <pkg>` removes a package another worker is mid-import on.
- A worker pinning a different numpy / pydantic / boto3 version.
- A worker installing a hive's deps to smoke-test a hive change.

All of these are reasonable things a worker should be able to do, and all
of them currently corrupt shared state.

## Goal

Each kanban task workspace gets its own Python venv. A worker can
`pip install` freely without affecting other workers, the orchestrator, or
the live CLI. Worktrees stay — they're the right git primitive. Venv
isolation is added on top.

## Recommendation (one approach)

**Use `uv venv` from the canonical interpreter, plus `uv pip install -e
<canonical_hermes_agent_path>` to wire the worker's venv to the canonical
hermes-agent source.** Build the venv inside the workspace directory at
`<workspace>/.venv/`. Prepend `<workspace>/.venv/bin` to `PATH` in the
spawn env so `python`, `pip`, and `hermes` already resolve to the
workspace venv without an activation step.

Build it inline at workspace creation, in `resolve_workspace()` (or a
helper called immediately after it), behind a config flag.

### Why this approach

Three options were considered. Numbers below are from a one-shot bench on
the host that owns this task (M-series Mac, 363 MB canonical venv, warm
uv cache):

| Approach                                    | Create empty venv | Install -e hermes-agent (warm cache) | Disk per task | Isolation | Failure modes                                     |
|---------------------------------------------|------------------:|--------------------------------------:|--------------:|-----------|----------------------------------------------------|
| `uv venv` + `uv pip install -e .`           |          ~0.26 s |                                ~2.1 s |         ~60 MB | Full      | Cold uv cache adds ~5-10 s on first task only       |
| `python -m venv --copies` + `pip install -e .`|         ~3-5 s |                              ~25-40 s |        ~150 MB | Full      | Slow; doubles disk because pip vendors more         |
| Symlink-clone a template venv               |         ~0.05 s |                                  n/a |    ~1-2 MB (links) | Partial — site-packages files shared via symlink | Any worker's `pip install` mutates the template; template Python version drift; restoring after corruption needs a full rebuild |

`uv venv` wins on every axis except cold-cache create. The cold case only
hits once per host: every subsequent task reuses the on-disk uv cache
(`~/.cache/uv`, ~hundreds of MB, already populated by the canonical
install). Steady-state cost is ~2.5 s and ~60 MB per task. That's
acceptable for cards that run minutes-to-hours.

Template-clone-by-symlink was rejected because the isolation is fake —
any worker writing into the template (which is what `pip install` does by
default) corrupts every other worker that cloned from it. We just lived
this exact failure mode with the shared canonical venv; doing it again
with extra steps is not progress.

`python -m venv --copies` was rejected because the cost is real (5+
seconds is dispatcher-visible latency, 150 MB per task is dispatcher-visible
disk pressure) and it buys nothing `uv venv` doesn't already give us.

## How package installation works under this design

The hermes-agent code is editable-installed into the **task venv**, not
the canonical one. From inside the workspace:

```
uv venv .venv --python <canonical_python>
.venv/bin/python -m pip install --quiet -e /Users/.../hermes-agent
```

(or equivalently `VIRTUAL_ENV=.venv uv pip install -e
/Users/.../hermes-agent` — uv is faster and what the bench used.)

The editable install writes the worker's own
`__editable___hermes_agent_0_14_0_finder.py` into the task venv. That
finder points at the **canonical** hermes-agent checkout path (or at the
worktree path, see Open question 1 below). Worker `pip install`s never
touch the canonical venv. Today's incident becomes impossible by
construction.

Hermes-agent transitive deps (boto3, pydantic, click, …) come from the
warm uv cache as hardlinks where the FS allows it, copies otherwise. On
APFS / ext4 these are hardlinks → ~60 MB on disk per task is a worst-case
upper bound; in practice it's lower.

The worker can now do:

- `pip install -e <hive-repo>` to dev a hive without breaking anybody.
- `pip install <pkg>==<version>` to test a dep bump.
- `pip uninstall <pkg>` to verify graceful-degradation.

None of these reach outside `<workspace>/.venv/`.

## Dispatcher integration

Single point of change: `resolve_workspace()` in
`hermes_cli/kanban_db.py`. After the workspace directory exists, before
returning the path, the dispatcher:

1. If `kanban.per_workspace_venv` is false, return — unchanged behavior.
2. If `<workspace>/.venv/pyvenv.cfg` exists, return — venv is already
   built (worker re-spawns shouldn't pay the cost twice).
3. Run `uv venv <workspace>/.venv --python <canonical-python>`.
4. Run `uv pip install -e <canonical-hermes-agent-path>` against the new
   venv.
5. Return the workspace path.

The spawn function (currently around line 5700 of `kanban_db.py`) gets
two additions to `env`:

```python
env["PATH"] = f"{workspace}/.venv/bin:{env['PATH']}"
env["VIRTUAL_ENV"] = f"{workspace}/.venv"
```

That's it. `which python`, `which pip`, `which hermes` inside the worker
all resolve to the task venv. No activation step.

`uv` itself must be on the dispatcher's PATH (it already is — this
machine runs `uv venv` for canonical installs). If `uv` is not available
the dispatcher refuses to build the venv and the task blocks with a
specific error (see Failure modes below).

## Cleanup

Confirmed: the venv lives at `<workspace>/.venv/`. `gc_scratch_workspaces`
(`hermes_cli/kanban.py` ~line 2670) shells out to `Path.rmdir` /
`shutil.rmtree` on the workspace directory; `.venv` is inside that
directory; existing GC removes it as part of normal workspace cleanup.
No new GC path needed.

Worktree workspaces are not removed by the kanban scratch-workspace GC —
they're removed when the human or another card runs `git worktree
remove`. That path also nukes the directory tree, so `.venv` goes with
it. No change.

## Migration plan

1. Land the implementation behind a config flag `kanban.per_workspace_venv:
   bool` defaulting to `false`. New tasks created while the flag is off
   behave exactly as today.
2. Flip the flag to `true` on the dispatcher host for one week and watch
   for venv-creation failures, disk pressure, and worker startup latency.
   In-flight tasks created under flag-off are unaffected (their
   `workspace_path` row was set before the venv-building branch existed,
   and the no-op branch in step 2 of "Dispatcher integration" catches
   re-spawns).
3. Flip default to `true` in `hermes-agent`. Update kanban-worker and
   kanban-orchestrator skills to document that workers can now `pip
   install` freely.
4. Optional follow-up: drop the flag entirely once the design has baked
   for a release cycle.

In-flight task safety: the dispatcher only calls the venv-building branch
on `resolve_workspace()`, which is called per task. If a worker spawned
under flag-off is still running when the flag flips, its workspace has no
`.venv/`, its `PATH` was set without the prepend, and it keeps using the
canonical venv until completion. The next task on the same workspace path
(rare — only `dir:` workspaces are reused) builds the venv on first
resolve.

## Failure modes

1. **`uv venv` fails** (e.g. `uv` binary missing, canonical Python
   missing, FS full). The dispatcher blocks the task with
   `kanban_block(reason="venv-build-failed: <stderr line>")` rather than
   falling back to the shared venv. Fail-fast per the BT eng philosophy:
   silently degrading to shared-venv is exactly the failure mode this
   project exists to eliminate.

2. **`uv pip install -e <canonical>` fails** (e.g. canonical path moved,
   pyproject broken on the canonical branch). Same handling: block the
   task with a precise reason. The card stays on the board for the human
   to fix the canonical install, then unblock.

3. **Disk pressure from many concurrent task venvs.** With ~60 MB per
   task and the dispatcher capping concurrent tasks at a small number
   (single-digit, per current config), worst-case live footprint is
   single-GB. Archived-task workspaces still get GC'd by `gc_scratch_
   workspaces` on the existing schedule. If this becomes a real problem,
   the next iteration is a shared **package cache** (uv already does
   this) plus per-task **hardlinked site-packages** — but we should not
   pay that complexity until measurements demand it.

## Risks and mitigations

**Risk 1: Cold-start latency on the first task after a fresh checkout
or uv-cache wipe.** Building a venv with no warm cache and pulling
hermes-agent transitives can take 30-60 s. This is a dispatcher-visible
delay (the task sits in `claimed` while `resolve_workspace` runs).

Mitigation: warm the uv cache as part of `hermes setup`, and surface a
log line ("building task venv…") so the human watching the board knows
what's happening rather than assuming the dispatcher is wedged. Add a
dispatcher-side metric or log for venv-build duration so we notice
regressions.

**Risk 2: Editable-install finder still points at a single canonical
hermes-agent path.** A task venv's
`__editable___hermes_agent_…_finder.py` is generated at venv-build time
and is a literal path. If the canonical hermes-agent checkout moves
(rename, fresh clone elsewhere) without rebuilding task venvs, in-flight
worker venvs break the same way the original incident did — just
contained to the kanban subsystem instead of the whole CLI.

Mitigation: (a) compute the canonical path once at dispatcher startup
via the same logic the CLI uses (`get_hermes_home() / "hermes-agent"`),
and refuse to start if it doesn't resolve to an existing repo; (b) treat
moving the canonical hermes-agent checkout as a "rebuild all task venvs"
operation, documented in the runbook; (c) when the GC runs, also evict
task venvs older than N days so stale finders don't accumulate.

**Risk 3 (lower-stakes, worth naming): the worker subprocess inherits
`PATH` from the dispatcher but profile-aware tools elsewhere read
`sys.executable` directly.** If anything in the worker codebase resolves
its Python via `sys.executable` (it does, in several places), it'll
correctly pick up the task venv's interpreter once we set `PATH` +
`VIRTUAL_ENV`. But anything that hardcodes the canonical path (e.g.
`run_tests.sh` probing `$HOME/.hermes/hermes-agent/venv`) keeps using
the canonical venv. That's fine for now — those paths run tests, not
`pip install` — but worth noting so we don't get surprised when a worker
test accidentally validates against the canonical site-packages.

## Open questions for Sahil

1. **Editable install: canonical path or worktree path?** For tasks whose
   workspace is a git worktree, we have two options for `pip install -e
   <X>`: install from the canonical hermes-agent checkout (workers see
   the same code regardless of worktree state) or install from the
   worktree itself (workers test against their own branch's code). The
   second is more correct for tasks that modify hermes-agent itself; the
   first is more predictable for everything else. Recommendation:
   install from the **worktree** when `workspace_kind == "worktree"` and
   the worktree contains a `pyproject.toml`, otherwise canonical. Want
   sign-off before locking this in.

2. **`pyproject.toml` extras.** `uv pip install -e .` installs the
   `[project]` deps only. Many hermes-agent extras (model providers,
   gateway platforms) live behind `[project.optional-dependencies]`. The
   canonical install on this box installs `.[full]` or similar. Confirm
   which extras the task venv needs — defaulting to bare `[project]` will
   silently break workers that depend on, say, `boto3` from the
   `[aws]` extra. Recommend installing the same extras set the canonical
   venv uses, read from a kanban config key.

## Acceptance criteria (per task body)

- [x] Design doc landed in repo, ~1-2 pages — this file
- [x] Recommends ONE approach with rationale — `uv venv` + editable
      reinstall, justified in the trade-off table
- [x] Addresses cold-start time per task — Risk 1, ~2.5 s warm / 30-60 s
      cold, mitigated by warm-cache + log line
- [x] Addresses disk cost — ~60 MB per task, Failure mode 3 covers the
      worst case
- [x] Addresses how `pip install -e .` becomes safe — "How package
      installation works under this design" section
- [x] Lists 2+ specific risks and mitigations — Risks 1, 2, and 3

Implementation lives in a separate card. This is design-only.
