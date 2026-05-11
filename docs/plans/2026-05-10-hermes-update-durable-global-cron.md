---
title: Hermes Update-Durable Global Cron and Local Patch Recovery Implementation Plan
date: 2026-05-10
type: implementation-plan
status: audited-revised-ready-for-implementation
tags:
  - hermes-agent
  - global-cron
  - update-durability
  - cron
  - stash-recovery
  - tdd
aliases:
  - update-durable global cron plan
  - permanent Hermes update conflict fix
  - global cron board implementation plan
canonical_path: /Users/hache/.hermes/hermes-agent/docs/plans/2026-05-10-hermes-update-durable-global-cron.md
---

# Hermes Update-Durable Global Cron and Local Patch Recovery Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Status:** Audited and revised planning artifact. Ready for implementation handoff, but code has not been executed. Do not treat any code as already changed.

**Goal:** Convert the current update-conflict/stash situation into a durable, tested workflow and implement global cron as an upstreamable feature instead of a fragile local stash.

**Architecture:** Keep upstream `main` clean and treat local work as explicit branches. Implement global cron as an additive shared cron store rooted at `get_default_hermes_root()`, while preserving profile-scoped cron semantics by default. Execute global jobs in a subprocess with `HERMES_HOME` set to the selected `run_as_profile` before Hermes imports, and keep global job metadata/output in the global store.

**Tech Stack:** Python 3.11, pytest, Hermes CLI/gateway/cron modules, JSON file stores, subprocess execution, React dashboard API/types if the web UI is included, git branches, strict TDD.

---

## Why this plan exists

`hermes update` completed and the CLI/gateway are healthy, but local work was autostashed because upstream changed overlapping files. Reapplying that stash directly to `main` would be unsafe because:

1. **Upstream already moved.** The update pulled 107 commits and reset local `main` to match `origin/main`.
2. **The stash conflicts in core files.** The update reported conflicts in `agent/account_usage.py` and `tui_gateway/server.py`.
3. **There is separate untracked global-cron work.** `tests/cron/test_global_cron.py` exists untracked and describes an intended global cron board API not present in current `cron/jobs.py`.
4. **Cron/security code is high-risk.** Current `tools/cronjob_tools.py` and `cron/scheduler.py` contain prompt-injection, script-path, delivery-target, and assembled-prompt security hardening. Any permanent fix must preserve these changes.
5. **A stash is not durable.** A stash is invisible to normal update/CI workflows, easy to forget, and likely to conflict again after the next update.

The permanent fix is therefore not “apply the stash.” The permanent fix is:

- turn local changes into named branches;
- implement/repair global cron against current upstream with tests first;
- add an update-repair workflow that refuses unsafe installs;
- upstream the feature or maintain it as an explicit local branch until upstreamed.

---

## Current observed state

- Hermes project: `/Users/hache/.hermes/hermes-agent`
- Current version after update: `Hermes Agent v0.13.0 (2026.5.7)`
- Gateway status after update: running for profile `chat_d3c3n`
- Clean branch status: `main...origin/main`, with untracked files only
- Preserved autostash: `a579938a538fcdde9ec7088ea146b4446c3bc7a5`
- Stash contents from `git show --stat`:
  - `agent/account_usage.py`
  - `tests/test_account_usage.py`
  - `tests/tools/test_session_search.py`
  - `tools/session_search_tool.py`
  - `tui_gateway/server.py`
- Untracked files relevant to local work:
  - `docs/plans/2026-05-01-usage-left-bar.md`
  - `tests/cron/test_global_cron.py`
  - `tests/tools/test_session_search_fix_tests.py`
  - `ui-tui/src/__tests__/appChrome.test.ts`
  - `web/.npmrc`

---

## Local execution prerequisites

All Python commands in this plan must run inside the repository virtualenv on this macOS checkout. Plain `python` is not available in the current tool shell, and system `python3` does not have pytest installed.

Before running tests or installs:

```bash
cd /Users/hache/.hermes/hermes-agent
source ./venv/bin/activate
python --version        # expected Python 3.11.x
./venv/bin/python -m pytest --version
```

If the venv is not activated, use `./venv/bin/python` explicitly. This plan uses `./venv/bin/python` in commands to be unambiguous.

For web checks, `web/package.json` currently has `build` and `lint` scripts but no `test` script. Use `pnpm --dir web build` and `pnpm --dir web lint` unless a frontend test script is added first.

---

## Non-goals

1. Do not collapse profile cron files into a single global cron file.
2. Do not mutate profile globals in-process to “become” another profile.
3. Do not blindly use `git stash apply` on `main` and mark conflicts resolved with `ours`/`theirs`.
4. Do not bypass current cron prompt-injection, script path, auth-header, or assembled-prompt scanners.
5. Do not implement global cron by making `hermes cron list` show every private profile’s jobs.
6. Do not run global jobs in-process if the subprocess entrypoint fails; fail loudly and record the job error.
7. Do not claim update durability until the update-repair workflow and test gates exist.

---

## Core design decisions and rationale

### Decision 1: Keep profile cron default behavior unchanged

**Requirement:** Existing `hermes cron list`, `cronjob(action='list')`, scheduler behavior, and dashboard behavior must continue to work for profile-scoped jobs.

**Rationale:** Cron jobs are currently profile-scoped. Users may rely on isolation between `chat_d3c3n`, `deepseek`, `qwen27b`, etc. Changing default list/mutation behavior risks leaking jobs across profiles or accidentally mutating a private job from another profile.

### Decision 2: Add a separate global store

**Requirement:** Add a `CronStore` abstraction so code can operate against either:

- profile store: `$HERMES_HOME/cron/jobs.json`
- global store: `get_default_hermes_root()/cron/jobs.json` or a similarly explicit global root under the default Hermes root

**Rationale:** This is additive, migration-free for existing profile jobs, and update-friendly. It avoids relocating or rewriting existing job files.

### Decision 3: Use explicit scoped refs

**Requirement:** Use refs such as:

- naked `abc123`: current profile only
- `global:abc123`: global store
- optionally `profile:<profile_name>:abc123`: explicit profile ref for internal/admin tooling, not broad default UX

**Rationale:** Naked IDs mutating global jobs would be surprising and dangerous. Scoped refs make the operator’s intent explicit and solve ID collision ambiguity.

### Decision 4: Require `run_as_profile` for global jobs

**Requirement:** Every global job must include a valid `run_as_profile`.

**Rationale:** A global job is shared metadata, but it still needs a runtime identity: config, auth, scripts directory, messaging channels, skills, and memory behavior are profile-scoped. If `run_as_profile` is missing or invalid, record a job error and do not run.

### Decision 5: Execute global jobs through a subprocess only

**Requirement:** Global scheduler must launch an internal Hermes cron runner subprocess with:

```bash
HERMES_HOME=<selected profile home> hermes cron run-internal --store-root <global-store-root> --job-id <job-id>
```

or equivalent Python module invocation if the installed executable is unavailable.

**Rationale:** Hermes modules often resolve `get_hermes_home()` at import time. Mutating `os.environ`, module globals, or `cwd` inside a long-lived scheduler process is error-prone. A subprocess provides a clean import boundary.

### Decision 6: The internal runner must use the explicit `--store-root`

**Requirement:** `cron run-internal` must reconstruct `CronStore` from `--store-root` and must not call `global_store()` implicitly after `HERMES_HOME` has been changed for the runtime profile.

**Rationale:** In the subprocess, `HERMES_HOME` points to the runtime profile. Calling `global_store()` there can resolve the wrong global root. The store root must be carried explicitly.

### Decision 7: Preserve all upstream cron security hardening

**Requirement:** Global cron creation/update must use the same security validation as profile jobs:

- prompt injection scan in `tools/cronjob_tools.py`
- script path restriction to `HERMES_HOME/scripts/`
- dangerous prompt/auth-header scanning
- assembled prompt scan in `cron/scheduler.py`
- delivery target validation

**Rationale:** Global jobs have broader blast radius and may run under privileged profiles. They must be at least as constrained as profile jobs.

### Decision 8: Make update durability an explicit workflow, not magic

**Requirement:** Add a branch/update repair workflow that detects conflicts and refuses unsafe installation unless tests pass.

**Rationale:** Semantic conflict resolution in cron/security files should not be fully automated. Automation should detect, branch, rebase/merge, test, and refuse unsafe installs; a repair agent or human should resolve semantic conflicts with the plan and tests in hand.

---

## Required acceptance criteria

### Existing behavior must remain true

- Legacy jobs missing `scope` load as profile jobs.
- Existing `hermes cron list` remains profile-only by default.
- Existing `cronjob(action='list')` remains profile-only by default unless a new explicit option is used.
- Existing profile scheduler continues to run profile jobs.
- Existing delivery behavior still supports `local`, `origin`, explicit platform refs, media delivery, and silent outputs.
- Existing script-path validation rejects absolute paths and `~` paths.
- Existing prompt/security scanners still block known malicious patterns.

### Global cron behavior must be true

- Global job creation requires valid `run_as_profile`.
- Missing/invalid `run_as_profile` records an error and does not fallback to in-process/profile-default execution.
- Global jobs store metadata and output in the global store.
- Global no-agent script resolves relative to the selected runtime profile’s `scripts/` directory.
- Visible/dashboard list can intentionally show current profile jobs plus global jobs, but not every other private profile’s jobs.
- Naked short IDs cannot pause/resume/update/remove/trigger global jobs.
- `global:<id>` can pause/resume/update/remove/trigger global jobs.
- Multiple profile gateways ticking at the same time execute each due global job at most once.
- One-shot grace prevents duplicate execution.
- `context_from` is store-aware or cross-store context is explicitly rejected for v1.
- `cron run-internal` works as a subprocess entrypoint and honors `--store-root`.

### Update durability must be true

- Local work is on a named branch, not only a stash.
- The update-repair script detects whether global-cron symbols/tests are missing from active code.
- The script refuses to install or restart services if merge/rebase conflicts remain.
- The script refuses to install or restart services if required tests fail.
- The script reports the exact branch, stash ref, conflicts, and test commands.

---

## Audit-integrated implementation requirements

The independent audit findings below are now part of the main implementation spec, not optional review notes. Implementers must follow them while executing Tasks 1-16.

1. **Global job transaction owner:** `cron run-internal` owns the full global job transaction: load from explicit global store, run under `run_as_profile`, save output to the global store, deliver using runtime profile config, mark success/failure in the global store, then exit. Parent scheduler stdout/stderr capture is diagnostic only.
2. **Cross-process claim semantics:** Global due-job selection must be atomic under the global store file lock. In-process `threading.Lock` is insufficient for multiple gateway processes.
3. **Profile authorization:** For v1, normal tool/API global job creation may only use `run_as_profile` equal to the current profile. Cross-profile scheduling requires an explicit admin-only CLI/config override and tests.
4. **Secure profile resolution:** Use canonical Hermes profile rules; support default profile correctly; reject traversal/control/path-separator names and symlink escapes.
5. **Runtime-profile script validation:** Global job scripts validate and execute relative to `resolve_profile_home(run_as_profile) / "scripts"`, never the creator profile.
6. **Hardened internal runner:** `run-internal` validates `--scope`, `--job-id`, and `--store-root`; clears spoofable session/delivery env; uses no shell; and prefers current Python/current install over `PATH` `hermes`.
7. **Global delivery privacy:** Global jobs default to `deliver="local"`; origin/chat metadata must not leak through dashboard/API; parent live gateway adapters are not used for cross-profile child delivery.
8. **Exhaustive mutation rules:** Naked IDs mutate only current profile jobs. Every global update/pause/resume/remove/run requires `global:<id>`. `scope` is immutable; `run_as_profile` changes require creation-level authorization.
9. **Same-store context for v1:** `context_from` is same-store only. Profile→global and global→profile references are rejected unless a later explicitly authorized design is added.
10. **Shared validation paths:** Tool, CLI, dashboard/API, update, and `run-internal` paths must preserve prompt, script, auth-header/exfil, assembled-prompt, context, and delivery validation.
11. **Test isolation:** All new cron tests must use temp profile/global roots and must not read or write the user’s real Hermes cron stores. Split tests by phase or use phase-local imports so focused RED/GREEN cycles can collect.
12. **Repair/rollback safety:** Update-repair must refuse dirty/conflicted/untracked worktrees by default and never install/restart automatically. Rollback must stop services first, back up global cron state, and avoid broad untracked deletion without `git clean -nd`.

---

## Implementation tasks

### Task 1: Create explicit recovery branches without touching `main`

**Objective:** Preserve all local work in discoverable branches before implementing anything.

**Files:** none directly; git metadata only.

**Step 1: Verify current branch and status**

Run:

```bash
cd /Users/hache/.hermes/hermes-agent
git status --short --branch
git stash list | head -20
```

Expected: branch is `main...origin/main`; autostash exists; untracked files are visible.

**Step 2: Create a stash recovery branch**

Run:

```bash
git switch -c recovery/autostash-20260510-main
```

Expected: branch created from updated `main`.

**Step 3: Apply stash on the recovery branch only**

Run:

```bash
git stash apply a579938a538fcdde9ec7088ea146b4446c3bc7a5 || true
git status --short
```

Expected: conflicts may appear only on the recovery branch. Do not resolve yet.

**Step 4: Snapshot conflict metadata**

Run:

```bash
git diff --name-only --diff-filter=U > /tmp/hermes-autostash-conflicts-20260510.txt
git diff --stat > /tmp/hermes-autostash-diffstat-20260510.txt
```

Expected: conflict list saved outside the repo.

**Step 5: Reset the inspection branch safely or keep it for repair**

If this branch is only for inspection, after saving conflict metadata run:

```bash
git reset --hard HEAD
git status --short
git switch main
git status --short --branch
```

Do **not** run broad `git clean -fd`; important plan/test files are currently untracked. Only clean specific stash-created untracked paths after confirming they are unrelated to the planning/test files.

If the stash content needs repair, leave the branch and follow a separate plan for account usage/session search. Do not mix it with global cron.

**Rationale:** The autostash contains account usage, session search, and TUI work, not the untracked global cron test. Mixing unrelated repairs increases conflict risk. `git stash apply` conflicts are not always an active merge, so `git merge --abort` is not the safe cleanup primitive here.

---

### Task 2: Create the global cron feature branch and commit only planning/test artifacts first

**Objective:** Move global cron work from untracked files into a branch with a clean history.

**Files:**

- Add: `docs/plans/2026-05-10-hermes-update-durable-global-cron.md`
- Add: `tests/cron/test_global_cron.py` after repair in later tasks
- Add or modify: `docs/plans/README.md`

**Step 1: Switch back to clean main**

Run:

```bash
cd /Users/hache/.hermes/hermes-agent
git switch main
git status --short --branch
```

Expected: no staged conflicts; only untracked local files.

**Step 2: Create feature branch**

Run:

```bash
git switch -c feature/update-durable-global-cron
```

Expected: branch created from updated `main`.

**Step 3: Add planning artifacts only**

Run:

```bash
git add docs/plans/2026-05-10-hermes-update-durable-global-cron.md docs/plans/README.md
git commit -m "docs: plan update-durable global cron"
```

Expected: plan committed separately from implementation.

**Rationale:** A planning commit gives future update/merge agents a stable spec anchor before code changes begin.

---

### Task 3: Repair and commit the global cron tests as RED tests

**Objective:** Make `tests/cron/test_global_cron.py` the source of truth for desired behavior before production code exists.

**Files:**

- Modify/Add: `tests/cron/test_global_cron.py`

**Step 1: Read the current untracked test and remove assumptions from stale implementation**

Check imports and desired APIs. The existing test references:

```python
from cron.jobs import (
    CronStore,
    current_profile_store,
    global_store,
    resolve_store,
    store_lock,
    list_visible_jobs,
    _parse_job_ref,
    _normalize_job_scope,
    _make_job_ref,
)
```

These do not exist in current upstream and should be implemented later, not mocked away.

**Audit-required test phasing:** Do not import all future cron APIs at module import time. Either split tests by phase or move imports inside the tests/classes that need them. Recommended split:

- `tests/cron/test_cron_store.py` for `CronStore`, store resolution, locks, `load_jobs`, and `save_jobs`.
- `tests/cron/test_cron_scope_refs.py` for scope normalization and scoped refs.
- `tests/cron/test_global_cron_crud.py` for global/profile CRUD and visible-list behavior.
- `tests/cron/test_global_cron_scheduler.py` for global locking, claiming, subprocess execution, and output behavior.

All tests must use an autouse temp-root fixture that monkeypatches `HERMES_HOME`, `cron.jobs.get_hermes_home`, and `cron.jobs.get_default_hermes_root` before any store function is called. No global cron test may read or write real user cron files. Avoid signature-only tests for critical behavior; assert actual create/list/mutate/run/output behavior against temp stores.

**Step 2: Add a fixture that isolates both profile and global homes**

Use temp dirs and monkeypatch root resolvers so tests never touch real `~/.hermes`:

```python
@pytest.fixture
def isolated_cron_roots(tmp_path, monkeypatch):
    profile_root = tmp_path / "profiles" / "chat_d3c3n"
    global_root = tmp_path / "root"
    profile_root.mkdir(parents=True)
    global_root.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    monkeypatch.setattr("cron.jobs.get_hermes_home", lambda: profile_root)
    monkeypatch.setattr("cron.jobs.get_default_hermes_root", lambda: global_root)
    return profile_root, global_root
```

If `get_default_hermes_root` is not importable yet, mark the test as expecting import failure until Task 4 adds it.

**Step 3: Add or keep RED tests for store isolation**

Required tests:

- `test_legacy_job_missing_scope_loads_as_profile`
- `test_profile_list_excludes_global_jobs_by_default`
- `test_visible_list_includes_profile_and_global_jobs`
- `test_global_create_requires_run_as_profile`
- `test_global_create_rejects_invalid_run_as_profile`
- `test_naked_id_remove_does_not_touch_global_job`
- `test_global_ref_remove_touches_global_job`
- `test_global_output_is_saved_under_global_output_dir`

**Step 4: Run RED tests**

Run:

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -o 'addopts=' -q
```

Expected: FAIL because `CronStore`, `global_store`, `list_visible_jobs`, and scoped CRUD do not exist yet.

**Step 5: Commit RED tests**

Run:

```bash
git add tests/cron/test_global_cron.py
git commit -m "test: specify global cron store behavior"
```

**Rationale:** This locks the contract before production code changes and prevents implementing a convenient but unsafe design.

---

### Task 4: Add `CronStore` and store-aware load/save primitives

**Objective:** Introduce the smallest storage abstraction needed for profile/global stores.

**Files:**

- Modify: `cron/jobs.py`
- Test: `tests/cron/test_global_cron.py`

**Step 1: Add imports**

In `cron/jobs.py`, change:

```python
from hermes_constants import get_hermes_home
```

to:

```python
from dataclasses import dataclass
from hermes_constants import get_hermes_home, get_default_hermes_root
```

Current upstream already provides `get_default_hermes_root()` in `hermes_constants.py`; import and use it directly. Do not invent a separate shared-root helper.

**Step 2: Add `CronStore` near configuration constants**

```python
@dataclass(frozen=True)
class CronStore:
    scope: str
    root: Path
    cron_dir: Path
    jobs_file: Path
    output_dir: Path
    lock_file: Path


def _store_for_root(scope: str, root: Path) -> CronStore:
    cron_dir = root.resolve() / "cron"
    return CronStore(
        scope=scope,
        root=root.resolve(),
        cron_dir=cron_dir,
        jobs_file=cron_dir / "jobs.json",
        output_dir=cron_dir / "output",
        lock_file=cron_dir / ".tick.lock",
    )


def current_profile_store() -> CronStore:
    return _store_for_root("profile", get_hermes_home())


def global_store() -> CronStore:
    return _store_for_root("global", get_default_hermes_root())


def store_from_root(scope: str, root: Path | str) -> CronStore:
    return _store_for_root(scope, Path(root))


def resolve_store(scope: str | None = None, *, store: CronStore | None = None) -> CronStore:
    if store is not None:
        return store
    if scope == "global":
        return global_store()
    return current_profile_store()
```

**Step 3: Replace module-global lock with store-keyed locks**

```python
_store_locks: dict[Path, threading.Lock] = {}
_store_locks_guard = threading.Lock()


def store_lock(store: CronStore | None = None) -> threading.Lock:
    resolved = resolve_store(store=store)
    key = resolved.jobs_file.resolve()
    with _store_locks_guard:
        lock = _store_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _store_locks[key] = lock
        return lock
```

Keep `_jobs_file_lock` temporarily as an alias if needed for backwards-compatible tests, but move call sites to `store_lock(store)`.

**Step 4: Make `ensure_dirs`, `load_jobs`, and `save_jobs` store-aware**

Change signatures:

```python
def ensure_dirs(store: CronStore | None = None): ...
def load_jobs(store: CronStore | None = None) -> List[Dict[str, Any]]: ...
def save_jobs(jobs: List[Dict[str, Any]], store: CronStore | None = None): ...
```

Inside each function, call `resolved = resolve_store(store=store)` and use `resolved.cron_dir`, `resolved.jobs_file`, and `resolved.output_dir`.

**Step 5: Run focused tests**

Run:

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'CronStore or load_jobs or save_jobs or store_lock' -o 'addopts=' -q
```

Expected: relevant tests pass; later scoped CRUD tests still fail.

**Step 6: Run legacy cron tests**

Run:

```bash
./venv/bin/python -m pytest tests/cron tests/tools/test_cronjob_tools.py -o 'addopts=' -q
```

Expected: no legacy regression. If `tests/tools/test_cronjob_tools.py` path differs, locate current cron tests with `./venv/bin/python -m pytest --collect-only tests -q | grep -i cron`.

**Step 7: Commit**

```bash
git add cron/jobs.py tests/cron/test_global_cron.py
git commit -m "feat: add store abstraction for cron jobs"
```

**Rationale:** A store abstraction minimizes branching logic in every CRUD function and keeps the default profile store intact.

---

### Task 5: Add scope normalization and job refs

**Objective:** Make every returned job carry a safe scope and a stable ref.

**Files:**

- Modify: `cron/jobs.py`
- Test: `tests/cron/test_global_cron.py`

**Step 1: Add helpers**

```python
def _normalize_job_scope(job: Dict[str, Any], *, default: str = "profile") -> Dict[str, Any]:
    normalized = dict(job)
    scope = str(normalized.get("scope") or default).strip().lower()
    if scope not in {"profile", "global"}:
        scope = "profile"
    normalized["scope"] = scope
    return normalized


def _parse_job_ref(job_ref: str) -> tuple[str | None, str]:
    text = str(job_ref or "").strip()
    if text.startswith("global:"):
        return "global", text.split(":", 1)[1]
    if text.startswith("profile:"):
        parts = text.split(":")
        return "profile", parts[-1] if parts else ""
    return None, text


def _make_job_ref(job: Dict[str, Any]) -> str:
    scope = str(job.get("scope") or "profile")
    return f"{scope}:{job.get('id', '')}"


def _normalize_job_for_return(job: Dict[str, Any], *, store: CronStore | None = None) -> Dict[str, Any]:
    default_scope = resolve_store(store=store).scope if store is not None else "profile"
    normalized = _normalize_job_record(_normalize_job_scope(job, default=default_scope))
    normalized["job_ref"] = _make_job_ref(normalized)
    return normalized
```

**Step 2: Apply to `load_jobs` return path**

When loading jobs from a store, normalize each job with that store’s scope as default.

**Step 3: Run tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'scope or ref' -o 'addopts=' -q
```

Expected: scope/ref tests pass.

**Step 4: Commit**

```bash
git add cron/jobs.py tests/cron/test_global_cron.py
git commit -m "feat: add scoped cron job refs"
```

**Rationale:** Scoped refs are the foundation for safe mutation behavior.

---

### Task 6: Make CRUD operations store- and scope-aware

**Objective:** Support profile/global CRUD while preserving naked-ID profile-only behavior.

**Files:**

- Modify: `cron/jobs.py`
- Test: `tests/cron/test_global_cron.py`

**Step 1: Update signatures with backward-compatible defaults**

```python
def create_job(..., scope: str = "profile", run_as_profile: Optional[str] = None, store: CronStore | None = None): ...
def get_job(job_id: str, store: CronStore | None = None) -> Optional[Dict[str, Any]]: ...
def list_jobs(include_disabled: bool = False, store: CronStore | None = None) -> List[Dict[str, Any]]: ...
def update_job(job_id: str, updates: Dict[str, Any], store: CronStore | None = None) -> Optional[Dict[str, Any]]: ...
def remove_job(job_id: str, store: CronStore | None = None) -> bool: ...
```

**Step 2: Add profile existence validation for `run_as_profile`**

Add helper:

```python
def _profile_home(profile_name: str) -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "profiles" / profile_name


def _validate_run_as_profile(profile_name: Optional[str]) -> str:
    name = str(profile_name or "").strip()
    if not name:
        raise ValueError("Global cron jobs require run_as_profile")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"Invalid run_as_profile: {name!r}")
    if not _profile_home(name).exists():
        raise ValueError(f"run_as_profile does not exist: {name}")
    return name
```

Tests may monkeypatch this helper instead of creating real profile directories.

**Audit-required replacement:** The sketch above must be implemented as a secure canonical `resolve_profile_home(profile_name: str) -> Path` using Hermes profile layout rules. It must support the default profile correctly, reject `/`, `\`, `:`, NUL/control characters, traversal, symlink escapes, and names that do not round-trip as one path component.

**Authorization boundary:** For v1, global job creation through the `cronjob` tool and dashboard/API must require `run_as_profile` to match the current profile identity. Cross-profile scheduling is rejected unless an explicit admin-only CLI/config override is implemented and tested. Updates may not change `run_as_profile` without the same authorization checks.

**Step 3: In `create_job`, select target store**

```python
scope = str(scope or "profile").lower()
if scope == "global":
    run_as_profile = _validate_run_as_profile(run_as_profile)
    target_store = resolve_store("global", store=store)
else:
    scope = "profile"
    target_store = resolve_store("profile", store=store)
```

Store `scope` and, for global jobs, `run_as_profile` in the job dict.

**Step 4: In `get_job`, parse scoped refs**

- Naked ID: search only current `store` or current profile store.
- `global:<id>`: search global store.
- `profile:<name>:<id>`: only implement if needed; otherwise parse but restrict to current profile for v1.

**Step 5: In mutation functions, require scoped ref for global**

`remove_job("abc")` must not remove global `abc`. `remove_job("global:abc")` must.

**Step 6: Add `list_visible_jobs`**

```python
def list_visible_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    jobs = list_jobs(include_disabled=include_disabled, store=current_profile_store())
    jobs.extend(list_jobs(include_disabled=include_disabled, store=global_store()))
    return jobs
```

Do not change `list_jobs()` default behavior.

**Step 7: Run tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'CRUD or visible or naked or global_create' -o 'addopts=' -q
./venv/bin/python -m pytest tests/cron tests/tools/test_cronjob_tools.py -o 'addopts=' -q
```

Expected: all scoped CRUD tests pass; legacy tests pass.

**Step 8: Commit**

```bash
git add cron/jobs.py tests/cron/test_global_cron.py
git commit -m "feat: support scoped cron CRUD"
```

**Rationale:** This is the behavioral core. It must be done without changing default profile semantics.

---

### Task 7: Make outputs and `context_from` store-aware

**Objective:** Ensure global job output is saved/read from the global output directory, not the runtime profile.

**Files:**

- Modify: `cron/jobs.py`
- Modify: `cron/scheduler.py` if output/context reads happen there
- Test: `tests/cron/test_global_cron.py`

**Step 1: Update `save_job_output`**

Change signature:

```python
def save_job_output(job_id: str, output: str, store: CronStore | None = None):
```

Use `resolve_store(store=store).output_dir`.

**Step 2: Add context lookup helper**

If `context_from` currently assumes profile output paths, add:

```python
def load_latest_job_output(job_ref: str, *, current_store: CronStore | None = None) -> Optional[str]:
    scope, job_id = _parse_job_ref(job_ref)
    if scope == "global":
        store = global_store()
    elif scope in {None, "profile"}:
        store = current_store or current_profile_store()
    else:
        return None
    # load most recent output under store.output_dir / job_id
```

For v1, `context_from` is same-store only. Profile jobs may reference current-profile outputs; global jobs may reference global outputs. Profile→global and global→profile references are rejected with a clear error unless a later explicitly authorized design is added. In `cron/scheduler.py`, remove/avoid static `OUTPUT_DIR` assumptions and route context reads through a store-aware helper.

**Step 3: Run tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'output or context' -o 'addopts=' -q
```

Expected: global output writes under global store; context behavior is explicit.

**Step 4: Commit**

```bash
git add cron/jobs.py cron/scheduler.py tests/cron/test_global_cron.py
git commit -m "feat: make cron output store-aware"
```

**Rationale:** A global job’s metadata and audit trail must remain global even when runtime config comes from a profile.

---

### Task 8: Add global scheduler due-job selection with file locking

**Objective:** Let each profile gateway tick profile jobs plus shared global due jobs without duplicate execution.

**Files:**

- Modify: `cron/jobs.py`
- Modify: `cron/scheduler.py`
- Test: new or existing `tests/cron/test_global_cron.py`

**Step 1: Make due-job functions store-aware**

Change signatures:

```python
def get_due_jobs(store: CronStore | None = None) -> List[Dict[str, Any]]: ...
def _get_due_jobs_locked(store: CronStore | None = None) -> List[Dict[str, Any]]: ...
def mark_job_run(job_id: str, ..., store: CronStore | None = None): ...
def advance_next_run(job_id: str, store: CronStore | None = None): ...
```

**Step 2: Add a global lock path**

Use `global_store().lock_file`, not the profile lock, for global due-job selection.

**Step 3: Update `tick()`**

Pseudo-flow:

```python
def tick(verbose=True, adapters=None, loop=None) -> int:
    profile_jobs = get_due_jobs(store=current_profile_store())
    global_jobs = get_due_jobs(store=global_store())
    # run profile jobs in-process as today
    # run global jobs through subprocess runner task
```

**Audit-required global claim flow:** This pseudo-flow must not call `get_due_jobs(global_store())` without a cross-process claim. Add a file-lock helper using `store.lock_file`, e.g. `with_store_file_lock(store, callback)`, and claim global jobs atomically:

```python
def claim_due_jobs(store: CronStore) -> list[dict]:
    def _claim():
        due = get_due_jobs(store=store)
        for job in due:
            # Make the job not due before launching so another profile gateway cannot claim it.
            advance_next_run(job["id"], store=store)
            # For long-running/one-shot crash recovery, prefer durable claim fields
            # such as claimed_at, claimed_by, claim_expires_at.
        return due
    return with_store_file_lock(store, _claim) or []
```

Profile jobs continue using the current profile lock. Global jobs use `global_store().lock_file`; in-process `threading.Lock` is not sufficient.

**Step 4: Test duplicate prevention**

Add a test that simulates two ticks acquiring the global lock. If true concurrency is hard in unit tests, test the lock file/store lock functions and add an integration smoke test later.

**Step 5: Run tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'due or lock or duplicate or oneshot' -o 'addopts=' -q
```

**Step 6: Commit**

```bash
git add cron/jobs.py cron/scheduler.py tests/cron/test_global_cron.py
git commit -m "feat: include global cron jobs in scheduler tick"
```

**Rationale:** Multiple gateways may be running across profiles. Global jobs need one shared lock so they do not fire once per profile.

---

### Task 9: Add `cron run-internal` subprocess entrypoint

**Objective:** Execute a global job with runtime profile config while reading/writing job metadata in the explicit global store.

**Files:**

- Modify: `hermes_cli/cron.py` for handler/dispatch
- Modify: `hermes_cli/main.py` for cron argparse wiring
- Modify: `cron/scheduler.py`
- Test: `tests/cron/test_global_cron.py` or a new CLI test file

**Step 1: Add CLI function**

Before adding the CLI entrypoint, factor current private job processing from `tick()` into an importable helper in `cron/scheduler.py`:

```python
def process_job(
    job: dict,
    *,
    store: CronStore | None = None,
    adapters=None,
    loop=None,
    verbose: bool = True,
) -> bool:
    success, output_doc, final_response, error = run_job(job)
    output_file = save_job_output(job["id"], output_doc, store=store)
    # deliver final_response using the runtime profile config/adapters policy
    # record delivery_error if any
    mark_job_run(job["id"], success, error, delivery_error=delivery_error, store=store)
    return success
```

Then in `hermes_cli/cron.py`:

```python
def cron_run_internal(args):
    from cron.jobs import store_from_root, get_job
    from cron.scheduler import process_job

    store = store_from_root("global", args.store_root)
    job = get_job(args.job_id, store=store)
    if not job:
        print(f"Job not found: {args.job_id}", file=sys.stderr)
        return 1
    return 0 if process_job(job, store=store, adapters=None, loop=None) else 1
```

`cron run-internal` owns the full global transaction: run, save output, deliver, mark success/failure in the explicit global store. Parent scheduler stdout/stderr capture is diagnostic only.

**Step 2: Wire argparse**

Find the main cron argparse setup and add a hidden/internal subcommand:

```python
run_internal = cron_subparsers.add_parser("run-internal", help=argparse.SUPPRESS)
run_internal.add_argument("--scope", choices=["global"], required=True)
run_internal.add_argument("--store-root", required=True)
run_internal.add_argument("--job-id", required=True)
run_internal.set_defaults(func=cron_run_internal)
```


**Audit-required hardening:** Validate `--scope` is exactly `global`; validate `--job-id` is an unscoped canonical job ID; validate `--store-root` resolves to the expected global root or an explicitly allowed test root; reject symlink/path traversal roots; never use `shell=True`; clear inherited `HERMES_SESSION_*` and cron delivery spoofing env/context before child execution.

**Step 3: Add subprocess launcher**

In `cron/scheduler.py`, add a subprocess launcher that prefers the current Python/current installation over `PATH` lookup:

```python
def _run_global_job_subprocess(job: dict, store: CronStore) -> tuple[bool, str, str, Optional[str]]:
    profile = job.get("run_as_profile")
    profile_home = resolve_profile_home(profile)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(profile_home)
    for key in list(env):
        if key.startswith("HERMES_SESSION_"):
            env.pop(key, None)

    cmd = [
        sys.executable, "-m", "hermes_cli.main",
        "cron", "run-internal",
        "--scope", "global",
        "--store-root", str(store.root),
        "--job-id", job["id"],
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=_get_global_subprocess_timeout())
    ok = proc.returncode == 0
    combined = redact_sensitive_text((proc.stdout or "") + (proc.stderr or ""))
    return ok, combined, combined, None if ok else combined[-2000:]
```

If a module entrypoint is unavailable, add one or use an exact current-install executable path. Use `shutil.which("hermes")` only as a last resort and test PATH-missing/wrong-PATH cases. Do not fallback to in-process execution if the subprocess fails.

**Step 4: Add tests**

Tests must assert:

- command contains `cron run-internal`
- env contains `HERMES_HOME=<profile_home>`
- command contains explicit `--store-root <global_root>`
- missing executable records failure
- invalid profile records failure

**Step 5: Run tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -k 'run_internal or subprocess or run_as_profile' -o 'addopts=' -q
```

**Step 6: Commit**

```bash
git add hermes_cli/cron.py cron/scheduler.py tests/cron/test_global_cron.py
git commit -m "feat: run global cron jobs in profile subprocesses"
```

**Rationale:** This is the most important isolation boundary. It prevents scheduler profile leakage.

---

### Task 10: Update `cronjob` tool API for global jobs while preserving scanners

**Objective:** Let agents create/list/mutate global jobs explicitly through the `cronjob` tool.

**Files:**

- Modify: `tools/cronjob_tools.py`
- Test: `tests/tools/test_cronjob_tools.py` and/or `tests/cron/test_global_cron.py`

**Step 1: Extend schema**

Add optional parameters:

```python
scope: Optional[str] = None
run_as_profile: Optional[str] = None
```

Tool docstring must state:

- omit `scope` for profile jobs;
- set `scope="global"` only when the job should be shared;
- global jobs require `run_as_profile`;
- use `global:<id>` for update/pause/resume/remove/run.

Update the registry handler at the bottom of `tools/cronjob_tools.py` to pass:

```python
scope=args.get("scope"),
run_as_profile=args.get("run_as_profile"),
```

For `scope="global"`, validate script paths against `resolve_profile_home(run_as_profile) / "scripts"`, not current `get_hermes_home() / "scripts"`. The validator must not create the creator profile’s scripts directory as a side effect.

**Step 2: Keep prompt scanning before creation/update**

Do not move or remove:

```python
scan_error = _scan_cron_prompt(prompt)
```

Apply it to global jobs exactly as profile jobs.

**Step 3: Pass scope fields to `create_job`**

```python
job = create_job(..., scope=scope or "profile", run_as_profile=run_as_profile, ...)
```

**Step 4: For list action, decide API shape**

Default:

```python
cronjob(action="list")  # profile only
```

Optional:

```python
cronjob(action="list", scope="visible")  # profile + global
cronjob(action="list", scope="global")   # global only
```

If adding `visible` is too much for v1, expose only `scope="global"` and leave dashboard visible list to web API.

**Step 5: Tests**

Add tests:

- malicious prompt is blocked for global create;
- absolute script path is blocked for global create;
- create global without `run_as_profile` fails;
- create global with invalid `run_as_profile` fails;
- list default does not include global;
- list with explicit global/visible includes global.

**Step 6: Run tests**

```bash
./venv/bin/python -m pytest tests/tools/test_cronjob_tools.py tests/cron/test_global_cron.py -o 'addopts=' -q
```

**Step 7: Commit**

```bash
git add tools/cronjob_tools.py tests/tools/test_cronjob_tools.py tests/cron/test_global_cron.py
git commit -m "feat: expose global cron jobs through cronjob tool"
```

**Rationale:** Agents are the primary creators of cron jobs, so tool-level validation must be complete.


**Security scanner matrix:**

- Tool create/update: `_scan_cron_prompt` must run for global and profile jobs.
- CLI create/update must route through the same scanner path or explicitly call the same validator.
- Dashboard/API create/update must route through the same scanner path or explicitly call the same validator.
- `run-internal` must use existing `run_job()` prompt assembly so `_scan_assembled_cron_prompt` still runs after skills/scripts/context injection.
- Global update of prompt/script/context_from/deliver must receive the same validation as global create.

**Global delivery semantics:**

- Global job output and metadata are stored in the global store.
- Actual delivery is performed inside `run-internal` using the runtime profile’s config.
- Parent gateway live adapters are not used for global subprocess jobs unless an explicit authenticated adapter bridge is designed.
- For v1, default `deliver` for global jobs is `local` unless the caller explicitly sets a delivery target.
- If storing `origin` for global jobs, document that origin metadata enters the global store; otherwise reject `deliver="origin"` for global jobs created outside the runtime profile.
- Dashboard/API must not expose origin/chat_id fields for global jobs unless explicitly authorized.

**Mutation rules:**

- Naked IDs only operate on the current profile store.
- `global:<id>` is required for every global mutation: update, pause, resume, remove, trigger/run.
- Updates must not allow changing `scope`.
- Updates must not allow changing `run_as_profile` unless the same authorization checks as creation pass.
- `profile:<name>:<id>` is rejected in v1 except for explicitly admin-only code paths.

---

### Task 11: Update CLI list/create/mutate UX

**Objective:** Add explicit global cron CLI support without changing profile default commands.

**Files:**

- Modify: `hermes_cli/cron.py`
- Modify: CLI argparse setup file if elsewhere
- Test: CLI tests if present

**Step 1: Add flags**

For `hermes cron create`:

```bash
--global
--run-as-profile PROFILE
```

For list:

```bash
hermes cron list            # profile only
hermes cron list --global   # global only
hermes cron list --visible  # current profile + global
```

For mutation:

```bash
hermes cron pause global:<id>
hermes cron resume global:<id>
hermes cron remove global:<id>
hermes cron run global:<id>
```

**Step 2: Print scope in list output**

When job has `job_ref` or `scope`, show:

```text
ID: global:abc123 [global]
Run as: deepseek
```

Profile jobs can keep current compact display.

**Step 3: Tests**

Add tests that CLI helper calls use the right tool args. If no CLI test harness exists, test `cron_list` and `cron_create` with monkeypatched `cronjob` tool.

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/hermes_cli/test_cron.py -o 'addopts=' -q
```

**Step 5: Commit**

```bash
git add hermes_cli/cron.py tests
git commit -m "feat: add explicit global cron CLI controls"
```

**Rationale:** Operators need a clear way to distinguish profile/private from global/shared jobs.

---

### Task 12: Update dashboard/API visible cron behavior

**Objective:** Make the dashboard intentionally show the current profile’s jobs plus global jobs, not all private profiles.

**Files:**

- Modify: `hermes_cli/web_server.py` for `/api/cron/jobs` endpoints
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/pages/CronPage.tsx`
- Test: `tests/hermes_cli/test_web_server.py`

**Step 1: Locate API endpoints**

Run:

```bash
python - <<'PY'
from pathlib import Path
for p in Path('.').rglob('*.py'):
    txt = p.read_text(errors='ignore')
    if '/api/cron/jobs' in txt or 'api/cron/jobs' in txt:
        print(p)
PY
```

Do not use `grep` for this step in Hermes tool calls; use Python or `search_files`.

**Step 2: Change endpoint list behavior intentionally**

Dashboard `GET /api/cron/jobs` should call `list_visible_jobs()` and return `scope`, `job_ref`, and `run_as_profile` fields.

**Step 3: Change mutations to require scoped refs**

Dashboard should send `job.job_ref || job.id` for pause/resume/trigger/delete.

**Step 4: Update TypeScript type**

In `web/src/lib/api.ts`, extend `CronJob`:

```ts
scope?: "profile" | "global"
job_ref?: string
run_as_profile?: string
```

**Step 5: Update page display**

In `web/src/pages/CronPage.tsx`, display a badge for global jobs and show `Run as: <profile>`.

**Step 6: Tests/build**

Run:

```bash
pnpm --dir web build
pnpm --dir web lint
./venv/bin/python -m pytest tests/hermes_cli/test_web_server.py -k 'cron' -o 'addopts=' -q
```

Adjust commands if the repo uses a different package script.

**Step 7: Commit**

```bash
git add web/src/lib/api.ts web/src/pages/CronPage.tsx <api-test-files> <server-api-file>
git commit -m "feat: show global cron jobs in dashboard"
```

**Rationale:** Dashboard users need visibility into shared jobs, but it must not become a private profile browser.

---

### Task 13: Add update-repair workflow script

**Objective:** Make local update durability repeatable until the feature is upstreamed.

**Files:**

- Create: `scripts/repair-local-global-cron-after-update.py`
- Add tests if script logic is importable/testable
- Document in `docs/plans/2026-05-10-hermes-update-durable-global-cron.md` or a dedicated ops doc

**Step 1: Define script responsibilities**

The script should:

1. verify repo path;
2. verify current branch is `main` or a known integration branch;
3. fetch upstream;
4. create/update `feature/update-durable-global-cron` branch;
5. rebase or merge upstream;
6. detect conflicts and stop;
7. run required tests;
8. if tests pass, print exact install/restart commands;
9. if tests fail, stop without restarting services.

Before any git operation, the script must verify the repo root, refuse unresolved conflicts, require a clean worktree or stop, list untracked files and stop unless an explicit override is passed, and print current branch, upstream SHA, feature branch SHA, stash refs, and touched files. The script must never run install or restart commands automatically; it may only print them after all tests pass.

**Step 2: Implement detection checks**

Required symbols/tests:

```python
REQUIRED_PATTERNS = {
    "cron/jobs.py": ["class CronStore", "def global_store", "def list_visible_jobs"],
    "hermes_cli/cron.py": ["run-internal", "--store-root"],
    "tools/cronjob_tools.py": ["run_as_profile", "scope"],
    "tests/cron/test_global_cron.py": ["test_global_create_requires_run_as_profile"],
}
```

**Step 3: Required test commands**

The script should run:

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -o 'addopts=' -q
./venv/bin/python -m pytest tests/tools/test_cronjob_tools.py -o 'addopts=' -q
./venv/bin/python -m pytest tests/cron -o 'addopts=' -q
./venv/bin/python -m pytest tests -k 'cron or session_search or account_usage' -o 'addopts=' -q
```

Add web build tests only if dashboard files are touched:

```bash
pnpm --dir web build
```

**Step 4: Refuse unsafe install**

If any command fails, print:

```text
REFUSING TO INSTALL: required tests failed. Active Hermes remains upstream main.
```

**Step 5: Commit**

```bash
git add scripts/repair-local-global-cron-after-update.py
git commit -m "chore: add global cron update repair workflow"
```

**Rationale:** This is the bridge between local branch and true upstream permanence.

---

### Task 14: Resolve the autostash work separately

**Objective:** Prevent unrelated account usage/session search changes from contaminating the global cron branch.

**Files from stash:**

- `agent/account_usage.py`
- `tests/test_account_usage.py`
- `tests/tools/test_session_search.py`
- `tools/session_search_tool.py`
- `tui_gateway/server.py`

**Step 1: Create separate branch**

```bash
git switch main
git switch -c feature/recover-usage-session-search-20260510
git stash apply a579938a538fcdde9ec7088ea146b4446c3bc7a5 || true
```

**Step 2: Split into logical commits**

Resolve and commit as separate changes:

1. account usage API updates;
2. TUI gateway usage-left serialization;
3. session search fixes;
4. tests.

**Step 3: Run relevant tests**

```bash
./venv/bin/python -m pytest tests/test_account_usage.py -o 'addopts=' -q
./venv/bin/python -m pytest tests/tools/test_session_search.py tests/tools/test_session_search_fix_tests.py -o 'addopts=' -q
./venv/bin/python -m pytest tests/test_tui_gateway_server.py -o 'addopts=' -q
```

**Step 4: Merge only after global cron branch is stable**

Do not merge this branch into `feature/update-durable-global-cron` unless a dependency is proven.

**Rationale:** Smaller branches reduce update conflicts and make upstream review feasible.

---

### Task 15: Full verification gate before installing/running locally

**Objective:** Prove the feature is safe enough to run under Hermes services.

**Files:** all touched files.

**Step 1: Run focused tests**

```bash
./venv/bin/python -m pytest tests/cron/test_global_cron.py -o 'addopts=' -q
./venv/bin/python -m pytest tests/cron -o 'addopts=' -q
./venv/bin/python -m pytest tests/tools/test_cronjob_tools.py -o 'addopts=' -q
```

**Step 2: Run integration-relevant tests**

```bash
./venv/bin/python -m pytest tests -k 'cron or gateway or config or constants' -o 'addopts=' -q
```

**Step 3: Run frontend build if web was touched**

```bash
pnpm --dir web build
```

**Step 4: Run Hermes smoke checks from the branch**

```bash
./venv/bin/python -m pip install -e .
hermes --version
hermes cron list --all
hermes status --all
```

**Step 5: Run profile-scoped smoke test in temp profile only**

Use a temp `HERMES_HOME`, not real user cron jobs:

```bash
TMP_HOME=$(mktemp -d)
HERMES_HOME="$TMP_HOME" hermes cron create 30m --name smoke --prompt 'Say smoke test' --deliver local
HERMES_HOME="$TMP_HOME" hermes cron list --all
```

Expected: profile job created/listed; no global store involved.

**Step 6: Run global smoke test with harmless local profile**

Only after tests pass, with a known profile such as `deepseek` or `chat_d3c3n`:

```bash
hermes cron create 30m --global --run-as-profile chat_d3c3n --name global-smoke --prompt 'Say global smoke test' --deliver local
hermes cron list --visible --all
hermes cron remove global:<id>
```

Expected: global job appears with scope/run-as; removal requires `global:<id>`.

**Step 7: Commit final verification notes**

If verification reveals docs-only adjustments, commit them:

```bash
git add docs tests
git commit -m "docs: record global cron verification gate"
```

**Rationale:** Never restart/install a cron/security feature without focused and smoke tests.

---

### Task 16: Upstreaming and long-term permanence

**Objective:** Make the feature truly permanent by eliminating the local-only patch branch.

**Files:** PR metadata and docs.

**Step 1: Prepare PR branch**

```bash
git switch feature/update-durable-global-cron
git rebase origin/main
```

**Step 2: Run all required gates again**

Run the exact commands from Task 15.

**Step 3: Open PR**

Use a PR description with:

- problem statement: profile-scoped cron only and update fragility;
- architecture summary: additive global store and subprocess profile isolation;
- security summary: scanners preserved and global jobs require `run_as_profile`;
- test evidence: commands and pass/fail output;
- migration note: existing profile jobs unchanged.

**Step 4: Keep local branch only until merged**

Once upstream merges, delete the local branch and remove/update the repair workflow if no longer needed.

**Rationale:** The only true permanent fix is upstream inclusion. A local branch is durable operationally, but still requires maintenance.

---

## Independent audit requirements before implementation

Before coding beyond RED tests, request at least two independent reviews:

1. **Security/reliability review:** Focus on subprocess isolation, prompt scanners, script path validation, delivery target validation, locking, and failure modes.
2. **Spec/API review:** Focus on profile/default compatibility, scoped refs, dashboard semantics, and update durability.

Reviewers must answer:

- Does this preserve current profile cron behavior?
- Can a naked ID mutate a global job?
- Can a global job run without valid `run_as_profile`?
- Can the subprocess accidentally write output to the runtime profile instead of the global store?
- Are upstream security scanners preserved for global jobs?
- Are update conflicts detected/refused rather than silently installed?

Audit findings must be patched into this plan before implementation proceeds.

---

## Independent Team Audit Results

**Audit status:** Completed with blockers; blockers have now been incorporated into the main implementation tasks and audit-integrated requirements above. Keep this section as the audit trail and verification checklist.

**Auditors:**

1. **Security/reliability auditor:** reviewed cron security, subprocess isolation, `run_as_profile` boundaries, import-time `HERMES_HOME` behavior, prompt/script/auth-header scanners, locking/concurrency, delivery/output routing, and rollback safety.
2. **Implementation/TDD feasibility auditor:** reviewed task ordering, RED/GREEN sequencing, current repo tooling, branch/stash workflow, command correctness, and codebase fit.
3. **Spec/API compatibility auditor:** Gemini CLI fallback review after one delegate reviewer hit provider usage limits; reviewed profile cron compatibility, scoped refs, dashboard/API semantics, and `context_from`.

### Original audit verdict and resolution

**Original verdict:** Revise before implementation. The architecture was directionally correct, but blockers had to become implementation requirements, especially around cross-profile authorization, duplicate prevention, `run-internal` transaction ownership, test isolation, and exact local tooling commands.

**Resolution in this revision:** The blockers below have been patched into the main plan in `Local execution prerequisites`, `Audit-integrated implementation requirements`, and Tasks 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, and rollback. Future implementers should still use this audit trail as a checklist during execution.

### Blocking revisions required

#### 1. Define global job transaction ownership

Task 9 must specify whether the parent scheduler or child `cron run-internal` owns the full run/save/deliver/mark transaction. The revised requirement is:

- `cron run-internal` must reconstruct the explicit global `CronStore` from `--store-root`;
- load the job from that store;
- execute the job under the runtime profile;
- save output to the global store;
- deliver using the runtime profile configuration;
- call `mark_job_run(..., store=global_store_from_root)`;
- record failures in the global store before returning nonzero.

The parent scheduler must treat child stdout/stderr as diagnostics only. It may record a global-store error only when the child cannot start, times out, or fails before it can mark the job. The plan must also correct the current `run_job()` tuple order: `success, output_doc, final_response, error`.

#### 2. Add cross-process duplicate-prevention and claim semantics

Task 8 must require a store-aware **file lock**, not just `threading.Lock`. Global due-job selection and claiming must be atomic under `global_store().lock_file`. The plan must require a helper such as `with_store_file_lock(store, callback)` and a `claim_due_jobs(store)` path that prevents two profile gateways from launching the same global job.

Required test: two simulated/concurrent profile ticks cannot both claim the same due global job. Testing only in-process `store_lock()` is insufficient.

#### 3. Add authorization for `run_as_profile`

A valid profile path is necessary but not sufficient. The plan must add an authorization boundary:

- v1 `cronjob` tool and dashboard/API global job creation must only allow `run_as_profile` equal to the current profile identity;
- cross-profile creation must be rejected unless an explicit admin-only CLI/config override is implemented and tested;
- updates must not change `run_as_profile` without the same authorization checks.

Required tests: profile A cannot create/update a global job running as profile B through normal tool/API paths; profile A can create one running as profile A.

#### 4. Replace `_profile_home()` sketch with canonical secure profile resolution

Task 6 must use existing Hermes profile layout rules. `resolve_profile_home(profile_name)` must support the default profile correctly, reject traversal/control/path separator characters, reject symlink escapes, and return a canonical path used by subprocess launch and global script validation.

#### 5. Validate global scripts against the runtime profile

Task 10 must state that for `scope="global"`, script validation is performed against `resolve_profile_home(run_as_profile) / "scripts"`, not the creator/current profile’s scripts directory. Syntax validation must still reject absolute paths, `~`, drive-letter paths, and traversal. Runtime execution must reject symlink escapes.

#### 6. Harden `cron run-internal`

Task 9 must require:

- `--scope` exactly `global`;
- unscoped canonical `--job-id`;
- `--store-root` resolves to the expected global root, or to an explicitly allowed temp root in tests;
- no `shell=True`;
- inherited `HERMES_SESSION_*` and cron delivery spoofing env/context cleared;
- subprocess launcher prefers the current Python/current installation over `shutil.which("hermes")`;
- PATH-missing and wrong-PATH cases are tested.

#### 7. Define global delivery and origin privacy semantics

The plan must add a delivery subsection:

- global metadata/output are stored in the global store;
- delivery is performed inside `run-internal` using runtime profile config;
- parent live gateway adapters are not used for cross-profile subprocess jobs unless an authenticated adapter bridge is explicitly designed;
- global jobs default to `deliver="local"`;
- `deliver="origin"` for global jobs must either be rejected outside the runtime profile or documented as storing origin metadata in the global store;
- dashboard/API must not expose origin/chat IDs for global jobs unless authorized.

#### 8. Make mutation rules exhaustive

Task 6/10/11 must specify that naked IDs operate only on the current profile store. Every global mutation must require `global:<id>`: update, pause, resume, remove, trigger/run. Updates must not change `scope`; changes to `run_as_profile` require the same authorization as creation. `profile:<name>:<id>` should be rejected in v1 except for explicit admin-only paths.

#### 9. Decide `context_from` v1 behavior now

The audit recommends a firm v1 rule: `context_from` is same-store only.

- profile jobs may reference current-profile outputs;
- global jobs may reference global outputs;
- profile→global and global→profile context references are rejected unless later explicitly authorized.

The plan must also explicitly refactor current `cron/scheduler.py` behavior: remove/static-avoid `OUTPUT_DIR` assumptions and use a store-aware helper such as `load_latest_job_output(job_ref, store=...)`. Gemini specifically confirmed this current-code gap.

#### 10. Strengthen security scanner matrix

Task 10 must add a security matrix covering tool, CLI, dashboard/API, update, and `run-internal` paths:

- create/update prompt scanner applies to profile and global jobs;
- CLI/dashboard must route through the same validator or call it explicitly;
- script path validation applies to global jobs using runtime profile;
- `run-internal` uses existing `run_job()` prompt assembly so assembled-prompt scanning still runs after skills/scripts/context injection;
- global updates of prompt/script/context/deliver get the same validation as create.

#### 11. Fix TDD test phasing

The current untracked `tests/cron/test_global_cron.py` imports future symbols at module import time. That is acceptable for the first RED failure but will block focused Task 4 collection after only `CronStore` is added. The plan must require splitting tests by phase or moving imports inside tests/classes. Recommended split:

- `tests/cron/test_cron_store.py`;
- `tests/cron/test_cron_scope_refs.py`;
- `tests/cron/test_global_cron_crud.py`;
- `tests/cron/test_global_cron_scheduler.py`.

All tests must use temp profile/global roots; no test may touch real Hermes cron stores. Remove signature-only tests for critical behavior.

#### 12. Fix local command/tooling assumptions

On this machine, `python` is not available and system `python3` lacks pytest. The plan must add a prerequisite block and use the repo venv:

```bash
cd /Users/hache/.hermes/hermes-agent
source ./venv/bin/activate
python --version
./venv/bin/python -m pytest --version
```

or use `./venv/bin/python` explicitly in every test/install command. Replace `python -m ...` examples accordingly.

Task 12 must replace the invalid `pnpm --dir web test -- --run` command; `web/package.json` has no `test` script. Use:

```bash
pnpm --dir web build
pnpm --dir web lint
./venv/bin/python -m pytest tests/hermes_cli/test_web_server.py -k 'cron' -o 'addopts=' -q
```

#### 13. Name exact code files for CLI/API wiring

The plan must explicitly name:

- `hermes_cli/main.py` for cron argparse wiring;
- `hermes_cli/cron.py` for handlers/dispatch;
- `tests/hermes_cli/test_cron.py` for CLI tests;
- `hermes_cli/web_server.py` for dashboard/API endpoints;
- `tests/hermes_cli/test_web_server.py` for web-server cron tests.

Task 10 must also explicitly update the `tools/cronjob_tools.py` registry handler to pass `scope=args.get("scope")` and `run_as_profile=args.get("run_as_profile")`.

#### 14. Strengthen stash/branch cleanup safety

Task 1’s conflict cleanup must not rely on `git merge --abort` after `git stash apply`. Use `git reset --hard HEAD` on the inspection branch after saving conflict metadata, then verify status before switching back. Avoid broad `git clean -fd` because important plan/test files are currently untracked.

#### 15. Strengthen update-repair script gates

Task 13 must require the repair workflow to:

- verify repo root;
- refuse unresolved conflicts;
- require a clean worktree or stop;
- list untracked files and stop unless an explicit override is passed;
- never install/restart automatically;
- print current branch, upstream SHA, feature branch SHA, stash refs, and touched files.

#### 16. Strengthen rollback plan

Rollback must stop/kickstop affected gateway services before switching code, back up any global cron store, show `git clean -nd` before removing untracked files, reinstall upstream, restart only explicitly selected gateway profiles, and verify that profile cron still works while global jobs are inert under upstream main.

### Non-blocking recommendations

- Prefer `--scope global` as the primary CLI over `--global`, with `--global` only as an alias if desired.
- Add durable claim fields such as `claimed_at`, `claimed_by`, and `claim_expires_at` for clearer crash recovery.
- Define a separate global subprocess timeout and redact subprocess diagnostics before storing errors.
- Include `scope`, `job_ref`, and `run_as_profile` in `_format_job()` tool responses.
- Avoid exposing all profile names in dashboard/API; only show what the current profile is authorized to see.

### Revised implementation order

1. Patch this plan with the audit blockers.
2. Add local execution prerequisites and exact file/test targets.
3. Split/repair RED tests with strict temp-store isolation.
4. Implement `CronStore` and store-aware load/save.
5. Implement secure profile resolution and authorization checks before global create/update.
6. Implement scoped refs and same-store `context_from` policy.
7. Implement store-aware output/context helpers.
8. Implement cross-process global claim/lock semantics.
9. Factor `process_job(..., store=...)` before adding `run-internal`.
10. Add hardened `run-internal` and subprocess launcher.
11. Update cronjob tool, CLI, and dashboard/API through shared validation paths.
12. Add update-repair and rollback workflow hardening.
13. Run full focused gates and only then consider local installation/upstream PR.

### Revised acceptance criteria additions

- `run-internal` owns and tests run/save/deliver/mark for global jobs.
- Two concurrent profile gateways cannot claim the same global job.
- Unauthorized cross-profile `run_as_profile` is rejected.
- Global script validation uses runtime profile, not creator profile.
- `context_from` is same-store only in v1.
- Dashboard/API hides origin/chat IDs for global jobs unless authorized.
- All test commands use the repo venv.
- Update repair refuses dirty/untracked/conflicted worktrees and never restarts services automatically.

---

## Rollback plan

If implementation breaks Hermes CLI, gateway, or cron:

1. Stop using the feature branch.
2. Stop/kickstop affected gateway services before switching code. Do not assume only `chat_d3c3n` is affected; identify active profile services first.
3. Back up any global cron store if it exists, for example `cp ~/.hermes/cron/jobs.json ~/.hermes/cron/jobs.json.global-cron-backup-$(date +%Y%m%d-%H%M%S)`.
4. Switch source checkout back to upstream `main`:

```bash
cd /Users/hache/.hermes/hermes-agent
git switch main
git reset --hard origin/main
```

5. Reinstall upstream Hermes if needed:


Show untracked cleanup candidates before removing anything:

```bash
git clean -nd
```

Only run targeted `git clean` commands after confirming they do not remove saved plans/tests or unrelated user files.

```bash
./venv/bin/python -m pip install -e /Users/hache/.hermes/hermes-agent
```

6. Restart the relevant gateway profile:

```bash
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway-chat_d3c3n
```

7. Verify:

```bash
hermes --version
hermes status --all
hermes cron list --all
```

**Rationale:** Cron/gateway health is more important than keeping local features active.

---

## Final implementation checklist

- [ ] Plan committed before code.
- [ ] Autostash recovery handled on separate branch.
- [ ] Global cron branch starts from updated `main`.
- [ ] RED tests committed before production code.
- [ ] `CronStore` supports profile/global roots.
- [ ] Default `list_jobs` remains profile-only.
- [ ] `list_visible_jobs` returns current profile + global only.
- [ ] Global jobs require valid `run_as_profile`.
- [ ] Global jobs run only in subprocess with explicit `HERMES_HOME` and `--store-root`.
- [ ] `cron run-internal` uses the explicit store root.
- [ ] Global output writes to global output dir.
- [ ] `context_from` is store-aware or explicitly rejected cross-store.
- [ ] Naked IDs cannot mutate global jobs.
- [ ] Prompt/script/security scanners apply to global jobs.
- [ ] Dashboard/API visibly labels global jobs.
- [ ] Update-repair workflow refuses unsafe installs.
- [ ] Focused cron/tool tests pass.
- [ ] Web build passes if dashboard touched.
- [ ] Hermes CLI/gateway smoke checks pass.
- [ ] PR/upstream path prepared.

---

## Execution handoff

Plan complete. Implementation should proceed with strict TDD and subagent-driven-development:

1. one task per branch/commit;
2. RED test first;
3. minimal GREEN implementation;
4. focused tests;
5. spec-compliance review;
6. code-quality/security review;
7. commit;
8. continue.

Do not skip the independent audit before touching cron/security production code.
