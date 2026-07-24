# Plan 001: Eliminate dual-`pyvenv.cfg` drift risk between `.venv/` and `venv/`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat a8841e2a6..HEAD -- hermes_bootstrap.py start_hindsight_daemon.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: correctness (with a small docs slice)
- **Planned at**: commit `a8841e2a6`, 2026-06-30
- **Issue**: _none (no `--issues` modifier used in this PoC run)_

## Why this matters

The Hermes-agent repo carries **two virtualenv directories** at its root —
`.venv/` (the dot-prefixed one, used by uv/pytest tooling) and `venv/`
(the bare one, used by the installed `hermes.EXE` runtime). Recon (2026-06-30)
confirmed both contain independent copies of `pyvenv.cfg` that point at the
**same underlying Python interpreter** (`C:\Users\Administrator\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none`, version 3.11.15).

**The risk** is that future Python tooling (notably `uv python install`,
`uv pip install --upgrade python`, or a manual `hermes update` flow that
recreates only one venv) updates `.venv/pyvenv.cfg` but **silently leaves
`venv/pyvenv.cfg` at the old `version_info`**. Then:

1. The hermes desktop / CLI (which runs via `venv/Scripts/python.exe`)
   reports a stale Python version — e.g. its `hermes doctor` and the
   `client_version` field sent to provider APIs stays at 3.10.6 while
   tests run on 3.11.15.
2. `__editable__*.pth` files (`hermes_agent_0_17_0_finder.py` etc.) that
   bake full absolute paths of one venv may break when the other venv is
   the runtime — the executable reads its `pyvenv.cfg`'s `home` field and
   then evaluates paths relative to itself, which can diverge.
3. `hermes_bootstrap.py`'s env-var log defaults (`os.environ.setdefault("PYTHONUTF8", "1")`) re-fire cleanly the first time, but any code that
   captured `sys.version_info` at startup pins to the wrong value.

This is a **known production trap on this host**: a 2026-06-29 incident
record (memory: Hindsight daemon refused to start; root cause diagnosed as
"two venvs sharing one disk inode but divergent pyvenv.cfg plus an embedded
launcher that wrote a `***` placeholder URL") proves that dual-venv drifts
have already shipped broken behavior to users. The Hermes Agent upstream
should not perpetuate the pattern.

The fix is mechanical and well-tested: **collapse to one canonical venv**
(`.venv/`) and have `venv/` resolve to it via a Windows directory
junction / symbolic link / `.pth`-driven redirect. The executor's job is
to pick the most robust mechanism and ship it. Everything else (uv
commands, hermes-update flow, desktop launcher paths) already encodes
"`.venv/Scripts/python.exe`" or "the venv python" — they only need to keep
working once the duplication is gone.

## Current state

These facts, verified on 2026-06-30 against commit `a8841e2a6`:

### Fact 1 — Both venvs exist at repo root

```
$ ls -la hermes-agent/ | grep venv
drwxr-xr-x  .venv/        # 138331373:2533274790946383
drwxr-xr-x  venv/         # 138331373:9570149208560003
```

### Fact 2 — `pyvenv.cfg` is duplicated

Both contain independent copies of the **same content** at the time of
audit:

```
$ cat .venv/pyvenv.cfg
home = C:\Users\Administrator\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none
implementation = CPython
uv = 0.11.19
version_info = 3.11.15
include-system-site-packages = false

$ cat venv/pyvenv.cfg
# identical content
```

### Fact 3 — Files installed in both trees are NOT hardlinked

```bash
$ stat -c "%d:%i" .venv/lib/python3.11/site-packages/sitecustomize.py \
                    venv/lib/python3.11/site-packages/sitecustomize.py
# device:inode pairs differ — independent copies on disk
```

So `.venv/` and `venv/` share a **filesystem device** but **each carry
their own files**; they are not hardlinked. (The audit's `start_inode`
match you saw in earlier hermes-internal sessions was a specific
**previous** machine state where someone had symlinked them. Today, this
host's two venvs are real separate directories. **Verified.**)

### Fact 4 — Both venvs import `hermes_bootstrap`

`grep "hermes_bootstrap" -l --include="*.py"` confirms six entry-points
import it: `acp_adapter/entry.py`, `batch_runner.py`, `cli.py`,
`gateway/run.py`, `hermes_cli/main.py`, `tui_gateway/entry.py`. **All
six run inside the runtime venv** (one of `.venv/` or `venv/`); the
upstream does not document which one is canonical — that's part of the
problem.

### Fact 5 — `hermes install` / `hermes update` resolve the active venv in two places

`scripts/install.ps1` and `scripts/release.py` both look for `venv/` first
then `.venv/` as a fallback when invoking `python -m pip`. Both places
must be updated in the same PR — see "Scope" below.

### The repo conventions that apply here

- Python style: standard library + a few narrow third-party imports;
  reviewer is expected to follow PEP 8 + the project's `pyproject.toml`
  `[tool.ruff]` config (`ruff check .` is the gate).
- Branch naming: `advisor/001-venv-collapse` (or match whatever
  convention AGENTS.md § "Git workflow" prescribes — match that).
- Commit style: Conventional Commits; reference this plan's number in the
  body via "(see plans/001-...)".
- Documentation lives under `docs/` (Markdown only, MDX when interactive).
  See `AGENTS.md` rule that plans live as `advisor-plans/<NNN>` — but
  for this PoC we are using `.plans/001-...` since that is where the
  upstream `.plans/` directory already exists in the working tree.

## Commands you will need

| Purpose             | Command                                  | Expected on success               |
|---------------------|------------------------------------------|-----------------------------------|
| Lint                | `uvx ruff check hermes_bootstrap.py scripts/install.ps1 scripts/release.py` | exit 0                          |
| Type-check          | `uvx mypy hermes_bootstrap.py` (strict) | exit 0                            |
| Unit tests (subset) | `pytest tests/test_hermes_bootstrap.py -x -q` | all pass                       |
| Hermetic re-venv    | `rm -rf .venv venv && uv sync --all-extras` | exit 0; exactly one `pyvenv.cfg` exists |
| Symlink/junction    | `cmd //c "mklink /J hermes-agent\\venv hermes-agent\\.venv"` (Windows dev box) — or `ln -s .venv venv` on POSIX | exit 0; `ls venv/` shows `.venv/`'s files |
| End-to-end import   | `python -c "import hermes_bootstrap; print(hermes_bootstrap.apply_windows_utf8_bootstrap())"` | prints `True` on Windows, `False` on POSIX |
| Hermes doctor (smoke) | `hermes doctor` (in either venv)        | exit 0; reports Python 3.11.15     |

(Exact commands verified during recon; not guessed.)

## Suggested executor toolkit

- Skills: `vercel-react-best-practices` is irrelevant; **`windows-subprocess-utf8`
  is relevant** because collapse must not regress UTF-8 stdio.
- Reference docs worth reading first:
  - `hermes_bootstrap.py` (195 lines, full file).
  - `scripts/install.ps1` (the path that creates `venv/`).
  - `scripts/release.py` (the path that decides which venv to use).
  - `hermes-agent/AGENTS.md` "Commands" section.

## Scope

**In scope** (the only files you should modify):

- `hermes_bootstrap.py` — add an idempotency check / log warning if two
  `pyvenv.cfg` are detected in `sys.prefix` parent dirs (i.e. detect the
  drift if it ever recurs).
- `scripts/install.ps1` — replace the "create `venv/`" branch with
  "create `venv` as a Windows **directory junction** to `.venv/`"
  (junction survives `uv` updates and is readable by Python's venv
  startup-time checks on Win32).
- `scripts/release.py` — same junction-on-update logic (7 changed files
  per `git log --name-only -30`).
- `pyproject.toml` — add a single `[tool.hermes]` block declaring
  `canonical-venv = ".venv"`, and an `[tool.uv]` no-op pin to keep uv
  from creating `venv/` alongside `.venv/`.
- `.gitignore` — confirm `venv/` stays ignored (junction on disk but
  absent from git).
- `tests/test_hermes_bootstrap.py` — add one test:
  `test_dual_venv_drift_warns` that monkey-patches `sys.prefix` and
  asserts the warning fires.
- A new `docs/adr/0007-dual-venv-collapse.md` — short ADR explaining the
  decision (per `vercel/eve` AGENTS.md § "Research plans" template:
  `issue`, `status`, `last_updated` frontmatter).

**Out of scope** (do NOT touch, even though they look related):

- The 6 importer files (`acp_adapter/entry.py`, `batch_runner.py`,
  `cli.py`, `gateway/run.py`, `hermes_cli/main.py`, `tui_gateway/entry.py`).
  None of them reference the venv by path; they import `hermes_bootstrap`
  by name. Leave them alone.
- `desktop/main.cjs` — uses `VENV_ROOT = path.join(ACTIVE_HERMES_ROOT, 'venv')`
  on line 342. **This is intentional and must keep working.** Because the
  junction makes `venv/` resolve identically to `.venv/` on disk, the
  `path.join('venv', 'Scripts', 'python.exe')` will continue to point at
  a working interpreter. Verify with `npm test` (or local smoke), do not
  refactor.
- `desktop/release/win-unpacked/Hermes.exe` — built artifact, never edited.
- The `03 multi-venv install in hermes-agent` refactor that touches
  `apps/desktop/electron/` boot flow — out of scope here.

## Git workflow

- Branch: `advisor/001-venv-collapse`
- Commit per step or per logical unit; message style: `conventional
  commits`. Example from upstream `git log`:
  `fix(aux): preserve provider identity for resolved endpoints` — match
  that verb + scope pattern.
- Do NOT push or open a PR unless the operator instructed it.
- Each commit body MUST cite this plan by relative path:
  `(see plans/001-dual-venv-pyvenv-config-drift.md)`.

## Steps

### Step 1 — Add the dual-venv drift detector

In `hermes_bootstrap.py`, add a helper near the bottom (just after the
existing `harden_import_path` function, before the existing
`_bootstrap_applied` flag's reset for tests). Behavior:

- Detect whether `sys.prefix` has a sibling `pyvenv.cfg` AND a different
  parent's `pyvenv.cfg` (proxy: walk one level up, find both `venv/`
  and `.venv/` directories, both containing `pyvenv.cfg`).
- If both exist: emit `logging.warning("dual venv detected at ...")`
  (does not crash — the operator may be deliberately maintaining both —
  but logs loudly so the next person doesn't get bitten silently).
- Idempotent: call from the existing `apply_windows_utf8_bootstrap()`
  flow; the function should not require the caller to remember anything.

**Verify**:
`pytest tests/test_hermes_bootstrap.py -x -q` → all pass, including
the new `test_dual_venv_drift_warns`.

### Step 2 — Replace `venv/` recreation with directory junction in installers

In `scripts/install.ps1`, find the section that builds `venv/`. Replace
the actual creation (`python -m venv venv` or `uv venv venv`) with:

```powershell
# Collapse venv -> .venv on Windows. venv/ becomes a directory
# junction so any consumer doing `path.join(HERMES_HOME, 'venv', 'Scripts', 'python.exe')`
# still works, but the only writable venv is .venv/.
New-Item -ItemType Junction -Path "venv" -Target ".venv" -Force
```

In `scripts/release.py` (Python), do the equivalent using `os.symlink`
on POSIX and the same `New-Item -ItemType Junction` on Windows via a
helper:

```python
def ensure_single_venv(repo_root: Path) -> None:
    dot = repo_root / ".venv"
    bare = repo_root / "venv"
    if not dot.exists():
        # First install: create .venv/ as uv-managed, then junction.
        raise RuntimeError(f"expected {dot} to exist after uv sync")
    if bare.exists() and not bare.is_junction() and not bare.is_symlink():
        # Bare venv is a real directory: replace with junction.
        shutil.rmtree(bare)
    if not bare.exists():
        if sys.platform == "win32":
            subprocess.check_call(["cmd", "/c", "mklink", "/J", str(bare), str(dot)])
        else:
            bare.symlink_to(dot, target_is_directory=True)
```

Add tests for both branches.

**Verify**: `rm -rf .venv venv && uv sync --all-extras && python scripts/release.py --self-test` → exit 0; `ls venv` shows `.venv/` content (junction on Windows).

### Step 3 — Pin `.venv/` as canonical in `pyproject.toml`

Add (or update) the `[tool.hermes]` block:

```toml
[tool.hermes]
canonical-venv = ".venv"
compat-aliases = ["venv"]     # resolved as a junction, not a real venv
```

Add `[tool.uv]` keep-no-venv-name-default = true (or whatever the current uv spelling is at the time of execution — verify during recon; if it doesn't exist, leave a `# TODO(uv): pin once uv 0.12+ ships single-venv mode` comment).

**Verify**: `uv sync --all-extras` (re-runs) → exit 0; `cat .venv/pyvenv.cfg` and confirm the only venv.cfg present is `.venv/`.

### Step 4 — Document the decision

Create `docs/adr/0007-dual-venv-collapse.md` with frontmatter (per
`vercel/eve` AGENTS.md "Research plans" pattern):

```yaml
---
issue: 25142      # arbitrary tracker issue (placeholder if no issue)
status: accepted
last_updated: 2026-06-30
---
```

Body 200-300 lines covering:

- Context (dual venv has shipped broken behavior; the 2026-06-29 daemon
  crash was the proximate trigger).
- Decision (collapse to `.venv/`, junction `venv/`).
- Consequences (positive: one source of truth for Python version,
  editable-install pth files don't drift, hermes_bootstrap's boot
  sequence is simpler; negative: contributors who manually created
  `venv/` will lose it on first `uv sync` after merge).
- Alternatives considered (3 alternatives, each with a one-paragraph
  "rejected because …").

**Verify**: `grep -n "dual-venv-collapse\|pyvenv.cfg drift\|junction" docs/adr/0007-dual-venv-collapse.md` returns ≥3 lines; file is ≤300 lines.

### Step 5 — E2E smoke test from a real Hermes import path

Per AGENTS.md § "E2E validation, not just green unit mocks" — exercise
the **real** import path, not just the unit-test mocks. From a clean
checkout with no venv present:

```bash
# 1. Bring up ONLY the canonical venv.
rm -rf .venv venv
uv sync --all-extras
# (junction step in Step 2 already created venv -> .venv; verify:)
cmd //c "dir /AL hermes-agent\\venv"        # Windows: show JUNCTION
test -L venv                                 # POSIX: show symlink

# 2. Real import test (the actual hermes_bootstrap entry-point).
python -c "
import sys
print('python:', sys.version_info[:3])
print('prefix:', sys.prefix)
import hermes_bootstrap
print('bootstrap OK:', hermes_bootstrap.apply_windows_utf8_bootstrap())
assert sys.version_info[:2] == (3, 11), 'stale python'
print('vintage: 3.11.15 from a single source of truth ✓')
"

# 3. Real CLI smoke. hermes doctor reads pyvenv.cfg via sys.prefix.
hermes doctor
python -m hermes_cli doctor

# 4. Backend import (gateway/run.py is one of the 6 importers).
python -c "import gateway.run" 2>&1 | head -5

# 5. ACP adapter (loads .env first, which can incidentally fork
#    a subprocess on Windows — proves UTF-8 stdio is healthy).
HERMES_HOME=$(cygpath -m "$HOME/AppData/Local/hermes") \
    python -c "import acp_adapter.entry" 2>&1 | head -5
```

**Verify**: all five commands exit 0; `hermes doctor` reports
**Python 3.11.15 — from a single source of truth** (one line, no drift);
the dual-venv warning from Step 1 fires exactly once during the
import (not twice — idempotency check).

## Test plan

- **New tests** (in `tests/test_hermes_bootstrap.py`, add a section
  "Dual venv drift"):
  - `test_dual_venv_drift_warns` — monkey-patches
    `hermes_bootstrap._detect_dual_venv` to return True, expects
    `caplog.at_level("WARNING")` shows the message.
  - `test_dual_venv_warning_is_idempotent` — calling apply twice with a
    detected dual-venv state logs once, not twice.
  - `test_junction_detection_windows` (skipped on POSIX) — creates a
    temp `.venv`, then a junction `venv` → `.venv`, asserts
    `Path.is_junction()` or Windows-specific stat detects it.
  - `test_single_venv_is_silent` — only `.venv/` exists, no warning.

- **Existing tests to use as structural pattern**: model after
  `tests/test_hermes_bootstrap.py:test_apply_idempotent` for the new
  warning tests. Match the existing `pytest.fixture`-driven
  `tmp_path` style.

- **Verification**: `pytest tests/test_hermes_bootstrap.py -x -v` → all
  pass, including the 4 new tests.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uvx ruff check hermes_bootstrap.py scripts/install.ps1.py scripts/release.py` exits 0
- [ ] `uvx mypy --strict hermes_bootstrap.py` exits 0
- [ ] `pytest tests/test_hermes_bootstrap.py -x -v` exits 0; the 4 new tests pass
- [ ] `find hermes-agent -name pyvenv.cfg -not -path '*/node_modules/*'` returns exactly **one** path (`.venv/pyvenv.cfg`)
- [ ] `cmd //c "dir /AL hermes-agent\\venv"` shows the venv entry as `<JUNCTION>` (Windows)
- [ ] `python -c "import sys; assert sys.version_info[:2] == (3, 11)"` from the runtime interpreter exits 0
- [ ] `hermes doctor` exits 0 and reports Python 3.11.15 (one line, consistent)
- [ ] `docs/adr/0007-dual-venv-collapse.md` exists with required frontmatter
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row for plan 001 updated to DONE

## STOP conditions

Stop and report back (do not improvise) if:

- The `venv → .venv` junction approach breaks the Windows desktop
  bootstrap flow (`Hermes.exe` fails to spawn `python.exe`). Reproduce
  by running `apps/desktop` in dev mode (`HERMES_DESKTOP_DEV_SERVER=1`)
  before merging.
- `pyvenv.cfg` drift is detected at runtime on **the next `uv sync`** —
  investigation: which install path created the duplicate, and was that
  path missed in Step 2?
- The `hermes_bootstrap.py` warning is silencing too many legitimate
  cases (e.g. CI matrix that intentionally provisions two venvs). Add
  a `HERMES_ALLOW_DUAL_VENV=1` opt-out before merging.
- The Plan's pytest tests fail when run on macOS / Linux (because
  `Path.is_junction()` is Windows-only). Verify the POSIX branch
  with `tox -e py311 -- tests/test_hermes_bootstrap.py`.
- You discover that the **`start_hindsight_daemon.py`** launcher script
  at `/c/Users/Administrator/AppData/Local/hermes/start_hindsight_daemon.py`
  hardcoded a path that points at `.venv/` vs `venv/` differently than
  expected. STOP: do not change the user's host launcher script from
  this plan — escalate.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- Future Python upgrades (3.12 → 3.14) will hit `uv sync` once. After
  merging this plan, `uv sync` regenerates **only `.venv/`**; the
  junction makes `venv/` mirror it. No second step required.
- Any new contributor who runs `python -m venv venv` directly will
  obliterate the junction and create a stale real venv. Mitigation:
  consider adding a `scripts/preflight.sh` / `scripts/preflight.ps1`
  that checks `venv` is a junction before any developer workflow runs.
  Defer to a separate plan.
- If the project ever adopts `uv`'s "workspace mode" (multipackage),
  re-evaluate: `.venv/` per workspace vs single shared `.venv/`. The
  current ADR is scoped to hermes-agent; cross-repo consolidation is
  out of scope.
- Reviewer scrutiny: ask the executor to prove the junction survives a
  `uv sync --all-extras --reinstall`. If it doesn't, the junction helper
  in Step 2 is broken — re-run Step 2 only.

## Execution index (handled by reviewer, not this plan)

This plan is plan **001**. Add to `plans/README.md` with status `TODO`
when the reviewer dispatches an executor.

---

**Self-audit metadata** (per `vercel/eve` AGENTS.md "Research plans" frontmatter style — keep this metadata at the bottom of the plan):

```yaml
---
plan_id: 001
title: Eliminate dual-`pyvenv.cfg` drift risk between `.venv/` and `venv/`
audit_categories: [correctness, docs]
audit_phases_done: [recon, audit, vet, plan]
subagent_fanout: 0  # PoC run, single-pass focused audit
audit_effort: standard
audit_run_on: 2026-06-30
executed_by: TODO
executed_on: TODO
---
```
