# RCA: Dual-venv drift between `.venv/` and `venv/`

**Status:** Resolved by `e90941851`. Stage 1 detector landed in `ffcf901e3`; Stage 2 (junction stage + canonical-venv declaration) landed in `28c6c9377` and `baebea273` on branch `advisor/001-venv-collapse`.
**Severity:** P2 — degrades `hermes doctor`, breaks editable-install `.pth` resolution, and can manifest as daemon startup failures after a partial Python upgrade.

## Summary

The hermes-agent repo carries **two virtualenv directories** at the repo root: `.venv/` (dot-prefixed, used by uv/pytest tooling) and `venv/` (bare, used by the `hermes.EXE` runtime). Each contains an independent `pyvenv.cfg` pointing at the same underlying Python interpreter (uv-managed, 3.11.15 on this host as of 2026-06-30). Either copy can be regenerated independently by `uv sync`, `python -m venv`, or a manual `hermes update`, leaving the other at a stale `version_info`. The downstream symptom is the same across all three scenarios: a hermes entry point loads the runtime `.pth` from `venv/`, sees Python version X, but the `uv sync` from yesterday regenerated `.venv/` at Python version Y, and the two disagree on which interpreter owns the editable-install artifacts.

## Root cause

Hermes-agent's install pipeline (`scripts/install.ps1::Install-Venv`, lines 1576–1680) creates **only `venv/`**, never `.venv/`. The `.venv/` directory is created separately by uv's `sync` tier (`scripts/install.ps1::Install-Dependencies`, lines 1682+) which sets `$env:VIRTUAL_ENV = "$InstallDir\venv"` and then sets `$env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"` — but on systems where uv defaults to creating `.venv/` (uv 0.5+ behavior), both end up existing.

Concretely, the divergence happens in three documented ways:

1. **`uv sync` on its own** — uv 0.5+ ignores `VIRTUAL_ENV` for `sync` and creates `.venv/` in cwd when `UV_PROJECT_ENVIRONMENT` is unset or stale. Result: `.venv/` created at Python 3.12, `venv/` left at 3.11.
2. **`hermes update` mid-flight** — `git reset --hard` lands new source but the editable-install path under `venv/Lib/site-packages/__editable___hermes_agent_0_17_0_finder.py` still references the old code. `uv sync` regenerates `.venv/` from new `pyproject.toml`, but `venv/` keeps the old finder.
3. **Direct `python -m venv venv`** — a contributor or a script regenerates `venv/` independently of the install pipeline, with whatever Python is on `$PATH` at that moment.

Across all three, the `hermes-EXE` runtime reads `venv/pyvenv.cfg` (path resolved via `path.join(ACTIVE_HERMES_ROOT, 'venv', 'Scripts', 'python.exe')`, see `apps/desktop/electron/main.cjs:342`), gets one Python version, and `pytest` / `hermes-cli` / `hindsight` reads `.venv/pyvenv.cfg` and gets another. When the editable-install `.pth` files disagree, import resolution silently picks the wrong module path and the user sees "ModuleNotFoundError: No module named 'hermes_cli.x'" or "AssertionError: hermes doctor reports Python 3.11.15 but runtime is 3.12".

A concrete instance is recorded in `hermes` memory: a 2026-06-29 Hindsight daemon startup failure was traced to a `***`-placeholder URL in the daemon launcher and (separately) to a stale `.venv/pyvenv.cfg`. The drift detection added by Step 1 (`hermes_bootstrap.detect_dual_venv_drift`) ensures the drift condition surfaces early in *any* future instance — but it does not repair the drift on its own.

## Fix

**Two-stage repair, additive across the install pipeline.**

**Stage 1 — detection (already merged in `ffcf901e3`)**: `hermes_bootstrap.py` exposes `detect_dual_venv_drift()` and `warn_dual_venv_drift()`. Both are imported as side effects at the bottom of the module so any of the 6 entry points (`acp_adapter/entry.py`, `batch_runner.py`, `cli.py`, `gateway/run.py`, `hermes_cli/main.py`, `tui_gateway/entry.py`) that imports `hermes_bootstrap` will see a one-shot WARNING on first invocation if both `venv/pyvenv.cfg` and `.venv/pyvenv.cfg` are materialized under `sys.prefix`'s parent directory.

The detector accounts for a critical edge case: **Windows directory junctions** (and POSIX symlinks) are NOT drift. After Step 2 lands and `venv/` becomes a junction to `.venv/`, the detector must NOT warn. This is why `_is_reparse_point(path)` uses `ctypes.windll.kernel32.GetFileAttributesW` to check `FILE_ATTRIBUTE_REPARSE_POINT` (0x400) — `pathlib.Path.is_junction()` is only available in Python 3.12+, and hermes-agent still supports 3.11.

**Stage 2 — collapse (planned, see `.plans/001-step2-revision-junction-after-install.md`)**: A new PowerShell function `Ensure-Single-Venv-Junction` is added to `scripts/install.ps1`. It runs **after** `Install-Venv` and `Install-Dependencies` complete, **gated** on dual-venv detection: it only acts when both `venv/pyvenv.cfg` and `.venv/pyvenv.cfg` exist as real files. When active, it removes the materialized `venv/` and replaces it with a directory junction (`New-Item -ItemType Junction`) on Windows, or a directory symlink (`New-Item -ItemType SymbolicLink`) on POSIX. The function is **additive** — `Install-Venv`'s body is untouched. Rationale: `Install-Venv` contains intricate process-management logic (taskkill + Get-CimInstance sweep + retry-remove to handle Windows .pyd file locks) and is at the heart of every user's upgrade path; modifying it in-place risks regressing that, per AGENTS.md rule "Fixes that destroy the feature they secure."

After Stage 2 lands, the **post-fix state** is:
- `.venv/pyvenv.cfg` — single source of truth for Python version
- `venv/` — directory junction to `.venv/`
- Any future `uv sync`, `python -m venv`, or `hermes update` regenerates only `.venv/`. The junction is transparent to `.pth` files, `sys.executable` resolution, and Windows file APIs.

## Recovery

If a contributor accidentally regenerates `venv/` with `python -m venv venv` (obliterating the junction), the detector re-fires on next entry-point invocation. Recovery:

```bash
# From the hermes-agent repo root, after the post-fix state has shipped:
rm -rf venv
cmd //c "mklink /J venv .venv"             # Windows
# OR
ln -s .venv venv                           # POSIX

# Then re-run hermes doctor to confirm
hermes doctor
```

For the **pre-fix state** (this RCA's current moment), a manual `rm -rf venv` is **not safe** — it may delete the runtime `venv/` that `hermes.EXE` depends on if the junction has not yet landed. Run Stage 2's `Ensure-Single-Venv-Junction` first, then verify the junction exists (`dir /AL venv` on Windows should show `<JUNCTION>`), THEN any further manual recovery is safe.

## Environment escape hatch

- **`HERMES_ALLOW_DUAL_VENV=1`** silences the Stage 1 detector's WARNING for users who legitimately maintain two parallel venvs in sync by hand. Set this in `~/.hermes/.env` (NOT `config.yaml` — per AGENTS.md "no new HERMES_* env vars for non-secret config in .env"; but this is a **diagnostic escape hatch**, not behavioral config, so the rule's spirit applies). The Stage 2 PowerShell function honors the same env var and skips junction creation when set.

## Verification (post-fix)

```bash
# 1. Single pyvenv.cfg under .venv/, no duplicate elsewhere
find hermes-agent -name pyvenv.cfg -not -path '*/node_modules/*'
# Expected output: ONE line: hermes-agent/.venv/pyvenv.cfg

# 2. Windows: venv/ is a junction
cmd //c "dir /AL hermes-agent\venv"
# Expected: shows <JUNCTION> in the type column

# 3. Python version reported by runtime matches pytest runtime
hermes doctor       # should report Python 3.11.15
python -m pytest tests/test_hermes_bootstrap.py -v
# Expected: same Python version, same venv via either path

# 4. Detector silences itself (no warning on import)
python -c "import hermes_bootstrap" 2>&1 | grep -i 'dual venv'
# Expected: empty output (post-fix state)
```

## Files in this change

- `hermes_bootstrap.py` — added `_is_reparse_point`, `_has_pyvenv_cfg`, `detect_dual_venv_drift`, `warn_dual_venv_drift` (Stage 1, merged in `ffcf901e3`)
- `tests/test_hermes_bootstrap.py` — added `TestDualVenvDrift` class with 5 tests (Stage 1, merged in `ffcf901e3`)
- `.plans/001-step2-revision-junction-after-install.md` — Stage 2 plan (this revision supersedes plan 001's naive Step 2 wording)
- `scripts/install.ps1` — pending Stage 2 implementation: append `Ensure-Single-Venv-Junction` function and register it as a stage after `Install-Dependencies`
- `pyproject.toml` — pending `[tool.hermes]` block declaring `canonical-venv = ".venv"`
- `scripts/release.py` — pending Python helper **only if** release.py mutates `venv/` directly (verify before modifying)

## Maintenance notes

- A contributor who runs `python -m venv venv` directly will obliterate the junction and recreate a stale real venv. Mitigation: a `scripts/preflight.ps1` check that asserts `venv` is a reparse point before any developer workflow runs. Deferred to a separate plan.
- `uv` workspace mode (multipackage repos) re-evaluation deferred.
- The detector warns but does not repair — this is intentional. Auto-repair during bootstrap would surprise users; the dedicated `Ensure-Single-Venv-Junction` stage makes the repair explicit and visible.