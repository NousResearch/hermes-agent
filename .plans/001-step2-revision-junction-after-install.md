# Plan 001 — Step 2 Revision: Junction-after-install, NOT in-place replacement

> **Author**: This revision supersedes the Step 2 in plan 001 (`001-dual-venv-pyvenv-config-drift.md`). It was written after reading the **actual** `Install-Venv` source in `scripts/install.ps1` lines 1576–1680 — which is **104 lines**, not the ~50 estimated by plan 001.
>
> **Read this before executing**: the original plan's Step 2 wording ("replace `python -m venv venv` with `New-Item -ItemType Junction`") is a **naive simplification**. The real Install-Venv function kills running hermes processes (taskkill + Get-CimInstance) before removing the old venv. Replacing it in-place risks breaking every user's upgrade path.

## Status
- **Priority**: P1 (same parent plan)
- **Effort**: M (smaller diff than original Step 2; this revision is ~40 lines added)
- **Risk**: LOW (additive: new install stage that runs *after* Install-Dependencies; idempotent)
- **Depends on**: Step 1 already committed (`ffcf901e3` — drift detector)
- **Category**: correctness + small docs (ADR)

## Why revise

Plan 001 Step 2 said:
> "Replace `python -m venv venv` (or `uv venv venv`) with `New-Item -ItemType Junction -Path "venv" -Target ".venv" -Force`."

Three things wrong with that, **discovered by reading the source, not by guessing**:

1. **`Install-Venv` is 104 lines**, not ~50. The plan's diff estimate is **2x off**.
2. **`Install-Venv` does process management** (taskkill, Get-CimInstance, retry-remove) — replacing the whole function or its `uv venv venv` call means re-doing that kill-process logic around the junction.
3. **AGENTS.md "We don't want" rule #6**: *"'Fixes' that destroy the feature they secure."* The Install-Venv function exists to ensure hermes installs reliably across upgrades; gutting it to swap in a junction risks regressing that.

**Better design (this revision)**: do NOT touch `Install-Venv`. Add a new stage `Ensure-Single-Venv-Junction` that runs **after** `Install-Venv` and `Install-Dependencies` complete, **conditionally** — only if a real dual-venv drift condition is detected (i.e. `venv/` is a materialized dir AND `.venv/` exists, both with `pyvenv.cfg`). On existing single-venv or already-junction installs, it's a no-op.

## Current state (verified 2026-07-04)

Measured by `grep -nE "function Install-Venv|Install-Venv" scripts/install.ps1`:

| Line | What's there |
|---|---|
| 1576 | `function Install-Venv {` start |
| 1577–1580 | `$NoVenv` short-circuit |
| 1582–1592 | Python version re-resolution |
| 1594 | "Creating virtual environment..." info log |
| 1596 | `Push-Location $InstallDir` |
| 1598 | `if (Test-Path "venv")` — already-exists branch |
| 1600–1605 | Long comment explaining the kill-processes reason (.pyd locking) |
| 1606 | `if ($env:OS -eq "Windows_NT")` |
| 1608 | "Stopping any running hermes processes..." |
| 1610 | `taskkill /F /T /IM hermes.exe /FI "PID ne $myPid"` |
| 1611–1626 | Long comment explaining the Get-CimInstance cascade |
| 1629–1634 | `Get-CimInstance Win32_Process \| ... \| Stop-Process -Force` |
| 1638 | `Start-Sleep -Milliseconds 800` |
| 1640 | `Remove-Item -Recurse -Force "venv" -ErrorAction SilentlyContinue` |
| 1644–1647 | `if (Test-Path "venv") { Start-Sleep -Seconds 2; Remove-Item ... }` (retry) |
| 1654 | `Invoke-NativeWithRelaxedErrorAction { & $UvCmd venv venv --python $PythonVersion }` |
| 1659–1663 | Capture $LASTEXITCODE and fail fast |
| 1665–1675 | Pin `$env:UV_PYTHON` to `venv\Scripts\python.exe` |
| 1677 | `Pop-Location` |
| 1679 | "Virtual environment ready" success log |
| 1680 | `}` end of function |

`Install-Dependencies` (line 1682+) uses `$env:VIRTUAL_ENV = "$InstallDir\venv"` — **a string**, no path resolution. Junction is transparent to it.

## Scope (this revision)

**In scope**:

- `scripts/install.ps1` — **add** a new function `Ensure-Single-Venv-Junction` and an optional stage registration (see "Stage registration" below).
- `scripts/release.py` — Python equivalent helper, **only if** `release.py` ever creates or mutates the `venv/` directory. (Verify before modifying — see "Verify before modifying" below.)
- `pyproject.toml` — `[tool.hermes]` canonical-venv declaration (unchanged from plan 001 Step 3).
- `docs/rca-dual-venv-collapse.md` — **revised** RCA-style ADR reflecting the additive (not in-place) design. (Verified: hermes-agent's existing RCA-style docs live at `docs/*.md` directly with prefixes like `rca-` and `*-contract`. There is **no** `docs/adr/` subdirectory in this repo — plan 001's original Step 4 target was wrong on this.)

**Out of scope** (unchanged from plan 001):

- The 6 importer files (acp_adapter/entry.py, batch_runner.py, cli.py, gateway/run.py, hermes_cli/main.py, tui_gateway/entry.py).
- `apps/desktop/electron/main.cjs` (junction makes line 342 `path.join('venv', 'Scripts', 'python.exe')` continue to resolve through Windows reparse-point traversal).
- `desktop/release/win-unpacked/Hermes.exe` (built artifact).
- The body of `Install-Venv` itself — **do not touch**. The new function is purely additive.

## New function (additive, idempotent, gated)

Append after `Install-Venv` (or anywhere in install.ps1 before its stage registration):

```powershell
function Ensure-Single-Venv-Junction {
    # Collapse venv/ -> .venv/ via a Windows directory junction (POSIX symlink).
    # ADDS nothing on single-venv installs and on existing junction installs.
    # Only acts when venv/ is a *materialized* dir AND .venv/ also exists —
    # i.e. dual-venv drift has been detected on disk.

    if ($NoVenv) { return }
    if ($env:HERMES_ALLOW_DUAL_VENV -eq "1") {
        Write-Info "Ensure-Single-Venv-Junction: skipped (HERMES_ALLOW_DUAL_VENV=1)"
        return
    }

    Push-Location $InstallDir

    $bare = "venv"
    $dot  = ".venv"

    # Already-junction or symlink: nothing to do.
    if (Test-Path $bare) {
        $bareAttr = (Get-Item $bare).Attributes
        if ($bareAttr -band [System.IO.FileAttributes]::ReparsePoint) {
            Write-Info "Ensure-Single-Venv-Junction: $bare already a reparse point, no-op"
            Pop-Location
            return
        }
    }

    # .venv must exist before we can junction into it.
    if (-not (Test-Path $dot)) {
        Write-Info "Ensure-Single-Venv-Junction: $dot not present, skipping (single-venv install)"
        Pop-Location
        return
    }

    # Both must contain a real pyvenv.cfg for this to be drift. If one is a
    # junction, skip (junction-resolved cfg is excluded by the python-side
    # detector too — see hermes_bootstrap.detect_dual_venv_drift).
    $bareCfg = Join-Path $bare "pyvenv.cfg"
    $dotCfg  = Join-Path $dot  "pyvenv.cfg"
    if (-not (Test-Path $bareCfg)) {
        Write-Info "Ensure-Single-Venv-Junction: $bare/pyvenv.cfg absent, no drift to collapse"
        Pop-Location
        return
    }
    if (-not (Test-Path $dotCfg)) {
        Write-Info "Ensure-Single-Venv-Junction: $dot/pyvenv.cfg absent, no drift to collapse"
        Pop-Location
        return
    }

    # Drift confirmed. Replace materialized venv/ with a junction.
    # The kill-processes + retry-remove logic from Install-Venv does NOT
    # re-run here: by the time we get to this stage, Install-Venv has
    # already completed (so the just-built venv isn't holding any
    # imported .pyd in this same powershell.exe process), and any
    # running hermes.exe that was using the old venv was stopped by
    # Install-Venv's taskkill sweep above.
    Write-Info "Ensure-Single-Venv-Junction: collapsing $bare -> junction -> $dot"

    # Idempotent: if junction already exists, Remove-Item on a junction
    # is a no-op on the target (only the link itself). Safe.
    Remove-Item -Recurse -Force $bare -ErrorAction SilentlyContinue
    if (Test-Path $bare) {
        Start-Sleep -Milliseconds 500
        Remove-Item -Recurse -Force $bare
    }

    if ($env:OS -eq "Windows_NT") {
        # Junction: cannot traverse, doesn't follow symlinks, transparent
        # to .pth editable installs and to Windows file APIs.
        New-Item -ItemType Junction -Path $bare -Target $dot | Out-Null
    } else {
        # POSIX: directory symlink
        New-Item -ItemType SymbolicLink -Path $bare -Target $dot | Out-Null
    }

    Write-Success "Ensure-Single-Venv-Junction: $bare now a junction to $dot"

    Pop-Location
}
```

**Why this is safer than the original plan**:

| Concern | Original plan | This revision |
|---|---|---|
| Touches Install-Venv body | **Yes** | **No** |
| Process management re-implemented | Yes | Reuses Install-Venv's earlier sweep |
| Behavior on single-venv install | Would re-junction every install | No-op |
| Idempotent | Partial | Full (junction re-creation check) |
| `HERMES_ALLOW_DUAL_VENV` opt-out | (only python side) | **Also honored here** |
| Falls through if `.venv/` missing | (would silently break) | Skips with info log |

## Stage registration

In `scripts/install.ps1`, find the existing install-stage pipeline (look for where `Install-Venv` is invoked as a stage — likely in an `Install-All` or main bootstrap function). Append the new function as the **last** stage, **after** `Install-Dependencies`:

```powershell
# (existing install pipeline continues here...)
Install-Dependencies
# ... other stages ...

# NEW: collapse venv/ -> junction -> .venv/ if both materialized.
# No-op on single-venv installs and on already-junction installs.
Ensure-Single-Venv-Junction
```

Exact placement requires reading the install pipeline. **Verify before editing**.

## Verify before modifying scripts/release.py

```bash
# Read scripts/release.py and grep for venv-related code.
# If release.py ONLY uses sys.executable / venv as a runtime target
# (i.e. never creates or deletes the venv/ directory itself), then
# no Python-side helper is needed — the PowerShell stage handles
# every install path.
grep -nE "venv|/site-packages|virtualenv" scripts/release.py
```

If the grep hits anything that creates/removes the `venv/` directory, **then** add a Python `ensure_single_venv()` helper to `release.py`, mirroring the PowerShell function's idempotent / opt-out gating.

## ADR (Step 4) — Revised

Update the planned ADR to reflect the additive design:

- **Decision**: Add `Ensure-Single-Venv-Junction` stage to `install.ps1`. **Do not modify `Install-Venv`**. Junction happens post-install, gated on dual-venv detection.
- **Why**: Preserves the existing process-management guarantees of `Install-Venv` while collapsing the dual-venv drift. AGENTS.md rule #6 ("Fixes that destroy the feature they secure") applies to in-place replacement of `Install-Venv`; the additive stage avoids that.
- **Consequences**:
  - **Positive**: Single source of truth (`pyvenv.cfg` lives only in `.venv/`). Future `uv sync` and `python -m venv` regenerations update `.venv/` only; the junction is transparent.
  - **Negative**: A contributor who runs `python -m venv venv` directly will silently obliterate the junction and recreate a stale real venv. Mitigation: `scripts/preflight.ps1` check (deferred to a separate plan, mentioned in plan 001's maintenance notes).
- **Alternatives considered**:
  1. *In-place replacement of `Install-Venv`'s `uv venv venv` call.* Rejected because the function's process management is intricate and lives at the heart of every upgrade; touching it expands the blast radius.
  2. *Doing nothing — only the warning from Step 1 fires.* Rejected because the drift remains real and future `uv sync` invocations can recreate the divergence.

## Done criteria (this revision)

- [ ] `Ensure-Single-Venv-Junction` added to `scripts/install.ps1` after `Install-Venv`
- [ ] Stage registration in install pipeline (verify exact location by reading the bootstrap function)
- [ ] `scripts/release.py` reviewed; Python helper added **only if** `release.py` mutates `venv/`
- [ ] `[tool.hermes]` block in `pyproject.toml` (unchanged from plan 001 Step 3)
- [ ] `docs/rca-dual-venv-collapse.md` created with revised content (additive design)
- [ ] **No edits** to `Install-Venv` body
- [ ] No edits to the 6 importer files, `apps/desktop/electron/main.cjs`, or built artifacts
- [ ] All previous Step 1 tests still pass (re-run `pytest tests/test_hermes_bootstrap.py --deselect ...`)

## STOP conditions

- The install pipeline's exact stage-invocation location is not findable after reading the file — STOP and ask user.
- A second reading of `scripts/release.py` reveals it does mutate `venv/` and the Python helper is non-trivial — STOP and split this revision into a smaller change.
- The new `Ensure-Single-Venv-Junction` function conflicts with a future plan's preflight check (deferred in plan 001's maintenance notes) — STOP, hand off.

## Verification you must do BEFORE reporting

1. Re-read `Install-Venv` (lines 1576–1680) and confirm the new function doesn't shadow any variable.
2. Run the existing pytest suite for `tests/test_hermes_bootstrap.py` (Step 1's tests + others) to confirm no regression.
3. **Do NOT run** `install.ps1` end-to-end during a normal session — it requires admin and modifies system state. Review is enough; the E2E smoke belongs to a separate QA pass on a fresh VM.
4. Confirm `git diff` is limited to `scripts/install.ps1` (additive), `pyproject.toml`, `docs/rca-dual-venv-collapse.md`, and **optionally** `scripts/release.py` (only if Step "Verify before modifying" said yes).