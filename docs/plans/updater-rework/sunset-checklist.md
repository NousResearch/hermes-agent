# Legacy Updater Sunset Checklist

> Phase 5 task 5.5 — write now, execute later.
>
> **This is a time-gated plan.** The precondition for all items is:
> adoption prompts shipping ≥ 12 months; maintainer declares the legacy
> population negligible (support-channel signal / opt-in diagnostics).
>
> Each deletion lands as its own PR with the E2E gates green.

## Precondition (global)

- [ ] Adoption prompts (`hermes adopt` offer at launch + `/update adopt`)
      shipping for ≥ 12 months
- [ ] Maintainer declares legacy population negligible based on:
  - Support-channel signal (no "I can't adopt" reports for 3+ months)
  - Opt-in diagnostics showing <5% of active installs on legacy git-checkout
    update path
- [ ] All phase 1-4 E2E gates green in CI for 2+ consecutive weeks
- [ ] No open P1s against adopt/apply

## Deletions

Each item is its own PR. Verify E2E gates green after each.

### Frozen contract

- [ ] Delete `hermes_cli/updater_compat.py`
- [ ] Delete `tests/test_updater_compat_fence.py`
  - **Precondition:** no supported legacy updater still in the wild.
  - **Verification:** `scripts/run_tests.sh` still green (no dangling imports).

### Legacy git update flow

- [ ] Delete `_cmd_update_impl` git flow — keep only the thin dispatcher:
  slot → `hermes-updater apply`, checkout → `dev_update`, docker/nix/brew →
  messages
  - **Precondition:** `hermes update` in a checkout uses the worktree flow
    by default (phase 3 task 3.4).
  - **Verification:** `hermes update` in a checkout still works; slot update
    still works; docker still redirects to `docker pull`.

### Legacy venv/update machinery

- [ ] Delete `_UvResult` (`hermes_cli/managed_uv.py`)
- [ ] Delete `rebuild_venv` tombstone
- [ ] Delete `_update_via_zip`
- [ ] Delete `_quarantine_running_hermes_exe` + friends
- [ ] Delete `_pause/_resume_windows_gateways_for_update`
- [ ] Delete `_detect_concurrent_hermes_instances`
- [ ] Delete `_detect_venv_python_processes`
- [ ] Delete `.update-incomplete` recovery (`_recover_from_interrupted_install`)
- [ ] Delete `_install_hangup_protection` + `_UpdateOutputStream` (updater
      owns logs now)
  - **Precondition:** `hermes-updater apply` handles all these cases
    (concurrent instances, locked files, interrupted installs).
  - **Verification:** `scripts/run_tests.sh` green; E2E slot lifecycle gate
    green.

### Desktop legacy paths

- [ ] Delete `gateway/code_skew.py`
- [ ] Delete retry-once in any remaining Tauri path
- [ ] Delete `sourceDeclaresServe` + `dashboardFallbackArgs` (desktop)
  - **Precondition:** all managed installs use the launcher for backend
    spawn (phase 4 task 4.3).
  - **Verification:** desktop typecheck + electron tests green.

### Install scripts

- [ ] Shrink `scripts/install.sh` to: fetch updater + PATH setup + `--source`
- [ ] Shrink `scripts/install.ps1` similarly
  - **⚠ CRITICAL PRECONDITION:** `hermes_cli/dep_ensure.py` uses `install.sh`
    AS ITS RUNTIME BACKEND for lazy native-dep installs (ffmpeg,
    chromium/agent-browser, system packages — see its module docstring:
    "1900 lines of battle-tested OS detection"). Before shrinking, extract
    the OS-detection + package-manager logic into a standalone script the
    bundle carries (e.g. `scripts/install-native-dep.sh`, shipped in
    `app/scripts/` of every bundle) and re-point `dep_ensure.py` at it.
  - **Verification:** `hermes doctor` on a slot install with no system ffmpeg
    can still prompt-install it.

### Test infrastructure

- [ ] Remove `scripts/run_tests.sh` third venv probe (deprecated in phase 3
      task 3.6 — the `$HOME/.hermes/hermes-agent/venv` fallback with the
      deprecation warning)
  - **Precondition:** all active checkouts have per-checkout `.venv` (no one
    relies on the shared venv).
  - **Verification:** `scripts/run_tests.sh` still finds venvs correctly.

## Post-sunset state

After all deletions, the update surface is:

```
hermes-updater apply     # managed slots (the one true path)
hermes update            # thin dispatcher → updater or dev_update or docker message
hermes dev sync          # checkout provisioning
hermes adopt / eject     # switch between managed and source
```

No autostash, no venv rebuild, no concurrent-instance detection, no
install hangup protection, no code-skew warnings, no Tauri retry-once.
The updater owns all of that; the CLI is a thin dispatcher.
