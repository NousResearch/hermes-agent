## What does this PR do?

Fixes a stealth re-enable of the managed local runtime: clicking **Turn off** in Settings → Providers → Local Models stops the server and flips `local_runtime.enabled: false`, but the next model-activate (or any quickstart flow) silently spawns llama-server again and flips the flag back to `true`. Two independent paths in the dashboard code bypass the `enabled` gate:

1. `hermes_cli/web_routers/local_models.py::_start_local_server` calls `bootstrap.ensure_local_runtime(config, force=True)`, which skips the `enabled=False` check inside `ensure_local_runtime`.
2. The model-activate route calls `_set_runtime_enabled(True)` at the end of its job, undoing the user's toggle.

The fix removes both bypasses. `_start_local_server` now consults the `local_runtime` section up front and raises a clear error pointing at the UI when the runtime is off; it then calls `ensure_local_runtime` without `force` so the gate inside it takes over. The activate route no longer flips the flag — the only paths that set `local_runtime.enabled` are the explicit user actions: `POST /api/local-models/server` `start` and `stop`, and the `quickstart` flow (which IS a user-initiated "I want local models" intent).

## Why does it matter?

The managed llama-server holds VRAM and burns CPU/GPU even when idle — that's how a "turned off" runtime still costs users ~10W of combined power. On a 5-day-uptime laptop, the activate-on-remount loop is enough to keep the Mac in a hot, fan-spinning state for days without the user understanding why. Any user who has turned the runtime off expects it to stay off until they turn it back on; right now it doesn't.

## Reproduction (pre-fix)

1. Set `local_runtime.enabled: true` in `~/.hermes/config.yaml` and stage any model (or run quickstart).
2. Open Local Models in the dashboard → click **Turn off**. Confirm `local_runtime.enabled` is now `false` in your config and `pgrep -f llama-server` shows nothing.
3. In the same Local Models pane, click any downloaded model to **activate** it (or just re-open the pane — auto-activate is enough).
4. Observe: `local_runtime.enabled` is back to `true`, `llama-server` is running again, the user's preference was silently overridden.

## Root cause

`force=True` in `_start_local_server` was a force-spawn for the "Start" button's user-intent path. The same helper was reused by the activate and quickstart routes, where it was the wrong semantics — those routes were not the user pressing "Start", they were a downstream consequence of the user's "use this model" choice. With the user's `enabled=False` preference silently overridden, the toggle became a no-op.

## Changes Made

- **`hermes_cli/web_routers/local_models.py`**
  - `_start_local_server`: drops `force=True`, checks `local_runtime.enabled` up front, raises a `RuntimeError` with a UI-pointing message when off. Lets the gate inside `ensure_local_runtime` take over on the enabled path.
  - Removes the stealth `_set_runtime_enabled(True)` from the model-activate job. The flag is now only flipped by the explicit `/api/local-models/server` start action and by quickstart (both user-initiated).
- **`tests/hermes_cli/test_local_server_lifecycle.py`**
  - Updates `test_start_enables_and_boots` to assert `force=False` (new contract).
  - Adds `test_start_local_server_honors_disabled_toggle`: regression for the bypass — ensures `ensure_local_runtime` is not consulted when the user has turned the runtime off.
  - Adds `test_activate_does_not_reenable_when_disabled`: regression for the activate-side flag flip — verifies activate does not touch `local_runtime.enabled`.

## How to Test

1. **Manual repro of the fix** — with the patch applied, repeat the reproduction steps above. After step 3, `local_runtime.enabled` should still be `false` and the activate should surface a clear error: "The local runtime is turned off — enable it in Settings → Providers → Local Models first." (The dashboard UI surfaces this as a job failure with the same message; the explicit Start button on the same page still works.)
2. **Regression tests** — `pytest tests/hermes_cli/test_local_server_lifecycle.py -v`. All seven relevant tests pass; the pre-existing `test_status_reports_loaded_models_from_live_router` failure is unrelated (it hits `home_io_guard` on the real Hermes home, not patched here).
3. **Explicit Start still works** — `POST /api/local-models/server` with `{"action": "start"}` flips `enabled: true` and boots the server. Verified by `test_start_enables_and_boots`.
4. **Explicit Stop still works** — `POST /api/local-models/server` with `{"action": "stop"}` shuts down and flips `enabled: false`. Verified by `test_stop_disables_and_tears_down`.

## Backwards compatibility

The `/api/local-models/server` `start` and `stop` actions are unchanged. Quickstart behavior is unchanged (it still flips `enabled: true` because the user pressed "Set this up" — that's the right intent). The behavioral change is limited to: **a model activation no longer silently starts the local server when the user has it turned off.** That is the intended fix; if any downstream caller relied on the previous silent-start-on-activate behavior, they will now see a clear error and the user can opt in via the Settings pane.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow Conventional Commits (`fix(local-runtime): ...`)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate (related but distinct: #123810, #129342, #104070, #113333)
- [x] My PR contains **only** changes related to this fix
- [x] `pytest tests/hermes_cli/test_local_server_lifecycle.py -v` passes (7/7 relevant tests)
- [x] `python scripts/check` — *to be run in CI; not executed locally to keep this PR focused*
- [x] I've added tests for my changes (2 new regression tests, 1 updated)
- [x] I've tested on my platform: macOS 15.x (M-series)

### Documentation & Housekeeping

- [x] I've updated relevant documentation — or N/A (no doc change needed; the user-facing error message IS the docs)
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
