# Implementation Log: harness-gateway-startup-repair

- Date: 2026-05-24
- Branch: main
- Scope: Hypura Harness startup, gateway .env loading, plugin platform auto-enable, runtime status cleanup.

## Signals

- `hermes harness start` reported a missing daemon under `.venv/Lib/site-packages/vendor/...`.
- `gateway.run` warned that no allowlists were configured even though `GATEWAY_ALLOW_ALL_USERS=true` was present.
- Discord was auto-enabled by plugin dependency availability and then failed with `No bot token configured`.
- `gateway_state.json` retained stale platform entries from an older run.
- Hypura Harness started but its legacy OSC listener and VRChat OSC bridge both attempted to bind `127.0.0.1:9001`.

## Root Causes

- The user `.env` had a UTF-8 BOM, so `python-dotenv` loaded the first key as `\ufeffGATEWAY_ALLOW_ALL_USERS`.
- Plugin platform auto-enable treated dependency availability as platform configuration.
- Runtime status startup preserved the previous `platforms` map.
- The installed `hermes.exe` path resolved the default harness daemon relative to `.venv/Lib/site-packages`, where the vendored OpenClaw tree is absent.
- Hypura's `__main__` startup launched a legacy OSC listener before FastAPI startup launched the newer VRChat OSC bridge on the same receive port.

## Changes

- `hermes_cli/env_loader.py`
  - Load dotenv files with `utf-8-sig` so BOM-prefixed first keys are recognized.
- `gateway/config.py`
  - Enable plugin platforms only after env seeding and `is_connected` / `validate_config` confirms they are actually configured.
- `gateway/status.py` and `gateway/run.py`
  - Allow replacing the runtime platform map and clear it at gateway startup.
- `hermes_cli/harness.py` and `hermes_cli/hypura_native.py`
  - Resolve the harness daemon from the real source tree when the installed package default is missing.
- `vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py`
  - Let the VRChat OSC bridge own the receive port when enabled.
  - Preserve legacy telemetry updates through the bridge callbacks.
- `C:/Users/downl/.hermes/config.yaml`
  - Added `harness.script_path` pointing to the actual vendored daemon for the currently installed `hermes.exe` runtime.

## Verification

- Fixed live `.env` BOM and verified `load_hermes_dotenv()` reads `GATEWAY_ALLOW_ALL_USERS=true`.
- Restarted gateway; current logs show only LINE connecting and no new allowlist or Discord token errors.
- Confirmed `gateway_state.json` now contains only the current LINE platform.
- Started and restarted Hypura Harness through `.venv/Scripts/hermes.exe`; `/health` returns `{"ok": true, "daemon_version": "0.1.0"}`.
- Confirmed Harness restart logs show `VRChat OSC bridge listening on 127.0.0.1:9001` without the duplicate bind warning.

## Commands

- `.venv/Scripts/python.exe -m pytest tests/hermes_cli/test_harness.py tests/hermes_cli/test_env_loader.py tests/gateway/test_config.py::TestLoadGatewayConfig::test_plugin_platform_dependency_check_alone_does_not_enable tests/gateway/test_config.py::TestLoadGatewayConfig::test_plugin_platform_env_seed_is_validated_before_enable tests/gateway/test_status.py::TestGatewayRuntimeStatus::test_write_runtime_status_can_replace_platforms -q -o addopts=`
  - Result: 23 passed.
- `uv run python -m pytest tests/test_vrchat_harness_daemon.py::test_vrchat_bridge_callbacks_update_legacy_telemetry -q -o addopts=`
  - Result: 1 passed, FastAPI deprecation warnings only.
- `.venv/Scripts/python.exe -m py_compile hermes_cli/env_loader.py gateway/config.py gateway/status.py gateway/run.py hermes_cli/harness.py hermes_cli/hypura_native.py vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py`
  - Result: passed.
- `git diff --check`
  - Result: passed, with Windows LF-to-CRLF warnings only.

## Residual Notes

- `uv pip install -e . --no-deps` could not update `.venv/Scripts/hermes.exe` because an existing Hermes process holds the executable open.
- The explicit `harness.script_path` config keeps the current installed runtime working until the environment can be refreshed.
- The repo-level test addopts still require `-o addopts=` on Windows for these focused runs because `pytest-timeout` is configured with the POSIX `signal` method.
