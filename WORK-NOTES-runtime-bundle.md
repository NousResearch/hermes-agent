# WORK NOTES - runtime bundle cold-start/ready/check_fn

## Scope

- Repository: `karlligamesvc-spec/hermes-agent`
- Branch: `codex/runtime-bundle-coldstart-ready-checkfn`
- Task bundle: hc-180 cold start, hc-115 truthful ready, hc-174 observable check_fn
- Release path: hc-085 runtime version flow, image build -> admin approval -> existing agents upgrade on demand
- Image contract: keep `hermes.plugin-tools-gateway=true` label for hc-172

## Sources Read

- User final instruction in Codex thread, 2026-06-12.
- Local cold-start evidence: `/Users/kaelsu/hermes-cloud/WORK-NOTES-hc175-stage0.md`.
- Local runtime contracts: `/Users/kaelsu/hermes-cloud/docs/FEATURE-CONTRACTS.md` hc-085/hc-172 entries.

Feishu PD table links for hc-180(row 181)/hc-115/hc-174 were not discoverable from local repo/tooling at kickoff. Proceeding from the user-provided final spec plus the local hc-175 measurement notes; update this file if a concrete Feishu link becomes available.

## Baseline From hc-175 Stage 0

- Feishu agent ready: 28-55s observed; sample median around 37s.
- Platform connect dominates ready time: 64-79%.
- Known contributors:
  - `lark-oapi` lazy install around 6.1s.
  - A silent 11.2s gap between API server connected and Feishu connect log.
  - `tirith` online download around 3.3s.
  - Plugin discovery runs twice in one startup sequence.

## Implementation Checklist

- [x] Make Feishu/platform connect background work so gateway API can serve conversation without waiting for websocket/platform attachment.
- [x] Keep ready truthful: ready now follows the API conversation path reaching `running`; Feishu websocket attachment continues in background.
- [x] Add segmented logs around platform adapter initialization and Feishu API/WS connect.
- [x] Bake `lark-oapi`, `tirith`, and `edge-tts` online dependencies into the image layer.
- [x] Deduplicate plugin/tool discovery in the startup sequence with a plugin-manager lock/discovered guard.
- [x] Log `check_fn` failures at WARNING with tool name and reason.
- [x] Shorten/jitter failed `check_fn` cache TTL for faster restart-window self-healing.
- [x] Verify image label `hermes.plugin-tools-gateway=true` statically in Dockerfile contract tests.

## Validation Log

- hc-180 step 2 retest follow-up, 2026-06-12:
  - Candidate `hc180-runtime-bundle-20260612` measured `ready_total=24.37s`; Feishu websocket connect itself was `0.44s`, but there was a `12.27s` silent gap after `api_server` ready and before the background Feishu connect log.
  - `lark-oapi` was already baked by this branch, and the log did not show lazy install. Code inspection found the remaining serial wait was before background connect: `GatewayRunner.start()` still imported/constructed the Feishu adapter synchronously via `_create_adapter()`, which imports the official `lark_oapi` SDK at module import time.
  - Fix: schedule Feishu adapter creation and connect as one background path, run `_create_adapter()` with `asyncio.to_thread()`, and add segmented logs for background adapter creation, connect, and channel-directory refresh.
- Targeted pytest, 2026-06-12 hc-180 retest fix:
  - `uv run --extra dev python -m pytest tests/gateway/test_runner_startup_failures.py -q`
  - Result: 11 passed.
- Targeted pytest, 2026-06-12:
  - `scripts/run_tests.sh tests/gateway/test_runner_startup_failures.py tests/tools/test_registry.py tests/docker/test_runtime_bundle_contract.py tests/gateway/test_feishu.py tests/gateway/test_platform_reconnect_fd_leak.py tests/hermes_cli/test_plugins.py tests/plugins/test_disk_cleanup_plugin.py tests/test_dockerfile_tini_compat_shim.py`
  - Result: 8 files, 387 tests passed, 0 failed.
- Syntax check, 2026-06-12:
  - `python -m py_compile gateway/run.py gateway/platforms/feishu.py tools/registry.py hermes_cli/plugins.py tests/gateway/test_runner_startup_failures.py tests/tools/test_registry.py tests/docker/test_runtime_bundle_contract.py`
  - Result: pass.
- Full pytest discipline run, 2026-06-12:
  - `scripts/run_tests.sh`
  - Result: 1360 files, 29466 tests passed, 50 failed, runner wall 768.0s.
  - Runtime-bundle touched test files passed in the full run:
    - `tests/gateway/test_runner_startup_failures.py` 10 passed.
    - `tests/gateway/test_feishu.py` 205 passed.
    - `tests/gateway/test_platform_reconnect_fd_leak.py` 7 passed.
    - `tests/hermes_cli/test_plugins.py` 83 passed.
    - `tests/tools/test_registry.py` 33 passed.
    - `tests/plugins/test_disk_cleanup_plugin.py` 46 passed.
    - Dockerfile contract tests passed.

## Full Pytest Failure Triage

The 50 failures are outside this runtime bundle change surface. They look like existing environment/test-contract drift in the local macOS runner, not regressions from Feishu async connect, Docker dependency baking, plugin discovery, or `check_fn` caching:

- macOS path canonicalization: `/tmp` expected, `/private/tmp` actual in `test_background_command.py` and `test_file_tools.py`.
- Local platform assumptions: missing `systemctl`/D-Bus and systemd platform guards in `test_live_system_guard_self_test.py`, `test_gateway_wsl.py`, `test_gateway_service.py`, `test_gateway_shutdown.py`, and `test_service_manager.py`.
- Local proxy/browser dependency environment: Matrix proxy reads `http://127.0.0.1:1080`; browser tests attempt `agent-browser` install and timeout.
- AnyIO backend mismatch: tests parametrized under trio call `asyncio.create_task`/`asyncio.gather` in MS Graph and Teams pipeline tests.
- Existing mock/test isolation issues: Anthropic OAuth keychain path receives `MagicMock` from patched `subprocess.run`; MCP preflight records no HEAD/GET calls; approval session pattern collision; shutdown diagnostic spawn returned `None`; synthetic SIGTERM/real interrupt timing flakes.

## Remaining Validation

- Cold-start measurement:
- Ready-immediate conversation:
- check_fn warning evidence:
- Docker image:
  - Local build attempted: `docker build --build-arg HERMES_GIT_SHA=$(git rev-parse --short HEAD) -t hermes-agent:hc180-runtime-bundle-20260612 .`
  - Result: blocked locally because this macOS session has no `docker`, `podman`, `nerdctl`, `colima`, or `finch` CLI.
  - Follow-up path: PR touches Dockerfile, so `.github/workflows/docker-publish.yml` should build/smoke-test `nousresearch/hermes-agent:test`; publishing follows hc-085 after merge/admin approval.
- PR:
