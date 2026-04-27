# Hermes v0.11 Worktree Upgrade Interim Report

created_at: 2026-04-28

## Result

Controlled v0.11 upgrade test environment is ready in an isolated worktree. Production Hermes has not been switched.

## Production safety

- Production command still reports: `Hermes Agent v0.9.0 (2026.4.13)`.
- Production source remains: `/Users/beiming/.hermes/hermes-agent`.
- No `hermes update` was used.
- No production entrypoint was changed.
- Upstream remote push URL is disabled.

## Test worktree

- Path: `/Users/beiming/HermesUpgradeLab/hermes-agent-v011-test`
- Branch: `hermes-v011-release-test`
- Base: official `v2026.4.23`, commit `bf196a3fc0fd1f79353369e8732051db275c6276`
- Local migration commit: `88766a07 chore: stage v0.11 local migration in isolated worktree`
- Uncommitted leftover: `tests/hermes_cli/test_uninstall.py` only, intentionally not committed because it targets old uninstall API.

## What was migrated

- 北冥法典 / docs/agents governance docs.
- docs/exec-plans historical task contracts/ledgers/reports.
- safe-refactor/human-gate/review-orchestrator modules and passing tests.
- Browser visible-window support in v0.11 browser tool:
  - `browser.headless: false`
  - `--headed`
  - `AGENT_BROWSER_HEADED=true`
  - `DISPLAY=:0` fallback on macOS.
- Google/Gemini image generation as v0.11 plugin provider:
  - `plugins/image_gen/google/__init__.py`
  - provider name: `google`
  - model: `gemini-2.5-flash-image`
  - auth env: `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## What was not directly migrated

- `model_switch.py` old local patch: official v0.11 already supports grouped custom providers and `models` dict/list. Synthetic test passed.
- `api_server.py` old image enrichment patch: official v0.11 has native multimodal content support and API server tests passed.
- `uv.lock`: not transplanted from v0.9.
- `test_uninstall.py` / old uninstall patch: left uncommitted; needs v0.11-specific redesign.

## Verification

- `.venv/bin/hermes --version` in worktree: `Hermes Agent v0.11.0 (2026.4.23)`.
- `hermes doctor` runs under isolated venv. Expected warnings remain:
  - config v17→v22 available;
  - wrapper symlink points to production Hermes, not test Hermes;
  - optional deps/API keys missing;
  - agent-browser not installed in isolated environment.
- Focused harness tests: `33 passed in 0.08s`.
- API server normalize/multimodal/toolset tests: `52 passed, 6 warnings in 0.71s`.
- Google image plugin smoke test: passed.
- `py_compile` selected core files: passed.

## Current blockers before formal cutover

1. Do not run `hermes doctor --fix` against shared `~/.hermes/config.yaml` yet; config migration is a production config change.
2. Production wrapper still points to old Hermes, intentionally.
3. Need a real browser launch test with visible desktop window before relying on browser automation in v0.11.
4. Need real image generation test if Google API quota/key should be exercised.
5. Need decision on old uninstall/profile service cleanup patch: redesign for v0.11 or drop.
6. Need final cutover plan: symlink/wrapper switch, gateway restart, rollback path.

## Suggested next phase

Before production cutover, run a dedicated cutover Task Contract with explicit approval covering:

- backup checkpoint;
- config migration strategy;
- wrapper/entrypoint switch;
- gateway restart;
- browser visible-window live test;
- rollback command back to old production v0.9 path.
