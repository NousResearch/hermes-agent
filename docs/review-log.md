# Review Log

## 2026-06-28 AFK Audit Remediation

Framework run:

- `/Volumes/500G/Claude Code Projects/Codex Code Review/security-reviews/2026-06-27-active-30d`

Initial relevant finding handled in this pass:

- `Tests / CI`: `warn` - Python tests and GitHub workflows existed, but root `package.json` did not expose reusable `test` or `lint` commands, so the central audit could not identify a complete local gate entrypoint.

Changes made:

- Added root `npm test` as a direct wrapper for the canonical `scripts/run_tests.sh` runner.
- Added root `npm run lint` as a blocking `uv run ruff check .` gate.
- Added root `npm run check` to run lint before the test suite.
- Added a regression test to keep the root package quality-gate scripts present.

Verification run:

- `scripts/run_tests.sh tests/test_root_package_quality_scripts.py -- -q` - pass.
- `npm test -- tests/test_root_package_quality_scripts.py -- -q` - pass.
- `npm run lint` - pass; ruff reported one pre-existing invalid `# noqa` warning and `All checks passed`.
- Central re-audit after local fix - `Tests / CI`, `Deploy / Ops`, `Architecture / Maintainability`, `Docs / Handoff`, and `Product / UX Readiness` passed.

Known remaining central queue items:

- `Code Quality`: needs deep review due dangerous-pattern volume.
- `Security`: needs deep review due recent-history gitleaks findings.
- `Secrets / Env Hygiene`: needs deep review due recent-history gitleaks findings.
- `Dependency / Supply Chain`: needs deep review due dependency audit high/critical count.

## 2026-06-28 AFK Deep Review - Security / Dependencies / Code Quality

Framework run:

- `/Volumes/500G/Claude Code Projects/Codex Code Review/security-reviews/2026-06-27-active-30d`

Reviewed HEAD:

- `50dfa1eaf8c58d6f32dfaf5f248e24eff4946810`

Initial central findings reviewed in this pass:

- `Security`: `needs_deep_review` - recent-history gitleaks reported 10 redacted findings.
- `Secrets / Env Hygiene`: `needs_deep_review` - gitleaks reported 10 redacted secret candidates.
- `Dependency / Supply Chain`: `needs_deep_review` - dependency audit reported high/critical vulnerabilities.
- `Code Quality`: `needs_deep_review` - 115 dangerous-pattern hits needed reachability review.

Changes made:

- Fixed the WhatsApp bridge audio-conversion command execution path. `/send-media` no longer builds a shell string for `ffmpeg`; it now uses `execFileSync('ffmpeg', args)` so media file paths are passed as argv values.
- Added `scripts/whatsapp-bridge/ffmpeg-command.test.mjs` as a regression guard that fails if the bridge returns to shell-string ffmpeg execution.
- Added `npm --prefix scripts/whatsapp-bridge test` support via a package-level `test` script so the bridge regression tests are discoverable.

Verification run:

- `node --test scripts/whatsapp-bridge/ffmpeg-command.test.mjs` - failed before the fix, proving the regression test caught the unsafe execution shape.
- `node --test scripts/whatsapp-bridge/ffmpeg-command.test.mjs scripts/whatsapp-bridge/allowlist.test.mjs` - pass after the fix.
- `npm --prefix scripts/whatsapp-bridge test` - pass, 6 tests.
- `node --check scripts/whatsapp-bridge/bridge.js` - pass.
- `npm run lint` - pass.

Additional remediation loop:

- Hardened `agent/anthropic_adapter.py` keychain credential parsing so non-text `stdout` is ignored and JSON decode errors include `TypeError`. This prevents mocked or unexpected subprocess output from reaching `json.loads`.
- Added test isolation so Anthropic adapter tests do not read the live macOS Keychain by default, and added a keychain regression for non-string stdout.
- Made gateway/media path assertions compare canonical real paths on macOS, where `/tmp` resolves to `/private/tmp`.
- Made Linux/systemd gateway tests hermetic on macOS by mocking systemd availability, container state, and user-systemd preflight checks where the test is simulating Linux.
- Allowed read-only live-system guard tests to pass on hosts without `systemctl` while still verifying the guard does not block read-only calls.
- Restored final directory mode after `chown` in `hermes_cli/service_manager.py`, and made Darwin setgid assertions account for group-membership behavior on `/private/tmp`.
- Fixed the kanban SIGTERM worker test to treat a polled/reaped child as exited on platforms without `/proc` zombie state.
- Removed an empty untracked `plugins/model-providers/ai-gateway` directory that was being discovered as an invalid provider.
- Made `/browser connect` launch-hint tests hermetic by stubbing the manual launch command when asserting the "no executable found" branch.
- Updated file-tool tests to assert the resolved canonical path passed to the underlying file operation layer.

Additional verification run:

- `scripts/run_tests.sh tests/agent/test_anthropic_adapter.py tests/agent/test_anthropic_keychain.py -- -q` - pass, 168 tests.
- `scripts/run_tests.sh tests/gateway/test_background_command.py -- -q` - pass, 22 tests.
- `scripts/run_tests.sh tests/gateway/test_gateway_shutdown.py -- -q` - pass, 14 tests.
- `scripts/run_tests.sh tests/hermes_cli/test_gateway_wsl.py -- -q` - pass, 19 tests.
- `scripts/run_tests.sh tests/hermes_cli/test_gateway_service.py -- -q` - pass, 136 tests.
- `scripts/run_tests.sh tests/test_live_system_guard_self_test.py -- -q` - pass, 34 tests.
- `scripts/run_tests.sh tests/hermes_cli/test_service_manager.py -- -q` - pass, 52 tests.
- `scripts/run_tests.sh tests/hermes_cli/test_signal_handler_kanban_worker.py -- -q` - pass, 3 tests.
- `scripts/run_tests.sh tests/providers/test_plugin_discovery.py -- -q` - pass, 4 tests.
- `scripts/run_tests.sh tests/test_tui_gateway_server.py -- -q` - pass, 205 tests.
- `scripts/run_tests.sh tests/tools/test_file_tools.py -- -q` - pass, 36 tests.
- `npm run lint` - pass.
- `npm run check` - pass, 1344 files, 28970 tests passed, 0 failed, 100% complete in 429.1s.

Deep-review dispositions:

- Gitleaks findings: all 10 recent-history findings are in `tests/` paths. Static scanner runtime-ish hits were checked without printing raw values: `agent/google_oauth.py` documents a public desktop OAuth client constant; `gateway/platforms/yuanbao.py` uses `TOKEN_PATH` as a URL path constant; `.env.example` contains a container image example, not a secret value.
- Dependency audit: remains `approval_required`. High/critical groups include direct/runtime surfaces such as `scripts/whatsapp-bridge` `baileys`, plus `protobufjs`, `ws`, `undici`, `vite`, `shell-quote`, `form-data`, and `website` transitive groups. Upgrades can affect messaging, desktop, web, and build/runtime behavior, so no dependency upgrade was done in AFK mode.
- Code Quality: one validated command-injection sink in `scripts/whatsapp-bridge/bridge.js` was fixed and tested. Remaining representative hits were dispositioned as follows: `plugins/security-guidance/patterns.py` is mostly scanner self-reference; `apps/desktop/scripts/write-build-stamp.cjs` runs fixed git commands; `gateway/platforms/api_server.py` wildcard CORS is configuration-controlled and needs deployment threat-model review; `tools/browser_supervisor.py` wildcard CORS is an internal browser bridge path and needs scoped owner review; desktop measurement scripts use browser-dev instrumentation.

Known remaining central queue items:

- `Dependency / Supply Chain`: `approval_required` for high/critical package upgrades.
- `Code Quality`: `human_required` for scoped CORS / browser-supervisor / instrumentation review beyond the fixed WhatsApp bridge sink.
- `Security` and `Secrets / Env Hygiene`: `false_positive_or_test_fixture` for the redacted gitleaks evidence reviewed in this pass, with future real leaks still requiring normal triage.
