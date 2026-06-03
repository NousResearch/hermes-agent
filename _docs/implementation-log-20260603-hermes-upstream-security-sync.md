# Hermes upstream security sync implementation log

Date: 2026-06-03
Branch: codex/hermes-upstream-security-sync-20260603
Upstream: NousResearch/hermes-agent@918aef267
Fork base before sync: zapabob/hermes-agent@1a3a458ad

## Scope

- Installed and verified DannyMac180/skills `codex-dynamic-workflows`.
- Enabled Codegraph against this checkout and used it for merge and fallout review.
- Synced upstream `main` through `scripts/sync_all.py` with the fork policy map.
- Preserved fork policy while accepting upstream npm workspace lockfile consolidation.
- Repaired Windows and desktop test fallout from the upstream sync.

## Merge policy update

- Added `**/package-lock.json` to `scripts/merge_tools/hermes-merge-conflict-strategies.json` with `upstream` resolution.
- Rationale: upstream now keeps a single root npm lockfile. Nested lockfiles under `apps/desktop`, `ui-tui`, and `web` are intentionally removed upstream.

## Local repairs after merge

- Restored the official root `package-lock.json` after a local package-lock-only install removed upstream transitive entries needed by workspace tests.
- Scoped `apps/desktop` UI tests to `src` so Vitest no longer attempts to run Electron TAP tests under jsdom.
- Made desktop streaming tests deterministic by replacing wall-clock streaming waits with explicit rerenders.
- Honored saved pane width overrides independently from user-resizable drag capability.
- Made managed `uv` tests explicit about POSIX behavior when running on Windows.
- Enabled the example dashboard plugin in the isolated `HERMES_HOME` fixture before backend API auth route checks.
- Read the Dockerfile with UTF-8 in the PID 1 reaping tests to avoid Windows CP932 decode failures.
- Added targeted timeout margins to the slow first desktop settings and skills tests under full Windows jsdom suite load.

## Verification evidence

- `py -3 scripts/sync_all.py --dry-run --skip-fetch`: reduced blockers after policy update.
- `py -3 scripts/sync_all.py --merge --target codex/hermes-upstream-security-sync-20260603 --allow-preflight-blockers --skip-fetch --commit-message "merge: sync upstream/main 918aef267 with fork policy"`: completed.
- Merge report: `_docs/merge-reports/sync-all-ok-20260603T023041Z.json`.
- `uv lock --check`: passed.
- `npm ci --ignore-scripts`: passed.
- `npm audit --omit=dev`: 0 vulnerabilities.
- `uv run --with pytest python -m pytest tests/scripts/test_sync_all.py tests/agent/test_auxiliary_client.py tests/hermes_cli/test_default_interface_resolution.py tests/hermes_cli/test_managed_uv.py tests/hermes_cli/test_web_server.py tests/test_hermes_state.py tests/tools/test_dockerfile_pid1_reaping.py -o addopts=''`: 697 passed, 12 skipped.
- `npm run type-check --workspace apps/desktop`: passed.
- `npm run type-check --workspace ui-tui`: passed.
- `npm run build --workspace web`: passed.
- `npm test --workspace ui-tui`: passed.
- `npm run test:desktop:platforms --workspace apps/desktop`: passed.
- `npm run test:ui --workspace apps/desktop`: 40 files passed, 229 tests passed.
- `uv run ruff check tests/hermes_cli/test_managed_uv.py tests/hermes_cli/test_web_server.py tests/tools/test_dockerfile_pid1_reaping.py`: passed.
- `git diff --check`: no whitespace errors.

## Residual risk

- `npm ci` reports an engine warning for `@icons-pack/react-simple-icons@13.13.0`, which declares Node `>=24` while the local and several CI workflows use Node 20 or 22.
- Full repository test execution was not completed in this pass because the upstream sync is broad; targeted high-risk lanes were prioritized.
