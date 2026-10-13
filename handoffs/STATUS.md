# TH1-900 — Plugin development-group isolation

Branch: `fix/TH1-900-plugin-runtime-groups`
Base: stable Hermes `71d34bad1045610252eda02ee79549436896e784`.

Goal: install jev-effort-router 0.3.0 across M5/M4/WSL without changing stable core dependency pins or weakening plugin developer requirements.
Root cause: PM copies plugin-only dependency groups into its generated runtime uv workspace. uv resolves every group even if sync excludes it, so plugin Ruff>=0.17 conflicts with core Ruff==0.15.10.
Fix: project local-only dependency groups, legacy uv dev dependencies and group-only settings/conflicts out of generated runtime members. Published runtime deps/extras, original plugin metadata and core pins are preserved. Genuine runtime conflicts remain errors.

Done:
- Reproduced exact resolution failure with one new offline real-uv regression on unpatched source.
- Implemented runtime metadata projection; regression plus existing workspace tests all pass (7/7).
- Relevant PM suite run once: 897 passed, 17 failed, 39 skipped, 122 files. Failures are in other PM test files (network/reset/virtualenv/cross-target/activation paths); full suite is NOT green and no claim of baseline equivalence is made.
- Initial runner with a production-root interpreter was blocked by home I/O guard before useful test execution. Provisioned a separate PM-owned development/test environment under this worktree and reran safely.
- Ruff checks for changed Python files and git diff --check passed. No root pyproject or uv.lock change.

Next: preserve live files/config/plugins, apply minimal PM source patch to M5, prove actual catalog update and functional router/main-model smoke before peer changes. Then peer backups/rollout/readback. Do not alter model routing. Record local patch custody because later Hermes updates may park it until upstream merges.
PR pending. Finish/stop with pushed handoff and Linear comment.
