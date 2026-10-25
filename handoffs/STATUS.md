# TH1-900 — Plugin development-group isolation

Branch: `fix/TH1-900-plugin-runtime-groups`
Base: stable Hermes `71d34bad1045610252eda02ee79549436896e784`.
State: live fix and plugin upgrade verified on M5/M4/WSL; source PR awaiting review.

## Goal / fix
Install jev-effort-router 0.3.0 without altering stable core pins or weakening plugin developer requirements.
PM previously copied plugin-only dependency groups into a generated runtime uv workspace. uv resolves every group, even if sync excludes it, so plugin Ruff>=0.17 conflicted with core Ruff==0.15.10.
The runtime snapshot now omits non-published dependency groups, legacy uv dev dependencies, and group-only uv settings/conflicts. Published runtime deps/extras, original plugin metadata and core pins remain intact. Genuine runtime conflicts still refuse admission.

## Verification
- One new offline real-uv regression reproduced the resolution failure before the fix.
- Regression and existing workspace tests: 7/7 passed.
- Relevant PM suite once: 897 passed, 17 failed, 39 skipped, 122 files. Full suite is NOT green. Other failures involve network-reset/download, virtualenv marker, cross-target signing and activation paths; no baseline-equivalence claim is made.
- Initial test attempt used a production-root interpreter and the home I/O guard blocked useful execution. A separate PM-owned environment was provisioned in the worktree; meaningful red/green proof used that isolated interpreter.
- Ruff checks, commit hook and git diff --check passed. Root pyproject.toml and uv.lock unchanged.
- Backed up source/config/plugin before each live mutation. Applied only pm/workspace.py overlay to stable core.
- M5 actual catalog update succeeded before peer changes. Then M4/WSL actual catalog updates succeeded.
- Every host now has official catalog pin `6fd9f2ec1c53cb99b5e6c7043c25bbd37938d4a5`, manifest version 0.3.0. Final plugin checks report update_available=false.
- Plugin doctor passed import, registration and runtime discovery on every host.
- Actual registered status handler invoked with `{recent: 1}` on every host: passed, enabled=true, six grid entries, API key present (no secret values recorded publicly).
- Main Codex smoke returned `TH1_900_OK` on every host. Complete config comparison against backups passed; main/delegation/auxiliary routing unchanged.
- Gateway plugin reload acknowledged on every host; added agent tools remain next-session scoped. No blanket service restart.
- Patched pm/workspace.py SHA256 identical fleet-wide: `07a5cc4c18e44ea411c4bf1e3d2d9bc0c23545674cc1dcd203a552d5325e7de8`.

## Patch custody / recovery
The live cores remain stable at the original SHA with an intentional local pm/workspace.py overlay. Each host has private pre-change source/config/plugin backups under its Hermes patches directory, alongside the exact patch. Original plugin developer Ruff>=0.17 and core Ruff==0.15.10 were read back unchanged.
Later Hermes updates can park local source patches. Preserve this CURRENT user-requested fix until equivalent upstream code lands; compare before reapplying, never restore historical patches blindly. No claim that a future unpatched stable release already contains it.
Private raw rollout receipts are kept locally; do not publish config, credentials, plugin .env, or raw logs.

## Remaining / next
Source PR review and merger after rebasing onto then-current main. Broad PM suite failures remain outside this targeted dependency fix. Existing manifest/ambient-toolset/M4 previous-update warnings are not represented as fixed. No Ollama inference benchmark was performed; this task proves installation and registered handler/core smoke paths.
Janitor retirement follows source PR resolution. All stopped work must retain a pushed handoff and Linear comment.
