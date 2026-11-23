---
name: hermes-webapp-build-test
description: Use when building the Hermes webapp (apps/desktop/scripts/build-webapp.mjs --no-install with pinned runtime), running targeted Vitest suites in the build workspace, and enforcing strict verification-before-completion.
---

# Hermes Webapp Build and Test

## Overview
This skill defines the exact procedure for building the Hermes desktop webapp, executing targeted unit tests via Vitest in the workspace, and satisfying the strict verification-before-completion gate.

Based on: [obra/superpowers/verification-before-completion](https://skills.sh/obra/superpowers/verification-before-completion) (MIT License), [antfu/skills/vitest](https://skills.sh/antfu/skills/vitest) (license not verified — inspired only), and [antfu/skills/vite](https://skills.sh/antfu/skills/vite) (license not verified — inspired only).

## Actionable Steps

### 1. Build Webapp with Pinned Runtime (`--no-install`)
Always build the webapp using the pinned tools and `--no-install` to avoid updating lockfiles or modifying global/workspace state:
```bash
$HOME/.hermes/tools/node-26.12.1-linux-x64/bin/node apps/desktop/scripts/build-webapp.mjs --no-install
```
*Note:* Ensure that the build script runs successfully and exits with code `0`.

### 2. Run Targeted Vitest Suites
Navigate to or target the build workspace (`.build/webapp-workspace/apps/desktop`) for running targeted Vitest suites covering modified components and modules:
```bash
$HOME/.hermes/tools/node-26.12.1-linux-x64/bin/npx vitest run --dir .build/webapp-workspace/apps/desktop/...
```
Alternatively, invoke Vitest via the pinned pnpm/npx binary depending on the workspace configuration, ensuring zero test failures.

### 3. Verification-Before-Completion Gate
Before claiming any success or completion:
1. **Identify**: What exact command proves the build or test suite passed?
2. **Run**: Execute the full command fresh and complete (do not rely on previous runs or assumptions).
3. **Read**: Inspect the full stdout/stderr, verify exit code `0`, and count failures (must be `0`).
4. **Confirm**: State the outcome with exact fresh evidence before asserting completion.

## What this skill does NOT do
- Never installs dependencies or runs bare `pnpm install` / `npm install`.
- Never edits lockfiles (`pnpm-lock.yaml`, `package-lock.json`, etc.).
- Never restarts the gateway or backend daemon.
