# Hermes Agent Build TODO

This is the running execution log for making this checkout complete, verified,
and usable on this machine. Update it whenever scope, status, or evidence
changes.

## Working agreement

- [x] Keep all writes inside `/home/len/hermes-agent`.
- [x] Ask for approval before changing anything outside this directory.
- [x] Build in small, testable increments and commit each coherent slice.
- [x] Preserve the upstream architecture: stable prompt caching and a narrow
      core, with capabilities added through skills, plugins, and gated tools.
- [ ] Record commands and evidence for every completed milestone below.

## Milestone 0 — Establish the baseline

- [x] Populate this directory from the official
      `NousResearch/hermes-agent` repository.
- [x] Confirm the checkout has a clean upstream commit and full source tree.
- [x] Inventory supported runtimes and repository-defined verification commands.
- [x] Run fast, dependency-light structural checks.
- [x] Document the exact local development workflow in this TODO and the
      upstream `CONTRIBUTING.md`.

## Milestone 1 — Python core

- [x] Create an isolated project-local development environment.
- [x] Install the minimum development dependencies without changing global
      Python state.
- [x] Run focused unit tests for configuration, sessions, model providers, tool
      dispatch, memory, skills, and security boundaries.
- [ ] Fix reproducible failures in coherent, separately committed slices.
- [ ] Run the full Python test suite and record results.

## Milestone 2 — CLI and local model path

- [ ] Verify CLI help, diagnostics, setup, and non-interactive entry points.
  - [x] `hermes --help` and `hermes doctor --help` load with isolated state.
  - [ ] Run non-mutating diagnostics and setup-path tests.
- [ ] Verify configuration can target the existing local Ollama service without
      changing the installed Hermes configuration.
- [ ] Add a repository-local smoke-test configuration or fixture.
- [ ] Exercise a real local-model conversation and tool call with test-scoped
      state contained in this repository.

## Milestone 3 — JavaScript applications

- [ ] Inventory Node workspaces and lockfile integrity.
- [ ] Install dependencies locally in the repository.
- [ ] Run formatting, type checks, linting, and unit tests for the web, TUI,
      desktop, and shared packages.
- [ ] Build production bundles and fix reproducible failures.

## Milestone 4 — Gateway and integrations

- [ ] Verify gateway configuration and lifecycle behavior with test-scoped
      state.
- [ ] Run platform-independent gateway tests.
- [ ] Verify cron, plugin, MCP, and messaging adapter contracts.
- [ ] Keep credential-dependent integrations disabled unless explicitly
      configured by the user.

## Milestone 5 — Packaging, security, and operations

- [ ] Verify Python packaging, console scripts, Docker assets, and install
      documentation.
- [ ] Run dependency and secret scans available without external account setup.
- [ ] Review network listeners, command approval boundaries, and safe defaults.
- [ ] Verify clean-start and upgrade paths using only repository-local fixtures.

## Milestone 6 — Completion

- [ ] Run the complete repository verification matrix.
- [ ] Reconcile documentation with observed behavior.
- [ ] Produce a concise operator guide for this machine.
- [ ] Confirm the worktree is intentional and all changes are committed.
- [ ] Record remaining items that truly require credentials, external services,
      hardware, or user decisions.

## Evidence log

- 2026-07-24: Official upstream checkout established on branch `main` at
  `7e3acd02d925b25fcf5fb5afd0076954bb6fc769`.
- 2026-07-24: All writes remain scoped to `/home/len/hermes-agent`.
- 2026-07-24: Runtime inventory: Python 3.12.3 (supported range
  `>=3.11,<3.14`), Node 22.23.1, npm 10.9.8. The repository contains 2,354
  Python test files and 465 JavaScript/TypeScript test files.
- 2026-07-24: Dependency-light baseline passed: `compileall` for tracked Python
  sources, `bash -n` for shell scripts, JSON parsing for root/web/TUI/test/
  desktop manifests, and `git fsck --no-dangling`. One existing test docstring
  emitted a non-fatal invalid-escape `SyntaxWarning`.
- 2026-07-24: Canonical workflows are `scripts/run_tests.sh` for isolated
  Python tests and `npm run check` for Node workspaces. Local Python
  development uses a venv plus editable `.[all,dev]`; JavaScript uses the root
  npm lockfile and workspaces.
- 2026-07-24: Created repository-local `.venv` with repository-local uv 0.11.32
  and installed editable `.[dev]` dependencies. Confirmed pytest 9.0.2 can
  import the checkout. No global Python packages or configuration changed.
- 2026-07-24: Focused Python core matrix passed 660 tests covering session
  persistence and alternation repair, toolsets, provider wiring/fallback,
  memory, skills, and worktree security. The Anthropic fallback test required
  its repository-pinned optional dependency (`anthropic==0.87.0`) in `.venv`;
  after installation, all 23 fallback tests passed. The session-state file
  passed 413 tests in 102.55 seconds.
- 2026-07-24: `hermes --help` and `hermes doctor --help` loaded successfully
  with `HERMES_HOME` isolated under `.tmp/hermes-home`.
