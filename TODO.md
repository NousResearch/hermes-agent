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
- [x] Record commands and evidence for every completed milestone below.

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
- [x] Fix reproducible failures in coherent, separately committed slices.
- [x] Run the full Python test suite and record results.

## Milestone 2 — CLI and local model path

- [x] Verify CLI help, diagnostics, setup, and non-interactive entry points.
  - [x] `hermes --help` and `hermes doctor --help` load with isolated state.
  - [x] Run non-mutating diagnostics with repository-isolated state.
- [x] Verify configuration can target the existing local Ollama service without
      changing the installed Hermes configuration.
- [x] Add a repository-local smoke-test configuration or fixture.
- [ ] Exercise a real local-model conversation and tool call with test-scoped
      state contained in this repository.
  - [x] Transport and validation paths exercised against local Ollama.
  - [ ] Install/select a >=64K local model that reliably follows Hermes tool
        schemas; current installed models each miss one compatibility contract.

## Milestone 3 — JavaScript applications

- [x] Inventory Node workspaces and lockfile integrity.
- [x] Install dependencies locally in the repository.
- [x] Run formatting, type checks, linting, and unit tests for the web, TUI,
      desktop, and shared packages.
- [x] Build production bundles and fix reproducible failures.

## Milestone 4 — Gateway and integrations

- [x] Verify gateway configuration and lifecycle behavior with test-scoped
      state.
- [x] Run platform-independent gateway tests.
- [x] Verify cron, plugin, MCP, and messaging adapter contracts.
- [x] Keep credential-dependent integrations disabled unless explicitly
      configured by the user.

## Milestone 5 — Packaging, security, and operations

- [x] Verify Python packaging, console scripts, Docker assets, and install
      documentation.
- [x] Run dependency and secret scans available without external account setup.
- [x] Review network listeners, command approval boundaries, and safe defaults.
- [x] Verify clean-start and upgrade paths using only repository-local fixtures.

## Milestone 6 — Completion

- [x] Run the complete repository verification matrix.
- [x] Reconcile documentation with observed behavior.
- [x] Produce a concise operator guide for this machine.
- [x] Confirm the worktree is intentional and all changes are committed.
- [x] Record remaining items that truly require credentials, external services,
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
- 2026-07-24: Installed declared `.[all,dev]` plus pinned lazy integrations in
  `.venv`. Fixed Copilot ACP host-HOME propagation and made SQLite corruption/
  WAL tests portable across protected SQLite builds. The original 14-file
  failure matrix then passed 751 tests with zero failures.
- 2026-07-24: Definitive Python suite passed: 2,258 files, 46,326 tests, zero
  failures in 780.4 seconds with eight workers. Ruff passed. Whole-tree `ty`
  reports 12,512 existing diagnostics; upstream treats ty as a diff-comparison
  gate rather than requiring a legacy-clean tree.
- 2026-07-24: Root `npm run check` passed for all workspaces, including type
  checks, unit tests, builds invoked by workspace checks, and linting. Lint
  reported 24 warnings and zero errors.
- 2026-07-24: Explicit production builds passed for the web UI, TUI, desktop,
  and bootstrap-installer workspaces. Web emitted only a non-fatal large-chunk
  advisory; all four build commands exited 0 and left the tracked tree clean.
- 2026-07-24: `hermes doctor` exited 0 against repository-isolated state. Core
  tools are available; credential-gated integrations correctly remain disabled.
- 2026-07-24: Local Ollama validation: qwen3/hermes-qwen report 40,960-token
  contexts (below Hermes 64K minimum); gemma3:12b reports 131K but Ollama says
  it does not support tools; llama3.1:8b reports 131K and accepts requests/tools
  but returned a serialized tool schema instead of the requested response.
- 2026-07-24: Gateway, cron, plugin, MCP, and messaging contracts passed in the
  complete Python suite. `hermes-acp --check` passed. Credential-gated adapters
  remained disabled under isolated diagnostics.
- 2026-07-24: Packaging/operations: Docker Compose configuration and console
  entry points passed. Wheel/sdist refusal is an intentional upstream policy;
  supported distribution is installer, Docker, or Nix with editable installs
  for development.
- 2026-07-24: Security: upgraded vulnerable Node and Python dependencies. npm
  audit reports zero vulnerabilities. Hermes OSV reports zero high/critical
  findings; setuptools 81 retains one moderate macOS-only finding because the
  fixed 83.x line conflicts with the documented Torch `<82` constraint.
- 2026-07-24: Final post-upgrade verification completed. The full Python run
  exercised every test file; its only two assertion failures were metadata-pin
  drift corrected during the run, while the context-compressor file exceeded
  the per-file runner timeout under concurrent model load. An isolated rerun
  of those three files then passed all 230 tests in 386.05 seconds. Root Node
  checks, all four production builds, Ruff, CLI/ACP diagnostics, Docker Compose
  validation, npm audit, and the high-severity OSV gate also passed.
- 2026-07-24: Tracked-file secret-pattern scan found no candidates outside
  explicit examples, documentation, redaction fixtures, skills, and tests.
- 2026-07-24: Clean-start, setup, non-interactive CLI, installer divergence,
  lockfile churn, no-initial-commit, and update paths passed under repository-
  local fixtures in the complete suite.
- 2026-07-24: Remaining external choice: install or select a local model with
  a true >=64K context and reliable structured tool calling, or configure a
  compatible remote provider. No model download or production credential was
  added without user approval.
