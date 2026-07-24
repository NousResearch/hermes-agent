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
- [ ] Inventory supported runtimes and repository-defined verification commands.
- [ ] Run fast, dependency-light structural checks.
- [ ] Document the exact local development workflow.

## Milestone 1 — Python core

- [ ] Create an isolated project-local development environment.
- [ ] Install the minimum development dependencies without changing global
      Python state.
- [ ] Run focused unit tests for configuration, sessions, model providers, tool
      dispatch, memory, skills, and security boundaries.
- [ ] Fix reproducible failures in coherent, separately committed slices.
- [ ] Run the full Python test suite and record results.

## Milestone 2 — CLI and local model path

- [ ] Verify CLI help, diagnostics, setup, and non-interactive entry points.
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
