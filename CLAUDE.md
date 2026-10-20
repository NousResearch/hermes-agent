# Hermes Agent
Personal AI agent shared by a Python CLI/gateway and TypeScript TUI/Desktop surfaces.

Read `AGENTS.md` and the relevant area's `AGENTS.md` before changing code;
they are authoritative and contain the architecture, security and test contracts.

- Python agent/backend; React, TypeScript, Ink TUI and Electron desktop.
- Root npm workspace dependencies include js-yaml and semver; UI dependencies
  and TypeScript configurations belong to each workspace (no root tsconfig.json).
- `agent/`, `tools/`, `gateway/`: agent execution and integrations.
- `hermes_cli/`, `tui_gateway/`: CLI, configuration, plugins and transport.
- `plugins/`, `plugin-catalog/`: bundled extensions and external catalog.
- `apps/desktop/`, `apps/shared/`, `ui-tui/`, `web/`: UI workspaces.
- `pm/`, `hermes_platform/`: managed environments and host abstraction.
- `tests/`, `tests-js/`, `scripts/`, `website/`: verification and documentation.

Commands (use an isolated HERMES_HOME and runtime directory):
- `source ./activate`: provision/activate the managed development environment.
- `scripts/run_tests.sh <test paths>`: canonical Python test runner.
- `python scripts/check`: blocking CI lint and code-health checks.
- `npm run check` / `npm run fix`: workspace checks / formatting fixes.
- `npm run build --workspace ui-tui`: build the terminal UI.
- `npm run build --workspace apps/desktop`: build the desktop workspace.

Conventions: topic-owned siblings behind facades; profile-aware paths and
explicit background profile scope; preserve per-conversation prompt caching.
Extend plugin APIs generically and keep reserved config/security guards intact.
Use Conventional Commits and never modify generated output by hand.

GitHub Actions runs change-classified Python suites, workspace checks and
release/bundle smoke workflows. Local tests do not establish CI or release success.
