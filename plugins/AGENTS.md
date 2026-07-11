# Plugins Subtree Instructions

This file scopes plugin guidance to `plugins/` work. Root `AGENTS.md` still contains the non-negotiable project rules. Full reference: `docs/agent-context/plugins.md`.

## Plugin policy

- Plugins must not modify core files for plugin-specific behavior. If a plugin needs missing framework capability, widen the generic hook/ABC/ctx surface.
- No new in-tree third-party-product plugins or memory providers. Ship those as standalone plugin repos installed into `$HERMES_HOME/plugins/` or via entry points.
- Existing in-tree plugins may receive bug fixes and maintenance.
- General plugins register hooks/tools/CLI commands through `hermes_cli/plugins.py`; memory providers implement the `MemoryProvider` ABC.
- Be aware of discovery timing: code that reads plugin state before importing `model_tools.py` must explicitly call plugin discovery if needed.

## Verification

- Test the real discovery/registration path; do not only unit-test isolated plugin helpers.
- Run tests with `scripts/run_tests.sh ...`, never direct `pytest`.
