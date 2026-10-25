# Hermes Agent — agent entry point

Load [CODING_STANDARDS.md](CODING_STANDARDS.md) (task index inside) for code, tests, dependencies or PR review; docs-only work needs neither. Read every matching area guide below.

## Invariants

- Keep the cached system-prompt prefix stable: only compression changes past context, and prompt-changing slash commands defer unless invoked with `--now`.
- Keep core tools small: extend existing code or use CLI + skill, gated tool, plugin or MCP first.
- Verify bugs and original intent on current `main`; before planning, implementing, reviewing or triaging anything, read the [contribution rubric](website/docs/developer-guide/contributing.md). Automated triage closes only `implemented_on_main`, `cannot_reproduce` or `incoherent`; subjective scope belongs to maintainers.
- Report §3.1 vulnerabilities privately, never in a public PR: [SECURITY.md](SECURITY.md).

## Routes — every matching guide applies together

| Work | Read |
|---|---|
| Agent | [agent](agent/AGENTS.md) |
| CLI, profiles | [hermes_cli](hermes_cli/AGENTS.md) |
| Gateway, adapters | [gateway](gateway/AGENTS.md); new adapter: [guide](gateway/platforms/ADDING_A_PLATFORM.md) |
| Tools | [tools](tools/AGENTS.md) |
| Plugins, catalog | [plugins](plugins/AGENTS.md), [catalog](plugin-catalog/README.md) |
| TUI, web | [tui_gateway](tui_gateway/AGENTS.md), [web](web/AGENTS.md) |
| Desktop | [desktop](apps/desktop/AGENTS.md); UI: [src](apps/desktop/src/AGENTS.md) |
| Skills, curator | [skills](skills/AGENTS.md) |
| Cron, kanban | [cron](cron/AGENTS.md) |
| Tests | [tests](tests/AGENTS.md) |
| Dependencies, `uv.lock` | [pm](pm/AGENTS.md) |
| Host facts | [platform](hermes_platform/AGENTS.md) |

Root modules and other unlisted paths: the task index in [standards](CODING_STANDARDS.md#task-index). [Long form](website/docs/developer-guide/).
