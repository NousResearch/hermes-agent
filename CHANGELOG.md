# Changelog

## Unreleased

### Added

- `hermes doctor` now includes a **MCP Secret Propagation** section that audits every `~/.hermes/profiles/*/config.yaml` `mcp_servers` block for unresolved `${VAR}` references, using the same three-layer resolution order the gateway uses at boot (profile `.env` → shared `.env` → process env). Unresolved references emit per-profile warnings with actionable remediation steps. The same check runs at the end of `hermes setup` so fresh installs with broken MCP configs surface immediately. (kanban t_5239030e)
