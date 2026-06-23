# Changelog

All notable changes to Hermes Agent are documented here.

## Unreleased

### Security / Reliability

- **`hermes mcp add` now refuses to save when `${VAR}` can't be resolved.**
  Any placeholder in a new MCP server entry (command, args, env, headers, url,
  etc.) is resolved against the target profile's `.env`, the shared
  `~/.hermes/.env`, and the current process environment — in that order,
  matching the gateway's runtime resolution contract — before the entry is
  written to `config.yaml`.  If any placeholder is unresolved the save is
  refused with a one-line error per variable pointing the operator to the
  per-profile `.env` (the least-privilege target).

  Use `--allow-unresolved-env` to bypass the gate.  When bypassed, a warning
  is emitted to stderr and the unresolved variable names are stored as a
  `_unresolved_env_vars` advisory in the saved entry so `hermes doctor` can
  surface them later.

  The same gate fires on the dashboard's `POST /api/mcp/servers` endpoint
  (HTTP 422 with structured `issues` list on failure).

  **Root cause:** the ghost-MCP failure class from June 2026 (kanban
  `t_4a884ba1` / `t_76e39214`): an MCP entry with `env.KEY=${UNSET_VAR}` was
  saved and then crashed the gateway on every boot when the MCP child process
  received the literal string `${UNSET_VAR}` as its credential.  106 ghost-MCP
  failure events across 6 profiles in 24 hours before the operational fix.
  This save-time gate prevents the next instance of the same shape.

  **Implementation:** `hermes_cli/mcp_security.validate_mcp_server_secrets()`
  — a new peer helper alongside the existing `validate_mcp_server_entry()`
  exfiltration check.  Wired into `_save_mcp_server`, `cmd_mcp_add` (fast-fail
  before the discovery probe), and `web_server.add_mcp_server`.
