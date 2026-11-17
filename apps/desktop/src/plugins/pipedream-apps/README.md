# Pipedream Apps (Desktop plugin)

Adds a **Connectors** tab backed by the Pipedream apps catalog to
**Capabilities**, replacing the built-in Connectors tab. Old `?tab=connectors`
links open it instead. It is a runtime plugin — copy the folder, no app rebuild
for the plugin itself:

```powershell
Copy-Item -Recurse desktop-plugins\pipedream-apps $HOME\.hermes\desktop-plugins\pipedream-apps
```

The tab needs the Desktop core build that includes the `capabilities.tabs`
contribution area, and the `pipedream-apps` Python backend
(`plugins/pipedream-apps/dashboard/`) mounted by the gateway.

## Configuration

The project credentials and MCP proxy base are baked into the backend as module
constants in `plugins/pipedream-apps/dashboard/plugin_api.py`, so no gateway
environment or `~/.hermes/.env` entry is required. The configured `_MCP_BASE`
enables MCP server configuration after a successful connect.

## Behaviour

- Browses Pipedream Connect apps with search, category filters and cursor pagination.
- Connect opens the Pipedream Connect Link in the system browser and polls until the
  account appears.
- Lists connected accounts, shows health/errors, reconnects or disconnects.
- Uses the Clerk user id as `external_user_id` (persisted UUID fallback when Clerk is
  unavailable).
- After a connect, calls the plugin backend (`POST /mcp/enable`), which configures
  `pipedream_<slug>` through the Dashboard MCP API when an MCP proxy base is configured.
  Disconnecting the last account for an app disables the entry instead of deleting it.
  There is no live reload; tools become available in a new chat/session.
- Replaces the built-in Capabilities Connectors tab: the row always shows Skills,
  Plugins, Connectors, Tools, and `?tab=connectors` links open this catalog.

Enable/disable it in **Capabilities → Plugins**; it ships on by default.
