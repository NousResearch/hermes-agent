# Microsoft 365 Plugin

The bundled **Microsoft 365 Plugin** connects Hermes to the broader Microsoft 365
ecosystem: Outlook mail, SharePoint, OneDrive, Calendar, Teams and Planner/To Do.
It is one plugin and one `microsoft365` toolset; service sections are enabled with
`plugins.entries.microsoft365.settings.capabilities` flags.

Install the optional SDK dependencies with `pip install 'hermes-agent[microsoft365]'`.
The plugin uses `msgraph-sdk` and `azure-identity`; it does not implement Graph HTTP.

Configuration belongs in the plugin settings seam:

```yaml
plugins:
  entries:
    microsoft365:
      settings:
        tenant_id: "..."
        client_id: "..."
        client_secret: "..." # store through the secret-aware config flow
        user_id: "me"
        capabilities:
          outlook: true
          sharepoint: true
          calendar: true
          teams: false
          planner: false
```

`microsoft365_preflight` is local and side-effect-free: it validates configuration,
reports selected permissions, checks SDK availability, and redacts secret-shaped
values. Capability tools are registered only for enabled flags. Upload is an explicit
write operation and remains subject to Hermes approval policy.

## Desktop scope

The current Desktop plugin seam exposes plugin discovery/status but has no generic
schema-driven plugin configuration screen or secret-field editor. This plugin therefore
ships the real manifest/backend contract without a bespoke UI. The narrow follow-up is
to add a generic Desktop renderer for `PluginManifest.config_schema` plus a secret-aware
save endpoint; then map this manifest's `settings` fields to that existing generic seam.
No fake Microsoft 365-specific UI is included.
