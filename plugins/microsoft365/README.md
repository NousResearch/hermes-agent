# Microsoft 365 Plugin

The **Microsoft 365 Plugin** gives Hermes one connection to the work people already do
across Outlook, SharePoint, OneDrive, Calendar, Teams, Planner and To Do. That means an
assistant can find context, move files, coordinate conversations and turn follow-ups
into tasks without making users stitch together separate integrations. It remains one
plugin and one `microsoft365` toolset, while administrators choose exactly which
operations are exposed.

It uses Microsoft's official `msgraph-sdk` and `azure-identity`; it does not duplicate
Graph HTTP operations. Install the optional dependencies with
`pip install 'hermes-agent[microsoft365]'`.

## Operation-level configuration

Put this under `plugins.entries.microsoft365.settings`. Each enabled operation is a
boolean. A service-level `true` is retained for backward compatibility and means “all
operations for this service”; new configurations should use the explicit form:

```yaml
capabilities:
  outlook: {search: true, read: true}
  sharepoint: {search: true, read: true, download_files: true}
  onedrive: {search: true, read: true, download_files: false}
  calendar: {search: true}
  teams: {list_teams: true, list_channels: true}
  planner: {list_task_lists: true}
```

`planner` is the bundled task capability; its current bounded operation lists Microsoft To Do task lists. A service tool is registered
only when at least one operation is enabled; calls for disabled or unknown operations
are rejected before a Graph client is created. `microsoft365_preflight` is local and
side-effect-free and reports the least-privilege permissions derived from the selected
flags. The manifest contains the explicit operation and permission contract.

This bundled scope is read-only. It does not advertise draft/send, upload, event creation,
message sending, or task mutation because those SDK calls and approval wiring are not
implemented here. This plugin does not implement a separate approval gate.
Secrets are read through the host configuration flow and redacted from preflight and
SDK-shaped results.

## Desktop scope

The current Desktop plugin seam exposes discovery/status but has no generic schema-driven
plugin configuration screen or secret-field editor. This plugin ships the real
manifest/backend contract without a bespoke UI; a future generic renderer can map this
manifest's settings to the existing seam.
