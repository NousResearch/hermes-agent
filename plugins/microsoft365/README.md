# Microsoft 365 Plugin

## Business value

The **Microsoft 365 Plugin** gives Hermes one controlled path into the Microsoft work graph: it can find business context, manage documents, coordinate Teams conversations, schedule work, and turn follow-ups into tasks. This supports workflows such as researching a customer thread, attaching the relevant file, scheduling the next step, and recording the resulting task without copying data between unrelated integrations.

## Why one plugin

Outlook, SharePoint, OneDrive, Calendar, Teams, and Planner/To Do share Microsoft Graph authentication, request builders, permission review, and approval policy. Keeping them in one plugin avoids duplicated credentials and inconsistent safety gates while retaining service- and operation-level flags. The Teams Bot Framework adapter and `teams_pipeline` remain separate because they are different transports and runtime contracts.

## Tool-surface control for the model

Enable operations individually under `plugins.entries.microsoft365.settings.capabilities`:

```yaml
capabilities:
  outlook: {search: true, read: true, create_draft: true, send: false}
  sharepoint: {search: true, read: true, download_files: true, upload_files: true}
  onedrive: {search: true, read: true, download_files: true, upload_files: true}
  calendar: {search: true, create_events: true, update_events: true}
  teams: {list_teams: true, list_channels: true, search_messages: true, send_messages: false}
  planner: {list_task_lists: true, search: true, read: true, create_tasks: true, update_tasks: false}
```

The model sees one tool per enabled capability, and each tool's `action` enum contains the configured operation family. Disabled and unknown operations are rejected before Graph client creation. `microsoft365_preflight` is always local, reports the derived application permissions, and identifies operations unavailable in app-only mode. Service-level `true` remains supported and enables every operation for compatibility.

Supported operations are Outlook search/read/create draft/send; SharePoint and OneDrive search/read/download/upload; Calendar search/create/update events; Teams list teams/list channels/search/send messages; and Planner/To Do task-list search/read/create/update tasks. Reads use `get`; writes use generated SDK models and `post`, `patch`, or `put` request-builder methods. No raw Graph HTTP is used.

## Authentication and permissions

This implementation uses `azure.identity.ClientSecretCredential` with the Microsoft Graph `https://graph.microsoft.com/.default` scope. Its mode is explicitly represented as `client_credentials` / `application`: it obtains an app-only token from application permissions granted to the Entra app. It does **not** implement delegated authorization-code, device-code, or user-consent flow. Delegated scope names are not sufficient for this plugin.

`OPERATION_PERMISSIONS` is therefore application-permission-only and operation-specific:

- Outlook: `Mail.Read`, `Mail.ReadWrite`, `Mail.Send`.
- SharePoint: `Sites.Read.All`, `Files.Read.All`, `Files.ReadWrite.All`.
- OneDrive: `Files.Read.All`, `Files.ReadWrite.All`.
- Calendar: `Calendars.Read`, `Calendars.ReadWrite`.
- Teams reads: `Team.ReadBasic.All`, `Channel.ReadBasic.All`, `Chat.Read.All`, `ChannelMessage.Read.All` as applicable to the operation.
- Planner/To Do: `Tasks.Read.All`, `Tasks.ReadWrite.All`.

Teams `send_messages` remains in the requested operation surface and is approval-gated, but is reported as unsupported in this application-only mode because the channel/chat send endpoint has no corresponding application permission. The plugin does not relabel delegated `ChatMessage.Send` as an app permission. An administrator must grant tenant admin consent for the exact application roles required by enabled operations; Graph endpoint and tenant policy can impose additional restrictions. `user_id` must identify the target user for app-only user-resource calls; `/me` is a delegated convention and is not a substitute for an app-only user ID.

## Security invariants

- Every write calls Hermes' host-owned `tools.approval.request_tool_approval` using rule key `microsoft365.<capability>.<operation>`.
- Only the host result shape `{"approved": true}` can approve; plugin-local strings cannot bypass the host gate.
- Approval errors, denial, timeout, unavailable interactive context, or malformed results fail closed.
- The Graph client is created only after the operation is enabled and approval succeeds.
- Credentials are never included in tool results; returned objects are bounded and secret-key redacted.
- The permission report is derived from enabled operations rather than requesting a blanket permission set.

Approval is a policy layer, not a sandbox. It does not limit what an already-authorized Entra application can do if the application has broader roles, and it cannot contain a compromised dependency or Graph service. Use least-privilege app registration, tenant controls, secret rotation, and endpoint/network controls as defense in depth.

## PR-facing rationale

This change keeps the full requested Microsoft 365 workflow surface while making its trust boundary explicit: one plugin owns Graph auth and operation policy, the model receives only enabled capability tools, writes remain host-approved, and preflight tells administrators what app-only consent actually means. It improves safety claims without pretending that delegated permissions or approval prompts provide a sandbox.

## Test evidence

The targeted plugin tests pass with `py -3.11 -m pytest tests/plugins/test_microsoft365_plugin.py -q` (16 passed). They cover operation-specific least privilege, app-only permission names, delegated-permission non-claims, unsupported operation reporting, generated SDK models/request builders, disabled-operation rejection, and fail-closed approval before client creation. No credentials or network access are needed.

The repository environment has Trio installed but does not have the `pytest-trio` plugin installed. Consequently these plugin tests exercise async handlers through `asyncio.run`; no Trio test run is claimed, and this limitation is not a hidden test failure.
