# Microsoft 365 Plugin

The single **Microsoft 365 Plugin** provides bounded, operation-gated Graph access across Outlook, SharePoint, OneDrive, Calendar, Teams, Microsoft To Do, and Microsoft Planner. The shared boundary owns authentication, permissions, request limits, approval intent, and redaction; Teams Bot Framework and `teams_pipeline` remain separate adapters.

## Capability truth

| Service | Implemented operations | Application auth status |
|---|---|---|
| Outlook | search, read, create_draft, send | supported |
| SharePoint / OneDrive | search, read, download_files, upload_files | supported |
| Calendar | search, create_events, update_events | supported |
| Teams | list_teams, list_channels, search_messages | supported; `send_messages` is retained but blocked as delegated-only |
| Microsoft To Do | list_task_lists, search, read, create_tasks, update_tasks | supported |
| Microsoft Planner | list_plans, list_buckets, list_tasks, read, create_tasks, update_tasks | separate Planner builders; see permission evidence |

Enable individual operations under `plugins.entries.microsoft365.settings.capabilities`. One tool is registered per enabled capability, and its action enum includes only enabled operations supported by the configured auth mode.

## Binary file contract

Uploads require `content_base64`, a relative `path`, and optional `content_type` (default `application/octet-stream`). Downloads return bounded `content_base64`, `content_type`, `size`, and `path`. The limit is 10 MiB per transfer. Empty files are valid. Paths use UTF-8 URL encoding, must be relative, and reject empty, `.` or `..` segments (including backslash traversal). The plugin never writes arbitrary local destinations and never silently discards bytes. Files over the limit fail before Graph client creation. Uploads use the official drive-item content `put(bytes)` builder; downloads use the official content `get()` builder. Large-file upload sessions are not claimed.

## Authentication, permissions, and approval

The implemented mode is `azure.identity.ClientSecretCredential` with Graph `https://graph.microsoft.com/.default` and application roles. It does not implement delegated authorization-code, device-code, or To Do/Planner delegated flow. Endpoint-specific permissions and official sources are in [`references/graph-permissions.md`](references/graph-permissions.md). Preflight reports `authentication: not_tested`, `permissions: not_tested`, and `admin_consent: required`; local readiness is not proof of tenant consent or connectivity.

The `pre_tool_call` hook emits a host approval directive for enabled writes. Validation of action-specific arguments happens before approval/client work in the handler, while the host remains the canonical policy chokepoint. Credentials are never returned in results. Configure the client secret through Hermes' secret scope/environment (`MICROSOFT365_CLIENT_SECRET`); it is not printed, logged, or included in status responses.

## Verification

Run:

```bash
python -m pytest tests/plugins/test_microsoft365_plugin.py tests/plugins/test_microsoft365_tasks_6_12.py -q
python -m pytest tests/tools/test_microsoft_graph_client.py tests/tools/test_microsoft_graph_auth.py -q
python -m hermes_cli.plugin_validate plugins/microsoft365/plugin.yaml
python -m compileall -q plugins/microsoft365
```

Known limitation: the current implementation is app-only; Teams message sending remains visible in administrative configuration metadata but is mechanically blocked and absent from model-facing schemas. Planner permission support is documented as endpoint evidence, not a claim that remote tenant consent has been verified.
