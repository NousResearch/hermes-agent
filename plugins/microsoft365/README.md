# Microsoft 365 Plugin

The **Microsoft 365 Plugin** connects Hermes to the broader Microsoft ecosystem through Microsoft's official `msgraph-sdk` request builders. One bundled toolset can find business context, manage files, coordinate Teams conversations, schedule work, and turn follow-ups into tasks—without stitching together separate integrations. The existing Teams Bot Framework adapter and `teams_pipeline` remain separate.

## Useful scope

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

Supported operations are Outlook search/read/create draft/send; SharePoint and OneDrive search/read/download/upload; Calendar search/create/update events; Teams list teams/list channels/search/send messages; and Planner/To Do task-list search/read/create/update tasks. Reads use `get`; writes use generated SDK models and `post`, `patch`, or `put` request-builder methods. No raw Graph HTTP is used.

Every write requests explicit host approval through `tools.approval.request_tool_approval`. If approval is unavailable, denied, or times out, the handler returns `required_confirmation` and creates no client and performs no side effect. The approval rule key is `microsoft365.<capability>.<operation>`. Disabled and unknown operations are rejected before Graph client creation. Only capability tools with at least one enabled operation are registered; preflight is always local and reports least-privilege permissions.

Permission mapping is operation-specific: Mail.Read/Mail.ReadWrite/Mail.Send, Sites.Read.All/Files.Read.All/Files.ReadWrite.All, Files.Read/Files.ReadWrite, Calendars.Read/Calendars.ReadWrite, Team.ReadBasic.All, Channel.ReadBasic.All, Chat.Read, ChatMessage.Send, Tasks.Read and Tasks.ReadWrite. Administrators should grant only the permissions needed by enabled flags.

Install optional dependencies with `pip install 'hermes-agent[microsoft365]'`. No credentials or network access are needed by plugin tests.
