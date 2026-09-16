# Microsoft Graph permission evidence

Checked 2026-09-16 against the Microsoft Graph permissions reference and endpoint documentation. These links are the source of the operation matrix in `backend.py`; local preflight reports consent as `not_tested`.

| Capability / operation | Graph endpoint and SDK builder | App role used | Status | Official source |
|---|---|---|---|---|
| Outlook search/read | `/users/{id}/messages`, `client.users.by_user_id(id).messages.get` | `Mail.Read` | required; admin consent | https://learn.microsoft.com/graph/api/resources/mail-api-overview |
| Outlook draft/send | messages `post`; `sendMail.post` | `Mail.ReadWrite` / `Mail.Send` | required; admin consent | https://learn.microsoft.com/graph/api/user-sendmail |
| Calendar | `/users/{id}/events`, `calendar.events.get/post/patch` | `Calendars.Read` / `Calendars.ReadWrite` | required; admin consent | https://learn.microsoft.com/graph/api/resources/calendar |
| OneDrive files | drive item content/search builders | `Files.Read.All` / `Files.ReadWrite.All` | required; admin consent | https://learn.microsoft.com/graph/api/resources/driveitem |
| SharePoint files | site drive and drive item builders | `Sites.Read.All` and Files roles | required; tenant/site restrictions | https://learn.microsoft.com/graph/api/resources/sharepoint |
| Teams reads | joined teams, channels, search query builders | `Team.ReadBasic.All`, `Channel.ReadBasic.All`, `Chat.Read.All`, `ChannelMessage.Read.All` | endpoint-specific; admin consent | https://learn.microsoft.com/graph/permissions-reference |
| Teams send_messages | channel messages `post` | no application role claimed | unsupported in client credentials; delegated flow not implemented | https://learn.microsoft.com/graph/api/channel-post-messages |
| Microsoft To Do | `/users/{id}/todo/lists` and tasks | `Tasks.Read.All` / `Tasks.ReadWrite.All` | required; admin consent | https://learn.microsoft.com/graph/api/resources/todo-overview |
| Planner plans/buckets/tasks | `client.planner.plans`, buckets, tasks builders | `Group.Read.All` for plan enumeration; Tasks roles for task operations | explicit Planner capability; tenant/group restrictions; remote support not tested | https://learn.microsoft.com/graph/api/resources/planner-overview |

`client_credentials` is the only implemented authentication mode. Delegated permissions are not silently substituted for application roles. If Microsoft changes endpoint permission support, the status must be updated with new source evidence and tests rather than inferred from model construction.
