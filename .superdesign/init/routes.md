# Routes

Hermes uses React Router configured in `web/src/App.tsx`.

| Route | Page |
|---|---|
| / | Redirect to /sessions |
| /chat | ChatPage persistent embedded TUI |
| /sessions | SessionsPage |
| /files | FilesPage |
| /analytics | AnalyticsPage |
| /models | ModelsPage |
| /logs | LogsPage |
| /cron | CronPage |
| /skills | SkillsPage |
| /plugins | PluginsPage |
| /mcp | McpPage |
| /channels | ChannelsPage |
| /webhooks | WebhooksPage |
| /pairing | PairingPage |
| /profiles | ProfilesPage |
| /profiles/new | ProfileBuilderPage |
| /config | ConfigPage |
| /env | EnvPage |
| /system | SystemPage |
| /docs | DocsPage |

The full router and shell source is preserved in `layouts.md` to avoid duplicating the 1,395-line file. The route declarations are in `BUILTIN_ROUTES_CORE`, `BUILTIN_NAV_REST`, and the final `Routes` render within that file.

