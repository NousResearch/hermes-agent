# Rabbit CLI Reference

Live sources when anything looks stale: `rabbit --help`, `rabbit <command> --help`,
https://github.com/seven0070/Rabbit-/tree/main/website/docs/reference/cli-commands

### Global Flags

```
rabbit [flags] [command]        (no subcommand = interactive chat)

  --version, -V             Show version
  -z, --oneshot PROMPT      One-shot: print ONLY the final response (for scripts/pipes)
  -m MODEL  --provider P    Model/provider override for this invocation
  -t, --toolsets LIST       Comma-separated toolsets for this invocation
  --resume, -r SESSION      Resume session by ID or title
  --continue, -c [NAME]     Resume by name, or most recent session
  --worktree, -w            Isolated git worktree mode (parallel agents)
  --skills, -s SKILL        Preload skills (comma-separate or repeat)
  --profile, -p NAME        Use a named profile
  --yolo                    Skip dangerous command approval
  --tui / --cli             Force the Ink TUI / classic REPL
  --ignore-rules            Skip AGENTS.md/SOUL.md/memory/skill injection
  --safe-mode               Disable ALL customizations (troubleshooting)
  --pass-session-id         Include session ID in system prompt
```

### Chat

```
rabbit chat [flags]
  -q, --query TEXT          Single query, non-interactive
  --image PATH              Attach a local image to a single query
  -Q, --quiet               Suppress banner, spinner, tool previews
  --checkpoints             Enable filesystem checkpoints (/rollback)
  --max-turns N             Cap tool-calling iterations
  --source TAG              Session source tag (default: cli)
```
(plus the global flags above)

### Configuration

```
rabbit setup [section]      Wizard (model|tts|terminal|gateway|tools|agent)
rabbit model                Interactive model/provider picker
rabbit fallback [add|remove|list]  Fallback provider chain
rabbit config [show|edit|get|set|unset|path|env-path|check|migrate]
rabbit login / logout       OAuth sign-in / clear stored auth
rabbit doctor [--fix]       Check dependencies and config
rabbit status [--full]      Component summary (--full: every section)
```

### Tools & Skills

```
rabbit tools [list|enable NAME|disable NAME]   Per-platform toolsets (curses UI with no args)

rabbit skills list|browse|search QUERY|inspect ID
rabbit skills install ID    Hub identifier OR a direct https://…/SKILL.md URL
rabbit skills config        Enable/disable skills per platform
rabbit skills check|update|uninstall|publish PATH
rabbit skills tap add REPO  Add a GitHub repo as a skill source
rabbit bundles              Skill bundles (one /<name> alias loads several skills)
```

### MCP Servers

```
rabbit mcp add NAME (--url or --command) | remove | list | test NAME
rabbit mcp catalog | install NAME     Curated catalog install
rabbit mcp configure NAME             Toggle tool selection
rabbit mcp serve                      Run Rabbit as an MCP server
```
Details (transport, tool discovery, catalog): `references/native-mcp.md`.

### Gateway (Messaging Platforms)

```
rabbit gateway run|install|start|stop|restart|status|setup
```

20+ platforms: Telegram, Discord, Slack, WhatsApp (Baileys + Business Cloud API), iMessage (Photon — `rabbit photon setup`), Signal, Email, SMS, Matrix, Mattermost, Teams, LINE, SimpleX, ntfy, Google Chat, Home Assistant, DingTalk, Feishu, WeCom, Weixin, API Server, Webhooks. Open WebUI connects via the API Server adapter. Most adapters ship under `plugins/platforms/`.
Docs: https://github.com/seven0070/Rabbit-/tree/main/website/docs/user-guide/messaging/

### Sessions

```
rabbit sessions list|browse|rename ID TITLE|delete ID|export OUT|prune|stats
```

### Cron / Webhooks

```
rabbit cron list|create SCHED|edit ID|pause|resume|run ID|remove|status
    Schedules: '30m', 'every 2h', '0 9 * * *', ISO timestamp
rabbit webhook subscribe NAME|list|remove NAME|test NAME
```
Webhook payloads/routes: `references/webhooks.md`.

### Profiles

```
rabbit profile list|create NAME (--clone|--clone-all|--clone-from)|use|show|delete
rabbit profile rename A B | alias NAME | export NAME | import FILE
rabbit profile migrate-identity A B   Retry a completed rename's session/routing identity migration
```

### Credentials & Pools

```
rabbit auth                 Interactive credential manager
rabbit auth add [PROVIDER]  Add OAuth or API-key credential (nous, openai-codex, qwen-oauth, …)
rabbit auth list|remove P IDX|reset PROVIDER|status
```
Multiple credentials per provider form a pool that rotates automatically and skips exhausted keys.

### Other

```
rabbit desktop / gui        Native desktop app
rabbit dashboard            Web admin panel + embedded chat (--stop / --status)
rabbit proxy                OpenAI-compatible local proxy backed by an OAuth provider
rabbit portal               Quick setup / sign in via Nous Portal
rabbit kanban <verb>        Multi-agent work-queue board
rabbit project              Named multi-folder workspaces
rabbit skin list|use|set    Switch/tweak skins (see references/themes.md)
rabbit pets <verb>          Pet mascots (see references/petdex.md)
rabbit memory setup|status|off|reset   Memory provider
rabbit secrets bitwarden|onepassword   External secret stores
rabbit moa                  Mixture-of-Agents slots
rabbit hooks / security / backup / import / checkpoints / console
rabbit logs [-f] [errors]   View agent/error logs
rabbit send                 One-off message through a gateway platform
rabbit pairing / plugins / insights / journey / computer-use
rabbit acp                  ACP server (IDE integration)
rabbit completion bash|zsh|fish
rabbit update / uninstall / claw migrate
```

Plugin- and provider-supplied subcommands (e.g. `rabbit photon setup`) only appear once their plugin is installed/active.

### Where to Find Things

| Looking for... | Location |
|---|---|
| Config options | `rabbit config edit` · [Configuration docs](https://github.com/seven0070/Rabbit-/tree/main/website/docs/user-guide/configuration) |
| Tools / toolsets | `rabbit tools list` · [Tools reference](https://github.com/seven0070/Rabbit-/tree/main/website/docs/reference/tools-reference) |
| Skills catalog | `rabbit skills browse` · [Skills catalog](https://github.com/seven0070/Rabbit-/tree/main/website/docs/reference/skills-catalog) |
| Provider setup | `rabbit model` · [Providers guide](https://github.com/seven0070/Rabbit-/tree/main/website/docs/integrations/providers) |
| Env variables | `rabbit config env-path` · [Env vars reference](https://github.com/seven0070/Rabbit-/tree/main/website/docs/reference/environment-variables) |
| Gateway logs | `~/.rabbit/logs/gateway.log` (or `rabbit logs`) |
| Sessions | `rabbit sessions browse` (reads state.db) |
