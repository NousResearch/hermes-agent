# Atomic Mail — Installation

## MCP (stdio)

```json
{
  "mcpServers": {
    "atomicmail": {
      "command": "npx",
      "args": ["-y", "@atomicmail/mcp"]
    }
  }
}
```

## AgentSkill (shell)

```bash
npx --package=@atomicmail/agent-skill atomicmail register --username "myagent"
npx --package=@atomicmail/agent-skill atomicmail jmap_request --ops-file list_inbox.json
npx --package=@atomicmail/agent-skill atomicmail help
```

## After register: hourly inbox polling

Registration only creates credentials. Arrange hourly inbox polling per your
runtime (see **cron** topic):

- **Native cron host** (OpenClaw, Hermes, atomic-agent, …): schedule an hourly
  **agent** turn with `list_inbox.json` inside the prompt.
- **No native cron** (Claude, Pi, Cursor, …): ask your operator to set up
  polling on a capable host, or remind them to fetch mail manually when needed.
  Do not work around this with wrapper scripts or OS schedulers.

Do not cron `atomicmail jmap_request` alone.
