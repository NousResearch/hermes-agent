# Nowledge Mem Memory Provider

Cross-tool knowledge graph memory for Hermes Agent. Decisions, procedures,
lessons, and conversation context can stay available across Hermes, Cursor,
Claude Code, Codex, Gemini, and other Nowledge Mem integrations.

## Requirements

- Nowledge Mem desktop app or reachable server
- `nmem` CLI on PATH

If the desktop app is installed on the same machine, `nmem` is already
bundled. Otherwise:

```bash
pip install nmem-cli
```

## Setup

```bash
hermes memory setup
```

Select `nowledge-mem` from the provider picker.

Or configure manually:

```bash
hermes config set memory.provider nowledge-mem
```

## What It Does

- injects Working Memory into the system prompt
- prefetches relevant memories before each turn
- mirrors Hermes user-profile writes into Nowledge Mem
- captures cleaned Hermes session transcripts as Mem threads when the session actually ends
- exposes native tools for search, save, update, delete, and thread lookup

Durable saves still happen through the native `nmem_` tools. In addition,
the provider captures cleaned Hermes session transcripts at real session
boundaries such as clean exit, `/new`, `/reset`, and gateway session expiry.
The first flush uses `nmem t import`; later flushes in the same live Hermes
session append only the delta with `nmem t append`.

## Tools

| Tool | Purpose |
|------|---------|
| `nmem_search` | Search durable memories |
| `nmem_save` | Save a decision, learning, or durable fact |
| `nmem_update` | Refine an existing memory |
| `nmem_delete` | Delete one or more memories |
| `nmem_thread_search` | Search past conversations |
| `nmem_thread_messages` | Fetch messages from a thread |

## Configuration

Store provider config in:

```text
$HERMES_HOME/nowledge-mem.json
```

Example:

```json
{
  "timeout": 30,
  "space": "Research Agent",
  "space_by_identity": {
    "research": "Research Agent",
    "ops": "Operations Agent"
  },
  "space_template": "agent-{identity}"
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `timeout` | `30` | Request timeout in seconds |
| `space` | `""` | Optional default space name for this Hermes provider |
| `space_by_identity` | `""` | Optional JSON object mapping Hermes identities to spaces |
| `space_template` | `""` | Optional template like `research-{identity}` when `space` is empty |

Use `space` when one Hermes profile always belongs to one lane. Use
`space_by_identity` when a few Hermes identities map to named lanes. Use
`space_template` when Hermes already has a stable identity and you want one
lane per identity.

If you are launching Hermes through a CLI wrapper with no provider config of
its own, you can still set one session-wide fallback lane with:

```bash
NMEM_SPACE="Research Agent" hermes
```

If you do not have a real ambient lane, stay on `Default`.

Server URL and API key are managed by `nmem`, not the provider. For remote
Mem, configure the machine running Hermes with:

```bash
nmem config client set url https://your-server:14242
nmem config client set api-key your-key
```

That writes the shared local client config `nmem` reads. It is separate from
server-side Access Anywhere or bind/allowlist settings on the Mem host.

## Verify

Ask Hermes:

> Search my memories for recent decisions.

The provider should call `nmem_search` and return results from Nowledge Mem.

Then ask:

> Save a memory that the Hermes Nowledge Mem plugin test passed.

The provider should call `nmem_save`. If Hermes instead falls back to its
built-in `memory` tool, you are running a build older than `0.5.5`.
