---
name: synapse-layer
version: 0.3.0
description: Zero-Knowledge persistent memory layer for Hermes Agent. Provides encrypted cross-session memory, Trust Quotient (TQ) scoring, and automatic recall across ALL channels (Telegram, WhatsApp, CLI, Discord).
homepage: https://synapselayer.org
credits: Rafa Martins
metadata: {"synapse":{"category":"memory","requires":{"env":["SYNAPSE_TOKEN"]},"mcp":{"type":"http","url":"https://forge.synapselayer.org/mcp","auth":"Bearer ${SYNAPSE_TOKEN}"}}}
---

# Synapse Layer — Persistent Memory for Hermes

**Credits**: Rafa Martins  
**Website**: [synapselayer.org](https://synapselayer.org) | **Forge**: [forge.synapselayer.org/forge](https://synapselayer.org/forge) | **GitHub**: [github.com/SynapseLayer/synapse-layer](https://github.com/SynapseLayer/synapse-layer)

Zero-Knowledge encrypted memory that persists across ALL Hermes channels. Context survives between sessions, Telegram/WhatsApp/CLI all share the same memory.

---

## Setup

### 1. Add MCP Server to config.yaml

Add to `~/.hermes/config.yaml` at the root level (same indentation as `display:`, `mcp_servers:`):

```yaml
mcp_servers:
  synapse-layer:
    url: "https://forge.synapselayer.org/mcp"
    headers:
      Authorization: "Bearer ${SYNAPSE_TOKEN}"
    timeout: 120
    connect_timeout: 60
```

### 2. Add Token to .env

Add your Synapse Layer token to `~/.hermes/.env`:

```
SYNAPSE_TOKEN=sk_connect_your_token_here
```

Get your token at: https://synapselayer.org/forge

### 3. Disable Tool Progress for Telegram (IMPORTANT)

The `tool_progress` feature shows tool names in Telegram when the model uses MCP tools. To prevent `mcp_synapse_layer_search` from appearing as a message in Telegram, add per-platform display config:

```yaml
display:
  tool_progress_command: false
  tool_progress_overrides: {}
  tool_preview_length: 0
  tool_progress: all
  platforms:
    telegram:
      tool_progress: off
```

### 4. Restart Hermes

```bash
hermes restart
```

---

## Available Tools

| Tool | Purpose |
|------|---------|
| `mcp_synapse_layer_recall` | Retrieve memories before responding (use at EVERY session start) |
| `mcp_synapse_layer_save_to_synapse` | Persist facts, preferences, decisions immediately |
| `mcp_synapse_layer_search` | Full-text cross-agent search across all memories |
| `mcp_synapse_layer_process_text` | Auto-detect milestones/decisions in free-form text |
| `mcp_synapse_layer_health_check` | Verify connection and system status |

---

## Critical Usage Patterns

### AT THE START OF EVERY NEW CONVERSATION

Always call `recall` before generating any response. This is the #1 rule for memory to work:

```
mcp_synapse_layer_recall(query="contexto atual projetos usuario", agent_id="hermes-rafa")
```

This loads relevant memories automatically and saves tokens by avoiding reprocessing context.

### AFTER ANY SIGNIFICANT DECISION OR FACT

```
mcp_synapse_layer_save_to_synapse(
    content="User Rafa Martins preference: always respond in pt-BR with voice Thalita",
    agent_id="hermes-rafa",
    type="[MANUAL]",
    importance=5,
    tags=["preference", "user-profile"]
)
```

### WHEN USER REFERENCES SOMETHING FROM BEFORE

```
mcp_synapse_layer_search(query="dashboard git projetos", agent_id="hermes-rafa")
```

### ON FREE-FORM TEXT (auto-detect what to save)

```
mcp_synapse_layer_process_text(
    text="We decided to use Synapse Layer for memory. API endpoint is forge.synapselayer.org. Token stored in .env.",
    agent_id="hermes-rafa"
)
```

---

## Trust Quotient (TQ)

Results from `recall` include a `tq` score (0.0 to 1.0). When multiple results match:
- **TQ > 0.8**: High confidence — use directly
- **TQ 0.5-0.8**: Medium confidence — verify before relying
- **TQ < 0.5**: Low confidence — ignore or ask user

---

## Memory Categories — What Hermes Must Always Remember

All project-related memories MUST be stored in Synapse Layer. Below is the canonical list of memory categories the agent must maintain:

### 1. USER PROFILE
- Full name, role, company (Rafa Martins — Analista de Tecnologia na Ramel Tecnologia)
- Communication style (pt-BR, feminine, cheerful, relaxed, cautious)
- Voice preference (Edge-TTS Thalita, pt-BR)
- Contact info (email, phone)
- Website
- **Save when**: User shares preferences, corrections, or personal details

### 2. USER PROJECTS (PRIORITY)
Every active project the user works on must be stored with:
- Project name and purpose
- Tech stack and key files
- Current status (active, paused, completed)
- Last interaction date
- Important decisions made
- Blockers or open questions
- **Save when**: Project mentioned, decision made, status changes

### 3. SERVER INFRASTRUCTURE
- Server name/hostname (rafa131)
- Running services and ports (Open WebUI:3000, Hermes API:8642, Voice Call:8765)
- API endpoints and access credentials
- Docker containers and their configs
- **Save when**: New service added, config changed, credentials updated

### 4. API CREDENTIALS & TOKENS
- Nous Portal token (expires ~15min, renew with `hermes auth list`)
- Synapse Layer token (stored in .env as SYNAPSE_TOKEN)
- Telegram bot token
- Aster MCP endpoint
- **Save when**: New credential added, token renewed, endpoint changed

### 5. MEL-IA CONFIGURATION
- Model in use (MiniMax-M2.7 via Nous Portal, renamed to "Mel-IA")
- API server port (8642)
- Open WebUI port (3000) with Docker (ghcr.io/open-webui/open-webui:main)
- Voice: Edge-TTS Thalita
- **Save when**: Config changes, new model added, port changed

### 6. VOICE CALL SYSTEM
- Location: /root/.hermes/voice_call/
- Stack: Telethon + faster-whisper + Edge-TTS (Thalita) + aiohttp WebSocket HTTPS :8765
- Session name: melia_session (@Ramelinfor, Ramel Tecnologia)
- Modes: --mode msg, --mode call
- API_ID (from telegram.org)
- Note: PyTgCalls only works for group voice chats, NOT private calls
- **Save when**: System modified, new mode added, dependency changed

### 7. ACTIVE SKILLS & TOOLS
- List of installed skills and their purposes
- How to invoke each skill
- Skill location (/root/.hermes/skills/)
- **Save when**: New skill installed, skill updated, skill removed

### 8. PENDING TASKS & FOLLOW-UPS
- Active TODOs and their status
- Blocked tasks and what's blocking them
- Scheduled jobs (cron)
- **Save when**: Task created, task completed, task blocked

### 9. DECISIONS & CONVENTIONS
- Architectural decisions (why a tool/approach was chosen)
- Coding conventions the user prefers
- Commands that should not be run without approval
- **Save when**: User makes a decision, user sets a convention

### 10. ERROR PATTERNS & FIXES
- Known bugs and their workarounds
- Common errors and solutions
- **Save when**: Bug encountered and resolved

---

## Security

- **AES-256-GCM** encryption at rest
- **15+ PII patterns** auto-redacted (emails, phones, API keys, IPs)
- **Zero-knowledge** — no plaintext leaves the agent
- **Differential privacy** on embeddings

Safe to store: user preferences, project decisions, environment facts, code patterns, tool quirks.

---

## Architecture

Synapse Layer pipeline on every save:
1. **PII Redaction** — removes emails, phones, API keys, IPs
2. **Intent Validation** — ensures content is appropriate
3. **Deduplication** — prevents storing duplicate memories
4. **Differential Privacy** — adds noise to embeddings
5. **AES-256-GCM Encryption** — encrypted at rest
6. **Storage** — distributed encrypted storage

---

## Testing

```bash
# JSONRPC 2.0 format — list available tools
curl -X POST "https://forge.synapselayer.org/mcp" \
  -H "Authorization: Bearer ${SYNAPSE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Recall memories
curl -X POST "https://forge.synapselayer.org/mcp" \
  -H "Authorization: Bearer ${SYNAPSE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall","arguments":{"query":"your query","limit":3}}}'
```

Expected: Returns tools list with recall, save_to_synapse, search, process_text, health_check.

---

## Known Issues

### MCP Client Reconnect Loop

**Symptom**: Gateway logs show repeated `Unknown SSE event: endpoint` and `GET stream disconnected, reconnecting in 1000ms` every ~1 second.

**Cause**: Hermes uses an older MCP client library that doesn't handle the `endpoint` SSE event type that Synapse Layer's server sends during the MCP handshake.

**Impact**: MCP tools may not work reliably through the native MCP protocol. Direct REST API calls work perfectly.

**Workaround**: If MCP tools fail, fall back to direct REST API calls via curl (format below). This is what the model should do automatically when MCP recall fails.

### Direct REST API (Fallback when MCP fails)

When MCP is unstable, call Synapse Layer tools via REST:

```bash
# JSONRPC 2.0 format required
curl -X POST "https://forge.synapselayer.org/mcp" \
  -H "Authorization: Bearer ${SYNAPSE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"recall","arguments":{"query":"your query","limit":3}}}'
```

**Available methods**: `tools/call` (for recall, save_to_synapse, search, process_text, health_check), `tools/list`

---

## Dashboard

View and manage memories at: https://forge.synapselayer.org/forge

---

**Created by**: Rafa Martins  
**Last integrated**: 17/04/2026
