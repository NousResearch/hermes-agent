# Synapse Layer Skill for Hermes Agent

## Overview

This skill integrates [Synapse Layer](https://synapselayer.org) — a zero-knowledge, encrypted, cross-session memory system — into Hermes Agent. All memories are encrypted with AES-256-GCM, pass through a 4-layer Cognitive Security Pipeline (PII redaction, differential privacy, intent validation, neural handover), and are retrievable from any channel (Telegram, WhatsApp, CLI, Discord).

## Credits

**Rafa Martins** — [synapselayer.org](https://synapselayer.org)

## What It Does

- **Persistent Memory**: Context survives between sessions. You don't repeat yourself.
- **Cross-Channel**: Telegram, WhatsApp, CLI, and Discord all share the same memory pool.
- **Trust Quotient (TQ)**: Every memory gets a confidence score (0.0–1.0) so the agent knows what to trust.
- **Zero-Knowledge Encryption**: Memories are encrypted before leaving the agent. The server never sees plaintext.
- **PII Auto-Redaction**: Emails, phone numbers, API keys, IPs are automatically stripped before storage.
- **Differential Privacy**: Noise is injected into embeddings to prevent inference attacks.

## Quick Start

### 1. Get a Token

Sign up at [forge.synapselayer.org](https://forge.synapselayer.org) and generate an `sk_connect_` token.

### 2. Configure Hermes

Add to `~/.hermes/config.yaml`:
```yaml
mcp_servers:
  synapse-layer:
    url: "https://forge.synapselayer.org/mcp"
    headers:
      Authorization: "Bearer ${SYNAPSE_TOKEN}"
    timeout: 120
    connect_timeout: 60
```

Add to `~/.hermes/.env`:
```
SYNAPSE_TOKEN=sk_connect_your_token_here
```

### 3. Restart Hermes
```bash
hermes restart
```

## Available Tools

| Tool | When to Use |
|------|-------------|
| `mcp_synapse_layer_recall` | **Every session start** — loads relevant context before responding |
| `mcp_synapse_layer_save_to_synapse` | After decisions, facts, preferences — immediate persist |
| `mcp_synapse_layer_search` | Cross-agent full-text search across all memories |
| `mcp_synapse_layer_process_text` | Auto-detect milestones/decisions in free-form text |
| `mcp_synapse_layer_health_check` | Verify connection and system status |

## Usage Examples

### Session Start (Automatic Recall)
```
mcp_synapse_layer_recall(query="current projects user preferences", agent_id="hermes-rafa")
```

### Save a Decision
```
mcp_synapse_layer_save_to_synapse(
    content="User prefers responses in pt-BR with Thalita voice",
    agent_id="hermes-rafa",
    type="[MANUAL]",
    importance=5,
    tags=["preference", "voice"]
)
```

### Search Across All Memories
```
mcp_synapse_layer_search(query="dashboard git projects", agent_id="hermes-rafa")
```

### Process Free-Form Text
```
mcp_synapse_layer_process_text(
    text="We chose Synapse Layer for memory. Endpoint is forge.synapselayer.org.",
    agent_id="hermes-rafa"
)
```

## Security Pipeline

Every memory passes through **4 non-bypassable layers**:

1. **Semantic Privacy Guard** — Regex-based PII/secret redaction (emails, SSNs, CPFs, credit cards, API keys, IPs)
2. **Differential Privacy** — Gaussian noise injection on embeddings (ε,δ)-DP
3. **Intent Validation** — Classifies into 8 categories (PREFERENCE, FACT, PROCEDURAL, BIO, EPHEMERAL, CRITICAL, UNKNOWN, INVALID)
4. **Neural Handover** — Conflict resolution and self-healing on recall

## Trust Quotient Guide

| TQ Range | Confidence | Action |
|----------|-----------|--------|
| 0.8–1.0 | High | Use directly |
| 0.5–0.8 | Medium | Verify before relying |
| 0.0–0.5 | Low | Ignore or ask user |

## Architecture

```
User Input → PII Redaction → Differential Privacy → Intent Validation → Encryption → Storage
                                                                            ↓
Recall ← Decryption ← Self-Healing ← Intent Re-validation ← ← ← ← ← ← ← ← ← ←
```

## Known Issues

### MCP Reconnect Loop

If gateway logs show:
```
Unknown SSE event: endpoint
GET stream disconnected, reconnecting in 1000ms
```

This is a known compatibility issue between Hermes's MCP client library and Synapse Layer's server. The **REST API works perfectly** as a fallback. The skill includes automatic fallback logic.

### Direct REST API (Always Works)

```bash
curl -X POST "https://forge.synapselayer.org/mcp" \
  -H "Authorization: Bearer ${SYNAPSE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"recall","arguments":{"query":"your query","limit":3}}}'
```

## Files

- `SKILL.md` — Full skill documentation with setup, usage, and troubleshooting
- `README.md` — This file (English overview)

## Links

- **Website**: https://synapselayer.org
- **Forge Dashboard**: https://forge.synapselayer.org/forge
- **GitHub**: https://github.com/SynapseLayer/synapse-layer
- **Docs**: https://synapselayer.org/docs

## Version History

- **0.3.0** (17/04/2026) — Fixed endpoint URL (/mcp not /api/mcp), added credits, improved README
- **0.2.0** — MCP client reconnect loop workaround documented
- **0.1.0** — Initial integration
