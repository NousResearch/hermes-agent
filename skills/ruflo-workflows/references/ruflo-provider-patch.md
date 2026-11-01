# Patch: ruflo agent-execute-core.js → provider routing (Ollama + Anthropic)

**Why:** ruflo's `agent_execute` hardcoded `https://api.anthropic.com/v1/messages` and sent
`x-api-key: <ANTHR...Y>`. It could only talk to Anthropic. The wrapper
(`~/.hermes/scripts/ruflo-mcp.sh`) now routes to **Ollama by default** (no Anthropic key
needed) or **Anthropic OAuth** as an option. This patch makes both work:

1. `ANTHROPIC_BASE_URL` is honored (falls back to `https://api.anthropic.com` when unset).
2. When `ANTHROPIC_AUTH_TOKEN` is set, sends `Authorization: Bearer <token>` instead of `x-api-key`
   (OAuth-compatible; works directly at `api.anthropic.com`, no proxy needed).
3. `resolveOllamaModel()` reads `OLLAMA_DEFAULT_MODEL` and maps any `claude-*` model id to it —
   so Anthropic model names route to the Ollama default (e.g. `deepseek-v4-flash:0731`).

**⚠️ Wiped by:** `npm update` / `npm install -g ruflo@latest` / any reinstall of ruflo.

## The durable fix: `~/.hermes/scripts/ruflo-patch-provider.sh`

The patch lives in this idempotent script (the **single source of truth**), and
`~/.hermes/scripts/ruflo-mcp.sh` **re-applies it on every MCP launch** —
so a reinstall can never silently break `agent_execute` again.

```bash
bash ~/.hermes/scripts/ruflo-patch-provider.sh            # apply / no-op if applied
echo $?                                                  # 0 applied|already, 1 anchor missing, 2 not found
```

Exit codes: `0` applied or already applied · `1` target exists but anchor changed (update this script) ·
`2` target not found (ruflo installed elsewhere — set `RUFLO_PATCH_TARGET` or pass path).

To simulate a wipe and self-heal:
```bash
npm install -g ruflo@latest        # patch gone
bash ~/.hermes/scripts/ruflo-mcp.sh   # auto re-applies, then starts MCP
```

## The patch itself (what the script applies)

**File:** `~/.local/lib/node_modules/ruflo/node_modules/@claude-flow/cli/dist/src/mcp-tools/agent-execute-core.js`

**What it changes:**
1. `ANTHROPIC_BASE_URL` is honored (falls back to `https://api.anthropic.com` when unset → zero behavior change for key-based users).
2. When `ANTHROPIC_AUTH_TOKEN` is set, sends `Authorization: Bearer <token>` instead of `x-api-key`.
3. `resolveOllamaModel` reads `process.env.OLLAMA_DEFAULT_MODEL` (was hardcoded `gpt-oss:120b-cloud`) and maps any `claude-*` id to it.

**Runtime env (injected by `~/.hermes/scripts/ruflo-mcp.sh`):**

*Ollama mode (default):*
- `RUFLO_PROVIDER=ollama` — forces ruflo's Ollama path
- `OLLAMA_API_KEY=<key>` — Ollama Cloud auth
- `OLLAMA_BASE_URL=https://ollama.com` — cloud endpoint (set to local for self-hosted)
- `OLLAMA_DEFAULT_MODEL=deepseek-v4-flash:0731` — default model for logical names

*Anthropic mode (`RUFLO_PROVIDER=anthropic`):*
- `ANTHROPIC_API_KEY=<oauth token>` — satisfies ruflo's provider gate
- `ANTHROPIC_AUTH_TOKEN=<oauth token>` — the patched code sends this as Bearer
- `ANTHROPIC_BASE_URL=https://api.anthropic.com` — direct (no proxy)

**Verified:** 2026-08-06 — full E2E via Ollama: `swarm_init` → `agent_spawn(coder)` →
`agent_execute("Reply with exactly: ollama-swarm-ok")` → `{"success": true, "output": "ollama-swarm-ok", "model": "deepseek-v4-flash:0731"}`
in 1.3s, then `swarm_shutdown` clean. No Anthropic key or proxy required.
