# Brain Path Discovery — Multi-Repo Pattern

## When to Use

When a mission requires finding an existing governed LLM brain endpoint,
CLI, or API path across a multi-repo ecosystem. The goal is to establish
how a CLI tool can call an LLM for dynamic reasoning without becoming an
unconstrained agent, without managing provider secrets directly, and
without depending on Hermes as the user-facing interface.

## Canonical Example

The Cloe CLI v0.2 brain bridge: a CLI cockpit that calls the existing
OpenClaw governed brain path (`POST /brain/ask`) for dynamic NL responses
while preserving local safety gates.

## Discovery Sequence (Order Matters)

### 1. Check PATH for existing brain CLIs

```bash
which openclaw openclaw-lah-chat openclaw-aionui 2>/dev/null
```

If found, inspect the binary's registry:

```bash
# Show help to find subcommands
openclaw --help | grep -i 'brain\|ask\|chat\|llm\|prompt'

# If the npm bin is a wrapper, check the actual entrypoint
cat $(which openclaw) | head -30
```

### 2. Check running processes

```bash
# Is there a Gateway already running? What port?
ps aux | grep -i 'openclaw\|gateway\|acp\|aionui' | grep -v grep

# Is a WebSocket or HTTP server listening?
ss -tlnp 2>/dev/null | grep -E '4000|3000|8765|9080'
```

### 3. Check the source repo for brain registration

The npm bin may delegate to a wrapper that **doesn't** expose all
commands directly. Always check the actual source:

```bash
# Find the bin entrypoint
cd <openclaw-runtime-repo>
cat package.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('bin',{}), indent=2))"

# Run the actual JS entrypoint directly (may reveal hidden commands)
node bin/openclaw.js --help | grep -i 'brain\|ask'
```

### 4. Search source for brain ask patterns

```bash
# In the runtime repo
grep -R "brain ask\|chat.send\|DeepSeek\|gateway\|token-file\|session/prompt\|connect.challenge" \
  -n bin/ src/ scripts/ tests/ docs/ 2>/dev/null | head -100

# In all candidate repos
for repo in lah-stack-tools openclaw-runtime lah-brain lah-core clawx-runtime cartelogic-v2; do
  echo "=== $repo ==="
  grep -R "brain ask\|chat.send\|DeepSeek\|gateway.*token\|buildBrainAskResponse\|brain/ask" \
    -n bin/ src/ scripts/ tests/ 2>/dev/null | head -30 || echo "(empty)"
done
```

### 5. Check environment variables

```bash
# Current shell
env | grep -iE 'DEEPSEEK|OPENROUTER|OPENCLAW_BRAIN|ADMIN_API_KEY|GATEWAY_TOKEN' | sort

# Running process environ (may hold provider keys not exported to shell)
cat /proc/<PID>/environ 2>/dev/null | tr '\0' '\n' | grep -iE 'DEEPSEEK|OPENROUTER|OPENCLAW_BRAIN|ADMIN_API_KEY'

# Env files
find /home/deploy/lah-stack-runtime /opt/data -name "*.env" 2>/dev/null | head -10
```

### 6. Check the POST /brain/ask endpoint (if server is running)

```bash
# First check basic health
curl -s http://127.0.0.1:4000/health

# Check the brain endpoint with any available admin key
curl -s -X POST http://127.0.0.1:4000/brain/ask \
  -H 'Content-Type: application/json' \
  -H 'x-admin-api-key: <key>' \
  -d '{"question":"test"}'
```

### 7. Read the brain ask implementation

```bash
grep -n "buildBrainAskResponse" src/services/readonly-operator-cli-client.js
```

Look for:
- What env vars are required (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, etc.)
- What URL is used
- What auth is needed
- What contextual enrichment is done (CarteLogic memory queries, runtime snapshot)

### 8. Check if Gateway WebSocket chat.send is available

```bash
grep -R "chat.send\|extractGatewayBrainPrompt\|brain:\|llm:" \
  src/services/gateway/ 2>/dev/null | head -20
```

Look for:
- `brain:` and `llm:` prefix routing
- Token file conventions (`--token-file`, `gateway.token`)
- What session format is required (`agent:main:main`)
- Whether mutating actions are blocked

## Discovery Record Template

For each mission, record the findings as a structured summary:

```json
{
  "discovered_brain_path": "POST /brain/ask at http://127.0.0.1:4000",
  "requires_auth": true,
  "auth_type": "x-admin-api-key header",
  "auth_env_var": "OPENCLAW_ADMIN_API_KEY",
  "provider_config_env_vars": ["OPENCLAW_BRAIN_PROVIDER", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"],
  "additional_paths_found": [
    "ws://127.0.0.1:4000/gateway (chat.send with brain:/llm: prefix)",
    "openclaw-lah-chat --token-file <path> (CLI chat via Gateway)"
  ],
  "path_available_now": false,
  "reason_unavailable": "ADMIN_API_KEY_MISSING",
  "alternative_fallback": "deterministic_v0_1_response"
}
```

## Wiring Unresolved — How to Handle

If no safe governed brain path is available (no admin key, no gateway
token, no provider configured), the mission verdict should be:

```
CLOE_CLI_LLM_BRAIN_BRIDGE_ADAPTER_READY_WIRING_UNRESOLVED
```

Implement the full adapter interface with fallback. Document the
wiring gap clearly in the operator packet and continuity JSON.
Do NOT invent a new provider client or secret management approach.

## What NOT to Do

- Do NOT create a new provider client from Cloe
- Do NOT store provider API keys in Cloe-managed files
- Do NOT print provider secrets or tokens
- Do NOT assume the path exists without checking source code
- Do NOT start Gateway/OpenClaw if not already running
- Do NOT modify production OpenClaw runtime config
- Do NOT read ~/.openclaw or ~/.hermes configs
- Do NOT call provider APIs directly

## Brain Adapter Architecture (for safe CLI-to-LLM bridge)

When implementing the discovered brain path in a CLI tool, use this pattern:

```
Brain Adapter Interface:
  resolveBrainMode()     → 'off' | 'auto' | 'openclaw'
  isBrainEnabled()       → boolean
  buildBrainPrompt()     → string (user question + continuity + safety constraints)
  callBrain()            → { ok, answer, error, provider }
  answerWithBrainOrFallback() → { answer, marker, brainUsed, fallback, mode }
  buildBrainMarker()     → "🧠 Brain: <status>"

Modes:
  off       → Force deterministic behavior. No brain call.
  auto      → Try governed brain path, fallback to deterministic. (default)
  openclaw  → Require governed brain path. Error + fallback in strict mode.

Safety:
  - Brain may: reason, synthesize, explain, assess risk, propose next steps
  - Brain may NOT: install, write files, git mutations, call provider APIs,
                  start/stop services, manage secrets
  - Cloe still blocks mutation/live execution locally regardless of brain response
  - Response markers indicate brain/fallback mode
```