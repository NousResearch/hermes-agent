# LLM Gateway Reference

The Bankr LLM Gateway is a unified API for Claude, Gemini, GPT, and other models. It provides multi-provider access, cost tracking, automatic failover, and SDK compatibility through a single endpoint.

**Base URL:** `https://llm.bankr.bot`

The gateway accepts both `https://llm.bankr.bot` and `https://llm.bankr.bot/v1` — it normalizes paths automatically. Works with both OpenAI and Anthropic API formats.

## Authentication

The gateway uses your **LLM key** for authentication. The key resolution order:

1. `BANKR_LLM_KEY` environment variable
2. `llmKey` in `~/.bankr/config.json`
3. Falls back to your Bankr API key (`BANKR_API_KEY` / `apiKey`)

Most users only need a single key for both the agent API and the LLM gateway. Set a separate LLM key only if your keys have different permissions or rate limits.

**Dashboard:** Manage usage, credits, and auto top-up at [bankr.bot/llm](https://bankr.bot/llm). Top up credits at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits). Generate and configure API keys at [bankr.bot/api](https://bankr.bot/api).

### Setting the LLM Key

**Via CLI:**
```bash
bankr login --llm-key YOUR_LLM_KEY            # during login
bankr config set llmKey YOUR_LLM_KEY           # after login
```

**Via environment variable:**
```bash
export BANKR_LLM_KEY=your_llm_key_here
```

**Verify:**
```bash
bankr config get llmKey
```

## Available Models

| Model | Provider | Best For |
|-------|----------|----------|
| `claude-opus-4.6` | Anthropic | Most capable, advanced reasoning |
| `claude-opus-4.5` | Anthropic | Complex reasoning, architecture |
| `claude-sonnet-4.6` | Anthropic | Balanced speed and quality |
| `claude-sonnet-4.5` | Anthropic | Previous generation Sonnet |
| `claude-haiku-4.5` | Anthropic | Fast, cost-effective |
| `gemini-3-pro` | Google | Long context (2M tokens) |
| `gemini-3-flash` | Google | High throughput |
| `gemini-2.5-pro` | Google | Long context, multimodal |
| `gemini-2.5-flash` | Google | Speed, high throughput |
| `gpt-5.2` | OpenAI | Advanced reasoning |
| `gpt-5.2-codex` | OpenAI | Code generation |
| `gpt-5.4-mini` | OpenAI | Fast, economical (400K context, image input) |
| `gpt-5.4-nano` | OpenAI | Ultra-fast, lowest cost (400K context, image input) |
| `minimax-m2.7` | MiniMax | Balanced performance (204.8K context) |
| `kimi-k2.5` | Moonshot AI | Long-context reasoning |
| `qwen3-coder` | Alibaba | Code generation, debugging |
| `glm-5` | Z.ai | General purpose reasoning |
| `glm-5-turbo` | Z.ai | Fast, cost-effective |

```bash
# Fetch live model list from the gateway
bankr llm models
```

## Credits

> **New wallets start with $0 LLM credits.** Top up via CLI (`bankr llm credits add 25`) or at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits) before your first LLM call. Without credits, all gateway requests return HTTP 402.

Check your LLM gateway credit balance:

```bash
bankr llm credits
```

Top up credits from your wallet:

```bash
bankr llm credits add 25                   # Add $25 credits (USDC default)
bankr llm credits add 50 --token 0x...     # Add $50 from a specific token
bankr llm credits add 25 -y                # Skip confirmation prompt
```

Configure automatic top-up so credits never run out:

```bash
bankr llm credits auto                     # View current auto top-up config
bankr llm credits auto --enable --amount 25 --threshold 5 --tokens USDC
bankr llm credits auto --disable
```

When credits are exhausted, gateway requests will fail with HTTP 402.

> **LLM credits vs trading wallet:** These are completely separate balances on the same account and API key. Your trading wallet (ETH, SOL, USDC) is for on-chain transactions. LLM credits (USD) are for gateway API calls. Having crypto does NOT give you LLM credits.

## LLM Gateway Setup

If the user already has a Bankr account, they just need to configure the gateway. If not, they need to create one first.

### Have Bankr Account

1. Get an API key with **LLM Gateway** enabled:
   - **Have a key?** Enable LLM Gateway at [bankr.bot/api](https://bankr.bot/api)
   - **Need a key?** Generate via CLI: `bankr login email user@example.com` → `bankr login email user@example.com --code OTP --accept-terms --key-name "My Agent" --llm`
2. Run: `bankr llm setup openclaw --install`
3. Set default model in `~/.openclaw/openclaw.json`:
   ```json
   { "agents": { "defaults": { "model": { "primary": "bankr/claude-sonnet-4.6" } } } }
   ```
4. Verify credits: `bankr llm credits` (must show > $0 — top up via `bankr llm credits add 25` or at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits))
5. Restart OpenClaw or run: `openclaw gateway restart`

### Need Bankr Account

1. Send OTP: `bankr login email user@example.com`
2. Complete setup: `bankr login email user@example.com --code OTP --accept-terms --key-name "My Agent" --llm`
   - Can also create/configure keys at [bankr.bot/api](https://bankr.bot/api)
3. **Top up credits:** `bankr llm credits add 25` or at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits) — new wallets start with $0
4. Verify: `bankr llm credits` (must show > $0)
5. Run: `bankr llm setup openclaw --install`
6. Set default model in `~/.openclaw/openclaw.json` (see above)
7. Restart OpenClaw or run: `openclaw gateway restart`

> **Model names:** In OpenClaw, prefix with `bankr/` (e.g. `bankr/claude-sonnet-4.6`). In direct API calls, use bare IDs (e.g. `claude-sonnet-4.6`).

For the full 4-path setup guide (including users who don't have OpenClaw yet), see https://docs.bankr.bot/llm-gateway/openclaw

### Separate LLM and Agent API Keys

By default, one key is used for both. To use separate keys:

```bash
bankr config set llmKey YOUR_LLM_KEY           # after login
bankr login email user@example.com --llm-key YOUR_LLM_KEY  # during login
```

Key resolution: `BANKR_LLM_KEY` env var → `llmKey` in config → falls back to API key.

### Key Permissions

Manage at [bankr.bot/api](https://bankr.bot/api):

| Toggle | Controls |
|--------|----------|
| **LLM Gateway** | Access to `llm.bankr.bot` for model requests |
| **Agent API** | Access to wallet actions, prompts, and transactions |
| **Read Only** | Agent API only — restricts to read operations |

## Tool Integrations

### OpenClaw

Auto-install the Bankr provider into your OpenClaw config:

```bash
# Write config to ~/.openclaw/openclaw.json
bankr llm setup openclaw --install

# Preview the config without writing
bankr llm setup openclaw
```

This writes the following provider config (with your key and all available models):

```json
{
  "models": {
    "providers": {
      "bankr": {
        "baseUrl": "https://llm.bankr.bot",
        "apiKey": "your_key_here",
        "api": "openai-completions",
        "models": [
          { "id": "claude-sonnet-4.6", "name": "Claude Sonnet 4.6", "api": "anthropic-messages" },
          { "id": "claude-haiku-4.5", "name": "Claude Haiku 4.5", "api": "anthropic-messages" },
          { "id": "gemini-3-flash", "name": "Gemini 3 Flash" },
          { "id": "gpt-5.2", "name": "GPT 5.2" }
        ]
      }
    }
  }
}
```

Claude models are automatically configured with `"api": "anthropic-messages"` per-model overrides while all other models use the default `"api": "openai-completions"`.

To use a Bankr model as your default in OpenClaw, add to `openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "bankr/claude-sonnet-4.6"
      }
    }
  }
}
```

### Claude Code

Two ways to use Claude Code with the gateway:

**Option A: Launch directly (recommended)**

```bash
# Launch Claude Code through the gateway
bankr llm claude

# Pass any Claude Code flags through
bankr llm claude --model claude-sonnet-4.6
bankr llm claude --allowedTools Edit,Write,Bash
bankr llm claude --resume
```

All arguments after `claude` are forwarded to the `claude` binary. The CLI sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` automatically from your config (using `llmKey` if set, otherwise `apiKey`).

**Option B: Set environment variables**

```bash
# Print the env vars to add to your shell profile
bankr llm setup claude
```

This outputs:
```bash
export ANTHROPIC_BASE_URL="https://llm.bankr.bot"
export ANTHROPIC_AUTH_TOKEN="your_key_here"
```

Add these to `~/.zshrc` or `~/.bashrc` so all Claude Code sessions use the gateway.

### OpenCode

```bash
# Auto-install Bankr provider into ~/.config/opencode/opencode.json
bankr llm setup opencode --install

# Preview without writing
bankr llm setup opencode
```

### Cursor

```bash
# Get step-by-step setup instructions with your API key
bankr llm setup cursor
```

The setup adds your key as the OpenAI API Key, sets `https://llm.bankr.bot/v1` as the base URL override, and registers the available model IDs. When the base URL override is enabled, all model requests go through the gateway.

## Direct SDK Usage

The gateway is compatible with standard OpenAI and Anthropic SDKs — just override the base URL.

### curl (OpenAI format)

```bash
curl -X POST "https://llm.bankr.bot/v1/chat/completions" \
  -H "Authorization: Bearer $BANKR_LLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4.6",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### curl (Anthropic format)

```bash
curl -X POST "https://llm.bankr.bot/v1/messages" \
  -H "x-api-key: $BANKR_LLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4.6",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.bankr.bot/v1",
    api_key="your_bankr_key",
)

response = client.chat.completions.create(
    model="claude-sonnet-4.6",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### OpenAI SDK (TypeScript)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://llm.bankr.bot/v1",
  apiKey: "your_bankr_key",
});

const response = await client.chat.completions.create({
  model: "gemini-3-flash",
  messages: [{ role: "user", content: "Hello" }],
});
```

### Anthropic SDK (Python)

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="https://llm.bankr.bot",
    api_key="your_bankr_key",
)

message = client.messages.create(
    model="claude-sonnet-4.6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Model Deprecation

The gateway supports model deprecation with automatic redirect to replacement models:

- **Soft-deprecated models** still work but return `X-Model-Deprecated: true` and `X-Model-Replacement: <new-model-id>` response headers. Migrate to the replacement model at your earliest convenience.
- **Hard-deprecated models** return HTTP 410 (Gone) with the replacement model in the `X-Model-Replacement` header. Update your model ID to continue.

Check `bankr llm models` for current model status and replacement mappings.

## Troubleshooting

### 401 Unauthorized
- Verify key is set: `bankr config get llmKey` or `echo $BANKR_LLM_KEY`
- Check for leading/trailing spaces
- Ensure the key hasn't expired

### 402 Payment Required
- Credits exhausted: `bankr llm credits` shows $0.00
- Top up via CLI: `bankr llm credits add 25` or at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits) — this is the most common error for new users
- Set up auto top-up to prevent this: `bankr llm credits auto --enable --amount 25 --threshold 5 --tokens USDC`
- New wallets start with $0 — you must add credits before first use
- LLM credits are separate from your trading wallet balance

### Model not found
- Use exact model IDs (e.g., `claude-sonnet-4.6`, not `claude-3-sonnet`)
- Check available models: `bankr llm models`

### Claude Code not found
- `bankr llm claude` requires Claude Code to be installed separately
- Install: https://docs.anthropic.com/en/docs/claude-code

### Slow responses
- Try `claude-haiku-4.5` or `gemini-3-flash` for faster responses
- The gateway has automatic failover — temporary slowness usually resolves itself

---

**Documentation**: https://docs.bankr.bot/llm-gateway/overview
