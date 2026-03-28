---
name: blockrun
description: Use BlockRun (blockrun.ai) as your LLM provider — 50+ models, no API key, pay per request with USDC on Base or Solana via x402 micropayments. Includes wallet setup, balance checking, and smart model routing.
version: 1.0.0
author: BlockRun
license: MIT
metadata:
  hermes:
    tags: [LLM, Payments, Crypto, Web3, Base, Solana, x402, OpenRouter, Models, Wallet]
    related_skills: [mcp]
---

# BlockRun / ClawRouter

BlockRun is an AI model gateway with **x402 micropayments** — think OpenRouter, but crypto-native. No API key, no subscription. Your wallet is your credential. Each request is paid automatically with USDC on Base or Solana.

**50+ models:** OpenAI GPT-5, Anthropic Claude, Google Gemini, DeepSeek, xAI Grok, and more.
**Free fallback:** `nvidia/gpt-oss-20b` is always available at no cost when balance is low.

---

## Quick Reference

| Task | Tool |
|------|------|
| Set up Base wallet | `blockrun_wallet_setup` |
| Set up Solana wallet | `blockrun_solana_wallet_setup` |
| Check balance (Base) | `blockrun_wallet_balance` |
| Check balance (Solana) | `blockrun_solana_wallet_balance` |
| Get wallet address | `blockrun_wallet_address` |
| Generate an image | `blockrun_image_generate` |
| Edit an image | `blockrun_image_edit` |
| Query prediction markets | `blockrun_prediction_markets` |
| Switch to BlockRun LLM provider | Edit `cli-config.yaml` → `provider: blockrun` |
| List available LLM models | `curl https://blockrun.ai/api/v1/models` |

---

## Step 1 — Install the SDK

```bash
pip install blockrun-llm eth-account
# For Solana support (optional):
pip install blockrun-llm solders base58
```

---

## Step 2 — Set Up Your Wallet

### Base (default — USDC on Base mainnet)

Call the `blockrun_wallet_setup` tool. It will:
- Create a new wallet at `~/.blockrun/.session` (or load existing)
- Show your address and current balance
- Print funding instructions

Or do it from the terminal:

```bash
python -c "
from blockrun_llm import setup_agent_wallet
client = setup_agent_wallet()
print('Address:', client.get_wallet_address())
print('Balance:', client.get_balance(), 'USDC')
"
```

Set the key in your environment:
```bash
export BLOCKRUN_WALLET_KEY=0x...    # from ~/.blockrun/.session
```

### Solana (alternative — SPL-USDC on Solana mainnet)

Call the `blockrun_solana_wallet_setup` tool, or:

```bash
python -c "
from blockrun_llm.solana_wallet import setup_agent_solana_wallet
client = setup_agent_solana_wallet()
print('Address:', client.get_wallet_address())
"
```

```bash
export SOLANA_WALLET_KEY=<base58-private-key>
export BLOCKRUN_CHAIN=solana
```

### Fund Your Wallet

Deposit $5–$20 USDC — enough for hundreds of requests.

```
Base:   https://blockrun.ai/fund
Solana: https://blockrun.ai/fund?chain=solana

Testnet (free, for development):
  https://faucet.circle.com  →  select "Base Sepolia"
  export NETWORK_MODE=testnet
```

---

## Step 3 — Configure hermes to Use BlockRun

Edit `~/.hermes/cli-config.yaml`:

```yaml
model:
  provider: "blockrun"           # or "clawrouter" (alias)
  default: "openai/gpt-5.2"     # any model below
  # chain: "solana"              # optional — defaults to "base"
```

That's it. All LLM calls now route through BlockRun and pay automatically.

---

## Available Models

### Free / Ultra-Cheap (no x402 required or very low cost)
| Model ID | Notes |
|----------|-------|
| `nvidia/gpt-oss-20b` | Smallest NVIDIA model — good for testing |
| `nvidia/gpt-oss-120b` | Larger NVIDIA model |
| `nvidia/kimi-k2.5` | Moonshot via NVIDIA |

### OpenAI
| Model ID | Notes |
|----------|-------|
| `openai/gpt-5.4` | Latest, 1M context |
| `openai/gpt-5.2` | Balanced |
| `openai/gpt-5-mini` | Fast + cheap |
| `openai/gpt-5.4-nano` | Smallest GPT-5 |
| `openai/o3` | Reasoning |
| `openai/o1` | Reasoning |

### Anthropic
| Model ID | Notes |
|----------|-------|
| `anthropic/claude-opus-4-6` | Most capable |
| `anthropic/claude-sonnet-4-6` | Balanced |
| `anthropic/claude-haiku-4-5` | Fast + cheap |

### Google
| Model ID | Notes |
|----------|-------|
| `google/gemini-3.1-pro` | Latest Gemini |
| `google/gemini-2.5-pro` | 1M context |
| `google/gemini-2.5-flash` | Fast |
| `google/gemini-3.1-flash-lite` | Cheapest Gemini |

### Other Providers
| Model ID | Provider |
|----------|---------|
| `x-ai/grok-4` | xAI |
| `x-ai/grok-3` | xAI |
| `deepseek/deepseek-v3` | DeepSeek |
| `minimax/minimax-m2.7` | MiniMax |
| `nvidia/nemotron-ultra-253b` | NVIDIA |

Full live list: `curl https://blockrun.ai/api/v1/models | python -m json.tool`

---

## Checking Balance & Spending

Use the `blockrun_wallet_balance` tool, or:

```python
from blockrun_llm import LLMClient

client = LLMClient()
print(f"Balance : ${client.get_balance():.4f} USDC")
print(f"Spent   : ${client.get_spending()['total_usd']:.4f} this session")
```

---

## Pricing

BlockRun charges provider cost + 5% margin. No subscription, no monthly fee.

Example costs per 1M tokens:
| Model | Input | Output |
|-------|-------|--------|
| GPT-5.2 | ~$2.50 | ~$10.00 |
| Claude Sonnet 4.6 | ~$3.00 | ~$15.00 |
| Gemini 2.5 Flash | ~$0.15 | ~$0.60 |
| DeepSeek V3 | ~$0.27 | ~$1.10 |
| `nvidia/gpt-oss-20b` | FREE | FREE |

Full pricing: `curl https://blockrun.ai/api/pricing`

---

## Switching Chains

```bash
# Use Base (default)
export BLOCKRUN_WALLET_KEY=0x...

# Use Solana
export SOLANA_WALLET_KEY=<base58-key>
export BLOCKRUN_CHAIN=solana

# Auto-detect: if only SOLANA_WALLET_KEY is set, Solana is used automatically
```

---

## Troubleshooting

**"BlockRun provider requires a wallet key"**
→ Run `blockrun_wallet_setup` tool or set `BLOCKRUN_WALLET_KEY` in `.env`

**"x402 payment failed"**
→ Check balance with `blockrun_wallet_balance`. Fund at https://blockrun.ai/fund

**"Low balance — switching to free model"**
→ Normal behavior. `nvidia/gpt-oss-20b` activates automatically. Top up to restore full model access.

**Want testnet (free, for development)?**
```bash
export NETWORK_MODE=testnet
# Get free testnet USDC: https://faucet.circle.com (select Base Sepolia)
```

---

## Image Generation

Generate images with `blockrun_image_generate`. Costs $0.02–$0.10/image, paid automatically via x402.

| Model key | Full ID | Price | Best for |
|-----------|---------|-------|----------|
| `nano-banana` | google/nano-banana | $0.05 | Fast, everyday use |
| `nano-banana-pro` | google/nano-banana-pro | $0.10 | High quality, up to 4K |
| `gpt-image-1` | openai/gpt-image-1 | $0.02 | Cheapest, good quality |
| `dall-e-3` | openai/dall-e-3 | $0.04 | Photorealistic |
| `flux` | black-forest/flux-1.1-pro | $0.04 | Artistic styles |

**Generate:**
```
Tool: blockrun_image_generate
Args: { "prompt": "a futuristic city at night", "model": "nano-banana", "size": "1024x1024" }
```

**Edit (inpainting):**
```
Tool: blockrun_image_edit
Args: {
  "prompt": "replace the sky with a sunset",
  "image": "data:image/png;base64,<base64-string>"
}
```

---

## Prediction Markets

Query live market data from Polymarket, Kalshi, dFlow, and more.
Cost: $0.001/call (GET) · $0.005/call (POST analytics).

**Available platforms:** Polymarket, Kalshi, dFlow, Limitless, Opinion, Predict.Fun, Binance

```
Tool: blockrun_prediction_markets
Args: {}   ← (no args = show all available endpoints)
```

### Common queries

```
# Polymarket — open markets
{ "path": "polymarket/markets" }

# Polymarket — top traders leaderboard
{ "path": "polymarket/leaderboard" }

# Polymarket — wallet positions & analytics
{ "path": "polymarket/wallet/0x123...abc" }

# Polymarket — smart money flows
{ "path": "polymarket/smart-money" }

# Kalshi — open markets
{ "path": "kalshi/markets" }

# dFlow — wallet P&L
{ "path": "dflow/wallet/pnl/0x123...abc" }

# Binance — BTC candlesticks
{ "path": "binance/candles/BTC-USD" }

# Cross-platform — same event on multiple markets
{ "path": "matching-markets" }
```

---

## How It Works

```
hermes-agent
    │
    ▼  (OpenAI-compatible request)
BlockRunX402Transport  ──── intercepts 402 ────► sign with wallet key (local)
    │                                             retry with PAYMENT-SIGNATURE
    ▼
blockrun.ai/api/v1  (or sol.blockrun.ai)
    │
    ▼  verify + settle USDC on-chain
    ▼
model provider (OpenAI / Anthropic / Google / ...)
    │
    ▼
response back to hermes
```

Payment signing is **non-custodial** — your private key never leaves your machine. Only the cryptographic signature is transmitted.
