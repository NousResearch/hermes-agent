# AntSeed Setup Guide

AntSeed is a P2P AI inference network. You run a **buyer proxy** that connects to seller nodes offering models. This requires a funded wallet.

## 1. Install AntSeed CLI

```bash
npm install -g @antseed/cli
```

Verify: `antseed --version`

## 2. Start Buyer Proxy

The proxy is an OpenAI-compatible endpoint on `http://127.0.0.1:8377/v1/`.

```bash
antseed buyer start
```

Persistent state is stored in `~/.antseed/buyer.state.json` (survives restarts).

## 3. Configure Chain

```bash
# Default chain is Base. AntSeed CLI resolves contracts automatically from chainId.
# No need to hardcode contract addresses.
antseed buyer config set chain base
```

## 4. Wallet Setup

You need a wallet funded with USDC on Base. **Buyer wallet does NOT need ETH for gas** — the seller pays on-chain tx fees.

```bash
# Import existing wallet
antseed buyer wallet import <private-key>

# Or create new
antseed buyer wallet create
```

**CRITICAL:** Never move `identity.key` from the host it was created on.

## 5. Fund Wallet

Deposit USDC into the buyer contract:

```bash
antseed buyer deposit 1   # $1 USDC minimum recommended
```

Check balance:

```bash
antseed buyer status
```

## 6. Pin a Peer

```bash
# Find peers
antseed network browse --top 15

# Pin a peer by ID
antseed buyer connection set --peer <peer-id>
```

Verify pin: `antseed buyer status` — look for "Pinned peer" field.

## 7. Wire Hermes Config

Add AntSeed as a custom provider in `~/.hermes/config.yaml`:

```yaml
model:
  default: deepseek-v4-flash
  provider: antseed

custom_providers:
  - name: antseed
    base_url: http://127.0.0.1:8377/v1
    api_key: antseed-p2p
    api_mode: chat_completions
    models:
      - deepseek-v4-flash
      - claude-sonnet-4-6
      - minimax-m2.7
      # Sync with: curl -s http://127.0.0.1:8377/v1/models | jq '.data[].id'

auxiliary:
  title_generation:
    provider: antseed
    model: minimax-m2.7
  compression:
    provider: antseed
    model: minimax-m2.7

delegation:
  model: deepseek-v4-flash
  provider: antseed
  reasoning_effort: minimal
```

Key settings:

| Setting | Value | Why |
|---------|-------|-----|
| `api_mode` | `chat_completions` | Mandatory. `openai-responses` requires streaming — breaks auxiliaries |
| `api_key` | `antseed-p2p` | Convention — proxy ignores the key |
| `model` | AntSeed service ID | NOT OpenAI model name — use the ID from proxy `/v1/models` |

The `@antseed/api-adapter` handles automatic protocol translation (e.g., `chat_completions` → `anthropic-messages`).

## 8. Verify

```bash
# Test proxy
curl -s http://127.0.0.1:8377/v1/models -H "Authorization: Bearer antseed-p2p" | jq '.data[].id'

# Run full preflight
bash scripts/best-peer.sh any
```
