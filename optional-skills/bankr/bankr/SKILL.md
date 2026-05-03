---
name: bankr
description: AI-powered crypto trading agent, wallet API, and LLM gateway via natural language. Use when the user wants to trade crypto, check portfolio balances (with PnL and NFTs), view token prices, search tokens, transfer crypto, manage NFTs, use leverage (Hyperliquid or Avantis), bet on Polymarket, deploy tokens, set up automated trading, sign and submit raw transactions, call x402 paid API endpoints, or access LLM models through the Bankr LLM gateway funded by your Bankr wallet. Supports Base, Ethereum, Polygon, Solana, and Unichain.
metadata:
  {
    "clawdbot":
      {
        "emoji": "📺",
        "homepage": "https://bankr.bot",
        "requires": { "bins": ["bankr"] },
      },
  }
---

# Bankr

Execute crypto trading and DeFi operations using natural language. Two integration options:

1. **Bankr CLI** (recommended) — Install `@bankr/cli` for a batteries-included terminal experience
2. **REST API** — Call `https://api.bankr.bot` directly from any language or tool

Both use the same API key. The API has two layers:
- **Wallet API** (`/wallet/*`) — Direct, synchronous endpoints for portfolio, transfers, signing, and transaction submission
- **Agent API** (`/agent/*`) — AI-powered async prompt endpoint for natural language operations

## Getting an API Key

Before using either option, you need a Bankr API key. Two ways to get one:

**Option A: Headless email login (recommended for agents)**

Two-step flow — send OTP, then verify and complete setup. See "First-Time Setup" below for the full guided flow with user preference prompts.

```bash
# Step 1 — send OTP to email
bankr login email user@example.com

# Step 2 — verify OTP and generate API key (options based on user preferences)
bankr login email user@example.com --code 123456 --accept-terms --key-name "My Agent" --read-write
```

This creates a wallet, accepts terms, and generates an API key — no browser needed. Before running step 2, ask the user which APIs they need (wallet, agent, both via `--read-write`, LLM gateway) and their preferred key name.

**Option B: Bankr Terminal**

1. Visit [bankr.bot/api](https://bankr.bot/api)
2. **Sign up / Sign in** — Enter your email and the one-time passcode (OTP) sent to it
3. **Generate an API key** — Create a key with **Wallet & Agent API** access enabled (the key starts with `bk_...`)

Both options automatically provision **EVM wallets** (Base, Ethereum, Polygon, Unichain) and a **Solana wallet** — no manual wallet setup needed.

## Option 1: Bankr CLI (Recommended)

### Install

```bash
bun install -g @bankr/cli
```

Or with npm:

```bash
npm install -g @bankr/cli
```

### First-Time Setup

#### Headless email login (recommended for agents)

When the user asks to log in with an email, walk them through this flow:

**Step 1 — Send verification code**

```bash
bankr login email <user-email>
```

**Step 2 — Ask the user for the OTP code and all preferences in a single message.** This avoids unnecessary back-and-forth. Ask for:

1. **OTP code** — the code they received via email
2. **Accept Terms of Service (REQUIRED)** — Present the [Terms of Service](https://bankr.bot/terms) link and confirm the user agrees. **The login command will fail for new users without `--accept-terms`.** You MUST ask for ToS acceptance and do not pass `--accept-terms` unless the user has explicitly confirmed.
3. **Which APIs do they need?**
   - **Wallet API** — enabled by default, use `--no-wallet-api` to disable
   - **Agent API** (`--agent-api`) — AI-powered prompts and natural language operations
   - **Token Launch** — enabled by default, use `--no-token-launch` to disable
   - Add `--read-write` to allow transactions (without it, enabled APIs are read-only)
4. **Enable LLM gateway access?** (`--llm`) — multi-model API at `llm.bankr.bot` (currently limited to beta testers). Skip if user doesn't need it.
5. **Key name?** (`--key-name`) — a display name for the API key (e.g. "My Agent", "Trading Bot")

**Step 3 — Construct and run the step 2 command** with the user's choices. **Do NOT execute if the user has not explicitly accepted the Terms of Service** — ask again if needed:

```bash
# Full access: wallet + agent with write + LLM
bankr login email <user-email> --code <otp> --accept-terms --key-name "My Agent" --agent-api --read-write --llm

# Agent with write access (AI can execute transactions)
bankr login email <user-email> --code <otp> --accept-terms --key-name "Trading Agent" --agent-api --read-write

# Default key (wallet + token launch, read-only)
bankr login email <user-email> --code <otp> --accept-terms --key-name "My Key"

# Agent read-only (research, prices, balances — no transactions)
bankr login email <user-email> --code <otp> --accept-terms --key-name "Research Agent" --agent-api

# LLM-only (no wallet, no token launch)
bankr login email <user-email> --code <otp> --accept-terms --key-name "LLM Client" --no-wallet-api --no-token-launch --llm
```

#### Login options reference

| Option | Description |
|--------|-------------|
| `--code <otp>` | OTP code received via email (step 2) |
| `--accept-terms` | Accept [Terms of Service](https://bankr.bot/terms) without prompting (required for new users) |
| `--key-name <name>` | Display name for the API key (e.g. "My Agent"). Prompted if omitted |
| `--no-wallet-api` | Disable Wallet API (enabled by default) |
| `--agent-api` | Enable Agent API (AI prompts, natural language operations) |
| `--read-write` | Disable read-only mode (allow transactions). Without this, enabled APIs are read-only |
| `--no-token-launch` | Disable Token Launch API (enabled by default) |
| `--llm` | Enable [LLM gateway](https://docs.bankr.bot/llm-gateway/overview) access (multi-model API at `llm.bankr.bot`). Currently limited to beta testers |
| `--allowed-ips <ips>` | Comma-separated IP/CIDR allowlist for the API key (e.g., `1.2.3.4,10.0.0.0/24`) |
| `--allowed-recipients <addresses>` | Comma-separated EVM/Solana addresses the key can send to (auto-classified by 0x prefix) |

**New key defaults** (when no flags are passed):

| Flag | Default | To change |
|------|---------|-----------|
| `walletApiEnabled` | Enabled | `--no-wallet-api` |
| `agentApiEnabled` | Disabled | `--agent-api` |
| `tokenLaunchApiEnabled` | Enabled | `--no-token-launch` |
| `llmGatewayEnabled` | Disabled | `--llm` |
| `readOnly` | Enabled (read-only) | `--read-write` |

Any option not provided on the command line will be prompted interactively by the CLI, so you can mix headless and interactive as needed.

#### Login with existing API key

If the user already has an API key:

```bash
bankr login --api-key bk_YOUR_KEY_HERE
```

If they need to create one at the Bankr Terminal:
1. Run `bankr login --url` — prints the terminal URL
2. Present the URL to the user, ask them to generate a `bk_...` key
3. Run `bankr login --api-key bk_THE_KEY`

#### Separate LLM Gateway Key (Optional)

If your LLM gateway key differs from your API key, pass `--llm-key` during login or run `bankr config set llmKey YOUR_LLM_KEY` afterward. When not set, the API key is used for both. See [references/llm-gateway.md](references/llm-gateway.md) for full details.

#### Verify Setup

```bash
bankr whoami
bankr wallet portfolio
bankr agent prompt "What is my balance?"
```

## Option 2: REST API (Direct)

No CLI installation required — call the API directly with `curl`, `fetch`, or any HTTP client.

### Authentication

All requests require an `X-API-Key` header:

```bash
curl -X POST "https://api.bankr.bot/agent/prompt" \
  -H "X-API-Key: bk_YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is my ETH balance?"}'
```

### Quick Example: Submit → Poll → Complete

```bash
# 1. Submit a prompt — returns a job ID
JOB=$(curl -s -X POST "https://api.bankr.bot/agent/prompt" \
  -H "X-API-Key: $BANKR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is my ETH balance?"}')
JOB_ID=$(echo "$JOB" | jq -r '.jobId')

# 2. Poll until terminal status
while true; do
  RESULT=$(curl -s "https://api.bankr.bot/agent/job/$JOB_ID" \
    -H "X-API-Key: $BANKR_API_KEY")
  STATUS=$(echo "$RESULT" | jq -r '.status')
  [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ] && break
  sleep 2
done

# 3. Read the response
echo "$RESULT" | jq -r '.response'
```

### Conversation Threads

Every prompt response includes a `threadId`. Pass it back to continue the conversation:

```bash
# Start — the response includes a threadId
curl -X POST "https://api.bankr.bot/agent/prompt" \
  -H "X-API-Key: $BANKR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the price of ETH?"}'
# → {"jobId": "job_abc", "threadId": "thr_XYZ", ...}

# Continue — pass threadId to maintain context
curl -X POST "https://api.bankr.bot/agent/prompt" \
  -H "X-API-Key: $BANKR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "And what about SOL?", "threadId": "thr_XYZ"}'
```

Omit `threadId` to start a new conversation. CLI equivalent: `bankr agent prompt --continue` (reuses last thread) or `bankr agent prompt --thread <id>`.

### API Endpoints Summary

#### Wallet API (`/wallet/*`) — Direct endpoints (synchronous)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/wallet/me` | GET | Read | Wallet info (address, chains) |
| `/wallet/portfolio` | GET | Read | Portfolio balances, supports `?include=pnl,nfts` for progressive loading |
| `/wallet/transfer` | POST | Write | Transfer tokens (multi-chain, supports `allowedRecipients` enforcement) |
| `/wallet/sign` | POST | Write | Sign messages, typed data, or transactions |
| `/wallet/submit` | POST | Write | Submit raw transactions to chain |

- **Read endpoints** (`/wallet/me`, `/wallet/portfolio`) — any valid API key with a wallet
- **Write endpoints** (`/wallet/transfer`, `/wallet/sign`, `/wallet/submit`) — require `walletApiEnabled`, `readOnly` check, and `allowedRecipients` enforcement
- IP allowlist enforced on all endpoints

#### Agent API (`/agent/*`) — AI-powered endpoints (async)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/prompt` | POST | Submit a prompt (async, returns job ID) |
| `/agent/job/{jobId}` | GET | Check job status and results |
| `/agent/job/{jobId}/cancel` | POST | Cancel a running job |

#### Deprecated endpoints

The following `/agent/*` endpoints still work but are deprecated in favor of `/wallet/*`:

| Deprecated | Use Instead |
|-----------|-------------|
| `GET /agent/me` | `GET /wallet/me` |
| `GET /agent/balances` | `GET /wallet/portfolio` |
| `POST /agent/sign` | `POST /wallet/sign` |
| `POST /agent/submit` | `POST /wallet/submit` |

For full API details (request/response schemas, job states, rich data, polling strategy), see:

**Reference**: [references/api-workflow.md](references/api-workflow.md) | [references/sign-submit-api.md](references/sign-submit-api.md)

## CLI Command Reference (v0.2.0)

CLI 0.2.0 organizes commands into three namespaces: `wallet`, `agent`, and `tokens`. Old flat commands (`balances`, `prompt`, `status`, etc.) still work as deprecated aliases.

### `bankr wallet` — Wallet Operations

| Command | Description |
|---------|-------------|
| `bankr wallet` | Show wallet info (default: whoami) |
| `bankr wallet portfolio` | Portfolio balances across all chains (hides tokens under $1 by default) |
| `bankr wallet portfolio --pnl` | Include profit/loss data |
| `bankr wallet portfolio --nfts` | Include NFT holdings |
| `bankr wallet portfolio --all` | Include both PnL and NFTs |
| `bankr wallet portfolio --chain <chains>` | Filter by chain(s): base, polygon, mainnet, unichain, solana (comma-separated) |
| `bankr wallet portfolio --json` | Output raw JSON |
| `bankr wallet transfer --to <recipient> --token <symbol> --amount <amount>` | Transfer tokens with symbol resolution |
| `bankr wallet transfer --to <recipient> --token USDC --amount 50 --chain base` | Transfer with explicit chain |
| `bankr wallet sign` | Sign messages/typed data/transactions |
| `bankr wallet submit` | Submit raw transactions |

### `bankr agent` — AI Agent Operations

| Command | Description |
|---------|-------------|
| `bankr agent prompt <text>` | Send a prompt to the Bankr AI agent |
| `bankr agent prompt --continue <text>` | Continue the most recent conversation thread |
| `bankr agent prompt --thread <id> <text>` | Continue a specific conversation thread |
| `bankr agent status <jobId>` | Check the status of a running job |
| `bankr agent cancel <jobId>` | Cancel a running job |
| `bankr agent profile` | View/manage agent profile |
| `bankr agent skills` | Show all Bankr AI agent skills with examples |

### `bankr tokens` — Token Discovery

| Command | Description |
|---------|-------------|
| `bankr tokens search <query>` | Search for tokens by name or symbol |
| `bankr tokens info <symbol-or-address>` | Get detailed token information |

### Auth & Config Commands

| Command | Description |
|---------|-------------|
| `bankr login` | Authenticate with the Bankr API (interactive menu) |
| `bankr login email <address>` | Send OTP to email (headless step 1) |
| `bankr login email <address> --code <otp> [options]` | Verify OTP and complete setup (headless step 2) |
| `bankr login --api-key <key>` | Login with an existing API key directly |
| `bankr login --api-key <key> --llm-key <key>` | Login with separate LLM gateway key |
| `bankr login --url` | Print Bankr Terminal URL for API key generation |
| `bankr logout` | Clear stored credentials |
| `bankr whoami` | Show current authentication info |
| `bankr config get [key]` | Get config value(s) |
| `bankr config set <key> <value>` | Set a config value |
| `bankr --config <path> <command>` | Use a custom config file path |

Valid config keys: `apiKey`, `apiUrl`, `llmKey`, `llmUrl`

Default config location: `~/.bankr/config.json`. Override with `--config` or `BANKR_CONFIG` env var.

### Deprecated Aliases

Old flat commands still work but prefer the namespaced versions:

| Deprecated | Use Instead |
|-----------|-------------|
| `bankr prompt` | `bankr agent` |
| `bankr status` | `bankr agent status` |
| `bankr cancel` | `bankr agent cancel` |
| `bankr balances` | `bankr wallet portfolio` |
| `bankr profile` | `bankr agent profile` |
| `bankr sign` | `bankr wallet sign` |
| `bankr submit` | `bankr wallet submit` |
| `bankr skills` | `bankr agent skills` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BANKR_API_KEY` | API key (overrides stored key) |
| `BANKR_API_URL` | API URL (default: `https://api.bankr.bot`) |
| `BANKR_LLM_KEY` | LLM gateway key (falls back to `BANKR_API_KEY` if not set) |
| `BANKR_LLM_URL` | LLM gateway URL (default: `https://llm.bankr.bot`) |

Environment variables override config file values. Config file values override defaults.

### LLM Gateway Commands

| Command | Description |
|---------|-------------|
| `bankr llm models` | List available LLM models |
| `bankr llm credits` | Check credit balance |
| `bankr llm credits add <amount> [--token <addr>] [-y]` | Top up LLM credits from wallet |
| `bankr llm credits auto [--enable/--disable] [--amount] [--threshold] [--tokens]` | View or configure auto top-up |
| `bankr llm setup openclaw [--install]` | Generate or install OpenClaw config |
| `bankr llm setup opencode [--install]` | Generate or install OpenCode config |
| `bankr llm setup claude` | Show Claude Code environment setup |
| `bankr llm setup cursor` | Show Cursor IDE setup instructions |
| `bankr llm claude [args...]` | Launch Claude Code via the Bankr LLM Gateway |

## Core Usage

### Simple Query

For straightforward requests that complete quickly:

```bash
bankr agent prompt "What is my ETH balance?"
bankr agent prompt "What's the price of Bitcoin?"
```

The CLI handles the full submit-poll-complete workflow automatically. You can also use the shorthand — any unrecognized command is treated as a prompt:

```bash
bankr What is the price of ETH?
```

### Interactive Prompt

For prompts containing `$` or special characters that the shell would expand:

```bash
# Interactive mode — no shell expansion issues
bankr agent prompt
# Then type: Buy $50 of ETH on Base

# Or pipe input
echo 'Buy $50 of ETH on Base' | bankr agent prompt
```

### Conversation Threads

Continue a multi-turn conversation with the agent:

```bash
# First prompt — starts a new thread automatically
bankr agent prompt "What is the price of ETH?"
# → Thread: thr_ABC123

# Continue the conversation (agent remembers the ETH context)
bankr agent prompt --continue "And what about BTC?"
bankr agent prompt -c "Compare them"

# Resume any thread by ID
bankr agent prompt --thread thr_ABC123 "Show me ETH chart"
```

Thread IDs are automatically saved to config after each prompt. The `--continue` / `-c` flag reuses the last thread.

### Manual Job Control

For advanced use or long-running operations:

```bash
# Submit and get job ID
bankr agent prompt "Buy $100 of ETH"
# → Job submitted: job_abc123

# Check status of a specific job
bankr agent status job_abc123

# Cancel if needed
bankr agent cancel job_abc123
```

## LLM Gateway

The [Bankr LLM Gateway](https://docs.bankr.bot/llm-gateway/overview) is a unified API for Claude, Gemini, GPT, and other models — multi-provider access, cost tracking, automatic failover, and SDK compatibility through a single endpoint.

**Base URL:** `https://llm.bankr.bot` | **Dashboard:** [bankr.bot/llm](https://bankr.bot/llm) | **API Keys:** [bankr.bot/api](https://bankr.bot/api)

### Key Concepts

- Uses your `llmKey` if configured, otherwise falls back to your API key
- **LLM credits** (USD) and **trading wallet** (crypto) are completely separate balances — having crypto does NOT give you LLM credits
- **New accounts start with $0 LLM credits** — top up via `bankr llm credits add 25` or at [bankr.bot/llm?tab=credits](https://bankr.bot/llm?tab=credits) before making any LLM calls, or you will get a 402 error
- Check credits: `bankr llm credits` | Top up: `bankr llm credits add <amount>` | Auto top-up: `bankr llm credits auto --enable --amount 25 --tokens USDC`
- In OpenClaw config, prefix model IDs with `bankr/` (e.g. `bankr/claude-sonnet-4.6`). In direct API calls, use bare IDs (e.g. `claude-sonnet-4.6`)

### Quick Commands

```bash
bankr llm models                           # List available models
bankr llm credits                          # Check credit balance
bankr llm credits add 25                   # Top up $25 credits (USDC)
bankr llm credits auto --enable --amount 25 --tokens USDC  # Auto top-up
bankr llm setup openclaw --install         # Install Bankr provider into OpenClaw
bankr llm setup claude                     # Print Claude Code env vars
bankr llm claude                           # Launch Claude Code through gateway
```

### Model Deprecation

The gateway supports model deprecation with auto-redirect to replacement models. Deprecated models return `X-Model-Deprecated` and `X-Model-Replacement` response headers. Hard-deprecated models return HTTP 410 — update your model ID to the replacement indicated in the header.

For full details — setup paths, model list, provider config, SDK examples, key management, and troubleshooting — see:

**Reference**: [references/llm-gateway.md](references/llm-gateway.md)

## Capabilities Overview

### Trading Operations

- **Token Swaps**: Buy/sell/swap tokens across chains
- **Cross-Chain**: Bridge tokens between chains
- **Limit Orders**: Execute at target prices
- **Stop Loss**: Automatic sell protection
- **DCA**: Dollar-cost averaging strategies
- **TWAP**: Time-weighted average pricing

**Reference**: [references/token-trading.md](references/token-trading.md)

### Portfolio Management

- Check balances across all chains (`bankr wallet portfolio` or `GET /wallet/portfolio`)
- View USD valuations with optional PnL tracking (`--pnl` or `?include=pnl`)
- View NFT holdings (`--nfts` or `?include=nfts`)
- Track holdings by token or chain
- Real-time price updates
- Multi-chain aggregation
- Filter by chain: `bankr wallet portfolio --chain base,solana` or `GET /wallet/portfolio?chains=base,solana`

**Reference**: [references/portfolio.md](references/portfolio.md)

### Market Research

- Token prices and market data
- Technical analysis (RSI, MACD, etc.)
- Social sentiment analysis
- Price charts
- Trending tokens
- Token comparisons

**Reference**: [references/market-research.md](references/market-research.md)

### Transfers

- Send to addresses, ENS, or social handles
- Multi-chain support
- Flexible amount formats
- Social handle resolution (Twitter, Farcaster, Telegram)

**Reference**: [references/transfers.md](references/transfers.md)

### NFT Operations

- Browse and search collections
- View floor prices and listings
- Purchase NFTs via OpenSea
- View your NFT portfolio
- Transfer NFTs
- Mint from supported platforms

**Reference**: [references/nft-operations.md](references/nft-operations.md)

### Polymarket Betting

- Search prediction markets
- Check odds
- Place bets on outcomes
- View positions
- Redeem winnings

**Reference**: [references/polymarket.md](references/polymarket.md)

### Leverage Trading

- **Hyperliquid** (primary) — Perpetual futures on Hyperliquid L1 with on-chain order book. Crypto, stocks (TSLA, AAPL, NVDA via HIP-3), spot trading. Up to 50x leverage.
- **Avantis** (secondary) — Perpetuals on Base for crypto (up to 50x), forex and commodities (up to 100x)
- Stop loss, take profit, and position management on both platforms

**Reference**: [references/leverage-trading.md](references/leverage-trading.md) | [references/hyperliquid.md](references/hyperliquid.md)

### Token Deployment

- **EVM (Base)**: Deploy ERC20 tokens via Clanker with customizable metadata and social links
- **Solana**: Launch SPL tokens via Raydium LaunchLab with bonding curve and auto-migration to CPMM
- Creator fee claiming on both chains
- Fee Key NFTs for Solana (50% LP trading fees post-migration)
- Optional fee recipient designation with 99.9%/0.1% split (Solana)
- Both creator AND fee recipient can claim bonding curve fees (gas sponsored)
- Optional vesting parameters (Solana)
- Rate limits: 1/day standard, 10/day Bankr Club (gas sponsored within limits)

**Reference**: [references/token-deployment.md](references/token-deployment.md)

### Automation

- Limit orders
- Stop loss orders
- DCA (dollar-cost averaging)
- TWAP (time-weighted average price)
- Scheduled commands

**Reference**: [references/automation.md](references/automation.md)

### x402 Paid API Calls

The agent can discover and call x402-protected API endpoints, automatically handling USDC payments on Base:

- **Discover** endpoints in the Bankr registry or via web search
- **Inspect** endpoint pricing, methods, and input/output schemas
- **Call** endpoints with automatic payment signing (max $10/request)
- Works with any x402-compatible endpoint (Bankr-hosted or external)

**Reference**: [references/x402-cloud.md](references/x402-cloud.md)

### Arbitrary Transactions

- Submit raw EVM transactions with explicit calldata
- Custom contract calls to any address
- Execute pre-built calldata from other tools
- Value transfers with data

**Reference**: [references/arbitrary-transaction.md](references/arbitrary-transaction.md)

## Supported Chains

| Chain       | Native Token | Best For                      | Gas Cost |
| ----------- | ------------ | ----------------------------- | -------- |
| Base        | ETH          | Memecoins, general trading    | Very Low |
| Polygon     | POL          | Gaming, NFTs, frequent trades | Very Low |
| Ethereum    | ETH          | Blue chips, high liquidity    | High     |
| Solana      | SOL          | High-speed trading            | Minimal  |
| Unichain    | ETH          | Newer L2 option               | Very Low |
| World Chain | ETH          | Uniswap V3/V4 swaps          | Very Low |
| Arbitrum    | ETH          | DeFi, low-cost transactions   | Very Low |
| BNB Chain   | BNB          | BSC ecosystem trading         | Low      |

## Safety & Access Control

**Dedicated Agent Wallet**: When building autonomous agents, create a separate Bankr account rather than using your personal wallet. This isolates agent funds — if a key is compromised, only the agent wallet is exposed. Fund it with limited amounts and replenish as needed.

**API Key Types**: Bankr uses a single key format (`bk_...`) with capability flags (`walletApiEnabled`, `agentApiEnabled`, `tokenLaunchApiEnabled`, `llmGatewayEnabled`). You can optionally configure a separate LLM Gateway key via `bankr config set llmKey` or `BANKR_LLM_KEY` — useful when you want independent revocation or different permissions for agent vs LLM access.

**Read-Only API Keys**: New keys default to `readOnly: true`. This filters all write tools (swaps, transfers, staking, token launches, etc.) from agent sessions. The `/wallet/sign`, `/wallet/submit`, and `/wallet/transfer` write endpoints return 403. Use `--read-write` during login or toggle in the web settings to disable. Ideal for monitoring bots and research agents.

**IP Whitelisting**: Set `allowedIps` on your API key to restrict usage to specific IPs or CIDR ranges (e.g., `10.0.0.0/24`). Requests from non-whitelisted IPs are rejected with 403 at the auth layer.

**Rate Limits**: 100 messages/day (standard), 1,000/day (Bankr Club), or custom per key. Resets 24h from first message (rolling window). LLM Gateway uses a credit-based system.

**Key safety rules:**
- Store keys in environment variables (`BANKR_API_KEY`, `BANKR_LLM_KEY`), never in source code
- Add `~/.bankr/` and `.env` to `.gitignore` — the CLI stores credentials in `~/.bankr/config.json`
- Test with small amounts on low-cost chains (Base, Polygon) before production use
- Use `waitForConfirmation: true` with `/wallet/submit` — transactions execute immediately with no confirmation prompt
- Rotate keys periodically via the dashboard or API key rotation endpoint, and revoke immediately if compromised at [bankr.bot/api](https://bankr.bot/api)

**Reference**: [references/safety.md](references/safety.md)

## Common Patterns

### Check Before Trading

```bash
# Check balance
bankr wallet portfolio --chain base

# Check price
bankr agent prompt "What's the current price of PEPE?"

# Then trade
bankr agent prompt "Buy $20 of PEPE on Base"
```

### Portfolio Review

```bash
# Direct portfolio check (no AI agent, instant response)
bankr wallet portfolio
bankr wallet portfolio --pnl        # Include profit/loss data
bankr wallet portfolio --nfts       # Include NFT holdings
bankr wallet portfolio --all        # PnL + NFTs
bankr wallet portfolio --chain base
bankr wallet portfolio --chain base,solana
bankr wallet portfolio --json

# Via AI agent (natural language, richer context)
bankr agent prompt "Show my complete portfolio"

# Chain-specific
bankr agent prompt "What tokens do I have on Base?"

# Token-specific
bankr agent prompt "Show my ETH across all chains"
```

### Set Up Automation

```bash
# DCA strategy
bankr agent prompt "DCA $100 into ETH every week"

# Stop loss protection
bankr agent prompt "Set stop loss for my ETH at $2,500"

# Limit order
bankr agent prompt "Buy ETH if price drops to $3,000"
```

### Market Research

```bash
# Token discovery
bankr tokens search PEPE
bankr tokens info USDC

# Price and analysis
bankr agent prompt "Do technical analysis on ETH"

# Trending tokens
bankr agent prompt "What tokens are trending on Base?"

# Compare tokens
bankr agent prompt "Compare ETH vs SOL"
```

## API Workflow

Bankr uses an asynchronous job-based API:

1. **Submit** — Send prompt (with optional `threadId`), get job ID and thread ID
2. **Poll** — Check status every 2 seconds
3. **Complete** — Process results when done
4. **Continue** — Reuse `threadId` for multi-turn conversations

The `bankr agent prompt` command handles this automatically. When using the REST API directly, implement the poll loop yourself (see Option 2 above or the reference below). For manual job control via CLI, use `bankr agent status <jobId>` and `bankr agent cancel <jobId>`.

For details on the API structure, job states, polling strategy, and error handling, see:

**Reference**: [references/api-workflow.md](references/api-workflow.md)

### Synchronous Endpoints (Wallet API)

For direct signing and transaction submission, use the Wallet API synchronous endpoints:

- **POST /wallet/sign** - Sign messages, typed data, or transactions without broadcasting
- **POST /wallet/submit** - Submit raw transactions directly to the blockchain
- **POST /wallet/transfer** - Transfer tokens with symbol resolution and multi-chain support

These endpoints return immediately (no polling required) and are ideal for:
- Authentication flows (sign messages)
- Gasless approvals (sign EIP-712 permits)
- Pre-built transactions (submit raw calldata)
- Programmatic token transfers

**Reference**: [references/sign-submit-api.md](references/sign-submit-api.md)

## Error Handling

Common issues and fixes:

- **Authentication errors** → Run `bankr login` or check `bankr whoami` (CLI), or verify your `X-API-Key` header (REST API)
- **Insufficient balance** → Add funds or reduce amount
- **Token not found** → Verify symbol and chain
- **Transaction reverted** → Check parameters and balances
- **Rate limiting** → Wait and retry

For comprehensive error troubleshooting, setup instructions, and debugging steps, see:

**Reference**: [references/error-handling.md](references/error-handling.md)

## Best Practices

### Security

1. Never share your API key or LLM key
2. Use a dedicated agent wallet with limited funds for autonomous agents
3. Use read-only API keys for monitoring and research-only agents
4. Set IP whitelisting for server-side agents with known IPs
5. Verify addresses before large transfers
6. Use stop losses for leverage trading
7. Store keys in environment variables, not source code — add `~/.bankr/` to `.gitignore`

See [references/safety.md](references/safety.md) for comprehensive safety guidance.

### Trading

1. Check balance before trades
2. Specify chain for lesser-known tokens
3. Consider gas costs (use Base/Polygon for small amounts)
4. Start small, scale up after testing
5. Use limit orders for better prices

### Automation

1. Test automation with small amounts first
2. Review active orders regularly
3. Set realistic price targets
4. Always use stop loss for leverage
5. Monitor execution and adjust as needed

## Tips for Success

### For New Users

- Start with balance checks and price queries
- Test with $5-10 trades first
- Use Base for lower fees
- Enable trading confirmations initially
- Learn one feature at a time

### For Experienced Users

- Leverage automation for strategies
- Use multiple chains for diversification
- Combine DCA with stop losses
- Explore advanced features (leverage, Polymarket)
- Monitor gas costs across chains

## Prompt Examples by Category

### Trading

- "Buy $50 of ETH on Base"
- "Swap 0.1 ETH for USDC"
- "Sell 50% of my PEPE"
- "Bridge 100 USDC from Polygon to Base"

### Portfolio

- `bankr wallet portfolio` (direct, no AI processing — hides low-value tokens by default)
- `bankr wallet portfolio --pnl` (include profit/loss)
- `bankr wallet portfolio --nfts` (include NFT holdings)
- `bankr wallet portfolio --all` (PnL + NFTs)
- `bankr wallet portfolio --chain base` (single chain)
- "Show my portfolio"
- "What's my ETH balance?"
- "Total portfolio value"
- "Holdings on Base"

### Market Research

- "What's the price of Bitcoin?"
- "Analyze ETH price"
- "Trending tokens on Base"
- "Compare UNI vs SUSHI"

### Transfers

- "Send 0.1 ETH to vitalik.eth"
- "Transfer $20 USDC to @friend"
- "Send 50 USDC to 0x123..."

### NFTs

- "Show Bored Ape floor price"
- "Buy cheapest Pudgy Penguin"
- "Show my NFTs"

### Polymarket

- "What are the odds Trump wins?"
- "Bet $10 on Yes for [market]"
- "Show my Polymarket positions"

### Leverage

- "Long $100 of BTC on hyperliquid with 10x"
- "Short ETH with 5x on hyperliquid"
- "Open 5x long on ETH with $100"
- "Short BTC 10x with stop loss at $45k"
- "Show my hyperliquid positions"
- "Show my Avantis positions"

### Automation

- "DCA $100 into ETH weekly"
- "Set limit order to buy ETH at $3,000"
- "Stop loss for all holdings at -20%"

### Token Deployment

**Solana (LaunchLab):**

- "Launch a token called MOON on Solana"
- "Launch a token called FROG and give fees to @0xDeployer"
- "Deploy SpaceRocket with symbol ROCK"
- "Launch BRAIN and route fees to 7xKXtg..."
- "How much fees can I claim for MOON?"
- "Claim my fees for MOON" (works for creator or fee recipient)
- "Show my Fee Key NFTs"
- "Claim my fee NFT for ROCKET" (post-migration)
- "Transfer fees for MOON to 7xKXtg..."

**EVM (Clanker):**

- "Deploy a token called BankrFan with symbol BFAN on Base"
- "Claim fees for my token MTK"

### x402 Paid API Calls

- "Find x402 endpoints for sentiment analysis"
- "Call the weather endpoint on x402"
- "What x402 endpoints are available for price data?"

### Arbitrary Transactions

- "Submit this transaction: {to: 0x..., data: 0x..., value: 0, chainId: 8453}"
- "Execute this calldata on Base: {...}"
- "Send raw transaction with this JSON: {...}"

### Transfers (Direct)

Transfer tokens via CLI or Wallet API without AI processing:

```bash
# CLI — token symbol resolution built in
bankr wallet transfer --to vitalik.eth --token USDC --amount 50 --chain base
bankr wallet transfer --to 0x1234... --token ETH --amount 0.1

# REST API
curl -X POST "https://api.bankr.bot/wallet/transfer" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"to": "vitalik.eth", "token": "USDC", "amount": "50", "chain": "base"}'
```

### Sign API (Synchronous)

Direct message signing without AI processing:

```bash
# Sign a plain text message
curl -X POST "https://api.bankr.bot/wallet/sign" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signatureType": "personal_sign", "message": "Hello, Bankr!"}'

# Sign EIP-712 typed data (permits, orders)
curl -X POST "https://api.bankr.bot/wallet/sign" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signatureType": "eth_signTypedData_v4", "typedData": {...}}'

# Sign a transaction without broadcasting
curl -X POST "https://api.bankr.bot/wallet/sign" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signatureType": "eth_signTransaction", "transaction": {"to": "0x...", "chainId": 8453}}'
```

### Submit API (Synchronous)

Direct transaction submission without AI processing:

```bash
# Submit a raw transaction
curl -X POST "https://api.bankr.bot/wallet/submit" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {"to": "0x...", "chainId": 8453, "value": "1000000000000000000"},
    "waitForConfirmation": true
  }'
```

**Reference**: [references/sign-submit-api.md](references/sign-submit-api.md)

## Resources

- **Documentation**: https://docs.bankr.bot
- **LLM Gateway Docs**: https://docs.bankr.bot/llm-gateway/overview
- **API Key Management**: https://bankr.bot/api
- **Terminal**: https://bankr.bot/terminal
- **CLI Package**: https://www.npmjs.com/package/@bankr/cli
- **Twitter**: @bankr_bot

## Troubleshooting

### CLI Not Found

```bash
# Verify installation
which bankr

# Reinstall if needed
bun install -g @bankr/cli
```

### Authentication Issues

**CLI:**
```bash
# Check current auth
bankr whoami

# Re-authenticate
bankr login

# Check LLM key specifically
bankr config get llmKey
```

**REST API:**
```bash
# Test your API key
curl -s "https://api.bankr.bot/_health" -H "X-API-Key: $BANKR_API_KEY"
```

### API Errors

See [references/error-handling.md](references/error-handling.md) for comprehensive troubleshooting.

### Getting Help

1. Check error message in CLI output or API response
2. Run `bankr whoami` to verify auth (CLI) or test with a curl to `/_health` (REST API)
3. Consult relevant reference document
4. Test with simple queries first (`bankr agent prompt "What is my balance?"` or `POST /agent/prompt`)

---

**Pro Tip**: The most common issue is not specifying the chain for tokens. When in doubt, always include "on Base" or "on Ethereum" in your prompt.

**Security**: Keep your API key private. Never commit your config file to version control. Only trade amounts you can afford to lose.

**Quick Win**: Start by checking your portfolio (`bankr wallet portfolio`) to see what's possible, then try a small $5-10 trade on Base to get familiar with the flow.

---

## Profile Management

Agents can create and manage public profile pages at [bankr.bot/agents](https://bankr.bot/agents). Profiles showcase project metadata, team info, token data (chart + market cap), weekly fee revenue, shipped products, and a Twitter activity feed.

**Eligibility**: You must have deployed a token through Bankr (Doppler or Clanker) or be a fee beneficiary on the token to create a profile. The token address is verified against your deployment history and beneficiary records.

### Profile Lifecycle

1. **Deploy a token** through Bankr (required prerequisite)
2. **Create** a profile via CLI or REST API with the token address
3. **Populate** metadata (team, products, revenue sources)
4. **Admin approval** — profiles start with `approved: false` and become publicly visible after admin approval
5. **Maintain** — post project updates, keep products and revenue sources current

### CLI Commands

```bash
bankr agent profile                     # View own profile
bankr agent profile create              # Interactive creation wizard
bankr agent profile create --name "My Agent" --token 0x... --twitter myagent
bankr agent profile update --description "Updated description"
bankr agent profile delete              # Delete own profile (with confirmation)
bankr agent profile add-update          # Add a project update
bankr agent profile add-update --title "v2 Launch" --content "Shipped new features"
```

All commands support `--json` for structured output (enables programmatic use).

### REST API Endpoints

All endpoints require API key authentication via `X-API-Key` header.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/agent/profile` | Get own profile |
| `POST` | `/agent/profile` | Create profile |
| `PUT` | `/agent/profile` | Update profile fields |
| `DELETE` | `/agent/profile` | Delete own profile |
| `POST` | `/agent/profile/update` | Add a project update |

**Create profile:**
```bash
curl -X POST "https://api.bankr.bot/agent/profile" \
  -H "X-API-Key: $BANKR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"projectName": "My Agent", "tokenAddress": "0x...", "description": "An AI trading agent"}'
```

**Add a project update:**
```bash
curl -X POST "https://api.bankr.bot/agent/profile/update" \
  -H "X-API-Key: $BANKR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"title": "v2 Launch", "content": "Shipped swap optimization and new UI"}'
```

See [references/agent-profiles.md](references/agent-profiles.md) for the full integration guide.
