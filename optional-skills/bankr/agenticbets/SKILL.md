---
name: agenticbets
description: Place prediction bets on token prices on Base via AgenticBets. Use when the user wants to bet UP or DOWN on whether a token price will go up or down, check prediction market odds, view open betting rounds, or claim winnings from settled rounds. Supports all tokens with active markets on AgenticBets (AGBETS, CLAWD, MOLT, WCHAN, and more). Uses Bankr Submit API to execute bet and claim transactions on Base.
metadata:
  {
    "clawdbot":
      {
        "emoji": "🎲",
        "homepage": "https://agenticbets.dev",
        "requires": { "bins": ["python3", "bankr"] },
      },
  }
---

# AgenticBets

Prediction markets on Base. Bet UP or DOWN on token prices with USDC.

## When To Use

Use AgenticBets when the user wants to:
- **Bet on whether a token price goes up or down** (e.g., "bet $5 UP on AGBETS")
- **Check prediction market odds** (e.g., "what are the odds on AGBETS?")
- **See which markets are open** (e.g., "what prediction markets can I bet on?")
- **Claim winnings** from settled rounds
- **Check their bet status** or claimable winnings

## Prerequisites

### Bankr CLI

Install the Bankr CLI and log in to get an API key:

```bash
bun install -g @bankr/cli
# or: npm install -g @bankr/cli
```

### Bankr API Key

The scripts read the API key from `~/.bankr/config.json` (or `$BANKR_CONFIG` if set).

**Option A: CLI login (recommended)**

```bash
# Step 1 — send OTP
bankr login email user@example.com

# Step 2 — verify and generate key with write access
bankr login email user@example.com --code 123456 --accept-terms --key-name "AgenticBets" --read-write
```

**Option B: Web login**

1. Visit [bankr.bot/api](https://bankr.bot/api)
2. Sign in with email + OTP
3. Generate an API key with **Wallet API** write access enabled

The API key must have **write access** (`walletApiEnabled`, not `readOnly`) to place bets and claim winnings. Read-only keys can still list markets and check odds.

### USDC Balance

Make sure your Bankr wallet has USDC on Base before betting. Check with:

```bash
bankr wallet portfolio
```

## Quick Start

### List Open Markets

```
What prediction markets are open on AgenticBets?
```

```bash
scripts/agenticbets.py list
```

### Check Odds

```
What are the odds on AGBETS?
```

```bash
scripts/agenticbets.py odds AGBETS
```

### Place a Bet

```
Bet $5 UP on AGBETS
```

```bash
scripts/agenticbets.py bet AGBETS up 5
```

### Claim Winnings

```
Claim my AgenticBets winnings for AGBETS epoch 42
```

```bash
scripts/agenticbets.py claim AGBETS 42
```

## Script Usage

### agenticbets.py

Single script that handles all AgenticBets operations. Reads and writes use the Bankr Wallet API.

```
scripts/agenticbets.py <command> [args...]
```

**Commands:**

| Command | Args | Description |
|---|---|---|
| `list` | `[status]` | List markets. Status: `all`, `open`, `locked`, `settled` (default: `open`) |
| `odds` | `<symbol>` | Show bull/bear odds and pool size for a market |
| `info` | `<symbol>` | Detailed market info including contract, epoch, time to lock |
| `bet` | `<symbol> <up\|down> <amount>` | Place a bet. Amount in USDC (e.g., `5` for $5) |
| `claim` | `<symbol> <epoch> [epoch...]` | Claim winnings for settled epochs |
| `claimable` | `<symbol> <epoch>` | Check if an epoch is claimable |

**Environment:**

| Variable | Default | Description |
|---|---|---|
| `BANKR_CONFIG` | `~/.bankr/config.json` | Path to Bankr config file containing `apiKey` |

## How It Works

### Prediction Market Flow

1. A **round** opens for a token (e.g., $AGBETS)
2. Users bet **UP** (bull — price goes up) or **DOWN** (bear — price goes down) with USDC
3. Betting window closes (typically 5 minutes)
4. Price is **locked** at close
5. After the round duration, price is checked again
6. If price went up → bull wins. If down → bear wins.
7. Winners split the entire pool proportional to their bet size (minus 3% fee)

### Transaction Flow (via Bankr Submit API)

All on-chain transactions go through the Bankr Wallet API:

1. Script fetches market data from `GET https://agenticbets.dev/api/bankr/markets`
2. Script gets wallet address from `GET https://api.bankr.bot/wallet/me`
3. For bets:
   - Approve USDC spend: `POST https://api.bankr.bot/wallet/submit` with ERC20 `approve()` calldata
   - Place bet: `POST https://api.bankr.bot/wallet/submit` with `bet()` calldata
4. For claims: `POST https://api.bankr.bot/wallet/submit` with `claim()` calldata
5. All transactions use `waitForConfirmation: true` and include a human-readable `description`

### Bankr Submit API Reference

**Endpoint:** `POST https://api.bankr.bot/wallet/submit`

**Headers:**
```
X-API-Key: bk_YOUR_API_KEY
Content-Type: application/json
```

**Request body:**
```json
{
  "transaction": {
    "to": "0xContractAddress",
    "chainId": 8453,
    "data": "0xCalldata...",
    "value": "0"
  },
  "description": "Place $5 UP bet on AGBETS",
  "waitForConfirmation": true
}
```

**Success response:**
```json
{
  "success": true,
  "transactionHash": "0x...",
  "status": "success",
  "blockNumber": "12345678",
  "signer": "0xYourWalletAddress",
  "chainId": 8453
}
```

**Wallet info:** `GET https://api.bankr.bot/wallet/me` returns the wallet address and supported chains.

## Contracts

| Contract | Address | Tokens |
|---|---|---|
| BankrBetsPrediction V1 | `0xABADeb002247f2bd908Eeedb32918aEc304A0233` | CLAWD, MOLT, WCHAN |
| BankrBetsPrediction V2 | `0x2CD785Ba87e0841A8458141bc43d23a56a00557f` | AGBETS |
| USDC (Base) | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` | 6 decimals |

### Contract Functions

**bet(address _token, uint256 _amount, uint8 _position)**
- `_token` — token contract address
- `_amount` — USDC amount in raw units (multiply by 1e6)
- `_position` — `0` = Bull (UP), `1` = Bear (DOWN)
- Selector: `0x37a02e62`

**claim(address _token, uint256[] _epochs)**
- `_token` — token contract address
- `_epochs` — array of epoch numbers to claim
- Selector: `0x45718278`

**claimable(address _token, uint256 _epoch, address _user) -> bool**
- Returns true if the user has unclaimed winnings for that epoch
- Selector: `0xd3c035fc`

### USDC Approval

Before betting, the script approves USDC spend on the prediction contract:

```
ERC20.approve(predictionContractAddress, betAmount)
```
- Selector: `0x095ea7b3`

## Token Addresses

| Token | Address | Prediction Contract |
|---|---|---|
| AGBETS | `0x37d183FCf1DA460a64D21E754b3E6144C4e11BA3` | V2 (`0x2CD785...`) |
| All others | Varies (from API) | V1 (`0xABADeb...`) |

Use the `/api/bankr/markets` endpoint to get current token addresses — don't hardcode.

## Markets API

**GET** `https://agenticbets.dev/api/bankr/markets`

Returns:
```json
{
  "markets": [
    {
      "token": "0x37d183FCf1DA460a64D21E754b3E6144C4e11BA3",
      "symbol": "AGBETS",
      "marketUrl": "https://agenticbets.dev/market#...",
      "poolUsdc": 42.50,
      "bullPct": 61.0,
      "bearPct": 39.0,
      "lockTimestamp": 1713100000,
      "secondsToLock": 83,
      "predictionContract": "0x2CD785Ba87e0841A8458141bc43d23a56a00557f",
      "status": "open",
      "epoch": "5",
      "poolAddress": "0x...",
      "creator": "0x...",
      "createdAt": 1700000000,
      "contractVersion": "v2"
    }
  ],
  "count": 4,
  "updatedAt": "2026-04-14T12:00:00.000Z"
}
```

**Key fields:**
- `status` — `"open"` (accepting bets), `"locked"` (waiting for settlement), `"settled"` (done)
- `secondsToLock` — seconds until betting closes. `null` if not open.
- `poolUsdc` — total USDC in the round pool
- `bullPct` / `bearPct` — current odds split
- `predictionContract` — which contract to call for this token

## Strategy Tips

- **Check odds before betting.** If 90% of the pool is on one side, the other side pays ~10x if it wins.
- **Pool size matters.** Larger pools = more reliable odds. Small pools can swing wildly with one bet.
- **Time your bet.** Betting late (< 30s to lock) lets you see the final odds but risks missing the window.
- **Minimum bet is $1 USDC.** No maximum.
- **Check balance first.** Make sure you have enough USDC on Base before betting (`bankr wallet portfolio`).

## Examples

### User: "What prediction markets can I bet on?"

```bash
scripts/agenticbets.py list open
```

Response format:
```
Open Markets:
  $AGBETS — Pool: $42.50 | UP: 61% DOWN: 39% | Closes in 83s
  $CLAWD — Pool: $18.00 | UP: 50% DOWN: 50% | Closes in 210s
```

### User: "Bet $10 on AGBETS going up"

```bash
scripts/agenticbets.py bet AGBETS up 10
```

Steps:
1. Fetch market data → get token address + prediction contract
2. Approve USDC via Bankr Submit: `ERC20.approve(0x2CD785..., 10000000)`
3. Place bet via Bankr Submit: `BankrBetsPrediction.bet(0x37d183..., 10000000, 0)`
4. Return tx hashes

### User: "Do I have any claimable winnings?"

```bash
scripts/agenticbets.py claimable AGBETS 5
```

Calls `claimable(token, epoch, walletAddress)` — returns true/false.

### User: "Claim my AGBETS winnings from epoch 5"

```bash
scripts/agenticbets.py claim AGBETS 5
```

Steps:
1. Call `claim(0x37d183..., [5])` via Bankr Submit
2. Return tx hash

## References

- **[references/agent-usage.md](references/agent-usage.md)** — Full agent guide: intent mapping, example conversations, pre-flight checks, error recovery, Markets API schema, and contract reference (calldata encoding, fees, round lifecycle)

## Resources

- **Website:** https://agenticbets.dev
- **Telegram Alerts:** https://t.me/agenticbets
- **GitHub:** https://github.com/viraj124/agentic-bets
- **MCP Server:** `npx agenticbets-mcp` (for Claude/Cursor users)
