# Agent Usage Guide

How an AI agent should use the AgenticBets skill — plus the full API and contract reference.

## Core Principles

1. **Always fetch live market data first** — never hardcode token addresses, pool sizes, or odds.
2. **Check the user's intent before acting** — reads are free, writes cost USDC + gas.
3. **Confirm destructive actions** — always echo back what you're about to do before `bet` or `claim`.
4. **Report tx hashes** — after any write, surface the Basescan link so the user can verify.

## Intent → Command Mapping

| User intent | Script command | Example |
|---|---|---|
| "What markets are open?" | `list open` | `agenticbets.py list open` |
| "Show me all markets" | `list all` | `agenticbets.py list all` |
| "What are the odds on X?" | `odds <symbol>` | `agenticbets.py odds AGBETS` |
| "Tell me about the X market" | `info <symbol>` | `agenticbets.py info AGBETS` |
| "Bet $N UP on X" | `bet <symbol> up <amount>` | `agenticbets.py bet AGBETS up 5` |
| "Bet $N DOWN on X" | `bet <symbol> down <amount>` | `agenticbets.py bet AGBETS down 5` |
| "Claim my winnings from X epoch N" | `claim <symbol> <epoch>` | `agenticbets.py claim AGBETS 42` |
| "Can I claim X epoch N?" | `claimable <symbol> <epoch>` | `agenticbets.py claimable AGBETS 42` |

## Example Conversations

### Example 1 — Discovery → Bet

**User:** "What prediction markets should I look at?"

**Agent:**
1. Run `scripts/agenticbets.py list open`
2. Parse output, identify markets with `poolUsdc >= 5` and imbalanced odds (e.g., one side > 70%)
3. Respond with ranked options:

> "Here are the open markets:
> - **$AGBETS** — Pool: $42.50, UP: 61% / DOWN: 39%, closes in 83s
> - **$CLAWD** — Pool: $18.00, 50/50 split, closes in 210s
>
> $AGBETS has the biggest pool and leans bullish. If you think the price will drop, DOWN pays ~2.5x. Want me to place a bet?"

### Example 2 — Explicit Bet

**User:** "Bet $5 UP on AGBETS"

**Agent:**
1. Run `scripts/agenticbets.py odds AGBETS` to confirm market is open
2. Confirm with user: "Placing $5 UP on $AGBETS at current 61/39 odds. Proceed?"
3. On approval, run `scripts/agenticbets.py bet AGBETS up 5`
4. Report both tx hashes (approve + bet) with Basescan links

### Example 3 — Claim Flow

**User:** "Did I win on AGBETS epoch 5?"

**Agent:**
1. Run `scripts/agenticbets.py claimable AGBETS 5`
2. If claimable, ask: "You have winnings on epoch 5. Claim now?"
3. On approval, run `scripts/agenticbets.py claim AGBETS 5`

### Example 4 — Strategy Question

**User:** "Is it a good idea to bet on 90/10 splits?"

**Agent:** (no script call needed — this is general strategy)

> "When odds are 90/10, the 10% side pays roughly 9x if it wins — but those odds usually reflect real signal. Good reasons to bet the underdog:
> - The pool is small (one big bet can swing it back)
> - You have specific info contradicting the consensus
> - You're near lock time and can see final odds
>
> Against the underdog:
> - Winners side's whales usually have better info
> - Fees eat into small pools
>
> Rule of thumb: only bet the underdog if pool < $20 OR you have an informational edge."

## Pre-flight Checks Before Writes

Before placing a bet, an agent should:

1. **Verify Bankr config exists:**
   ```bash
   test -f ~/.bankr/config.json && echo "OK" || echo "Run: bankr login email <email>"
   ```

2. **Verify write access** — the script will return 403 if key is read-only. On 403, instruct user to regenerate key with `--read-write` or toggle off "Read-only" at bankr.bot/api.

3. **Verify ETH balance for gas** — `/wallet/submit` broadcasts raw txs, so the Bankr wallet needs native ETH on Base. If the tx fails with "insufficient funds for gas," surface this clearly.

4. **Verify USDC balance** — `bankr wallet portfolio` shows Base USDC. Compare to bet amount + existing approvals.

## Error Recovery

| Error | Cause | Fix |
|---|---|---|
| `Read-only API key` | API key lacks write permission | Regenerate with `--read-write` |
| `insufficient funds for gas` | Bankr wallet has no ETH on Base | Send ETH to the wallet address |
| `ERC20: transfer amount exceeds allowance` | Approve tx not yet indexed, or spender mismatch | Retry once; if persistent, check V1 vs V2 routing |
| `BelowMinBet` | Bet < 1 USDC | Raise amount to at least $1 |
| `TokenNotEligible` | Market is inactive or paused | Check `info <symbol>` to confirm status |
| `No market found for 'X'` | Symbol typo or market doesn't exist | Run `list all` to see all symbols |

## What NOT to Do

- **Don't loop bets automatically** without user approval each time
- **Don't claim on every settled epoch** without checking `claimable` first — wasted gas if nothing to claim
- **Don't hardcode AGBETS → V2 routing** in new code — read `predictionContract` from the Markets API so future V3/V4 work without changes
- **Don't retry failed txs more than once** — if the second attempt fails with the same error, stop and ask the user
- **Don't bet within 10 seconds of lock** unless the user explicitly asks — high risk the tx misses the window

---

# Markets API Reference

Read-only API for fetching current prediction market state. No auth required.

## GET /api/bankr/markets

Base URL: `https://agenticbets.dev`

Returns all prediction markets with live odds, pool sizes, timing, and contract info.

### Request

```bash
curl https://agenticbets.dev/api/bankr/markets
```

No parameters. No headers required (JSON is returned by default).

### Response

```json
{
  "markets": [
    {
      "token": "0x37d183FCf1DA460a64D21E754b3E6144C4e11BA3",
      "symbol": "AGBETS",
      "marketUrl": "https://agenticbets.dev/market#AGBETS",
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

### Fields

| Field | Type | Description |
|---|---|---|
| `token` | address | ERC20 token contract address (the asset being bet on) |
| `symbol` | string | Token ticker (e.g., "AGBETS", "CLAWD") |
| `marketUrl` | string | Direct link to place bets via the web UI |
| `poolUsdc` | number | Current USDC pool size for this round |
| `bullPct` | number | % of pool on UP (0–100) |
| `bearPct` | number | % of pool on DOWN (0–100) |
| `lockTimestamp` | number | Unix seconds when betting closes |
| `secondsToLock` | number \| null | Seconds until lock. `null` if round isn't open. |
| `predictionContract` | address | Which prediction contract to call for this token |
| `status` | string | `"open"` \| `"locked"` \| `"settled"` \| `"cancelled"` \| `"not_started"` |
| `epoch` | string | Current round number for this token |
| `poolAddress` | address | Uniswap V4 pool used for price oracle |
| `creator` | address | Wallet that created this market |
| `createdAt` | number | Unix seconds of market creation |
| `contractVersion` | string | `"v1"` (CLAWD, MOLT, WCHAN) or `"v2"` (AGBETS) |

### Status Values

| Status | Meaning | Can Bet? |
|---|---|---|
| `open` | Betting window active | Yes |
| `locked` | Bets closed, waiting for settlement | No |
| `settled` | Round finished, winners can claim | No (a new round will start on next bet) |
| `cancelled` | Round cancelled (tie or error) — refunds | No |
| `not_started` | Market exists but no round has started yet | Yes (first bet starts the round) |

### Notes

- Data is cached for ~10 seconds server-side. Don't poll faster than that.
- `bullPct + bearPct` always sums to 100.0.
- Token addresses are canonical — never hardcode; always fetch from this API.

---

# Contract Reference

Binary prediction market contracts on Base. Users bet UP or DOWN on token prices using USDC.

## Deployed Contracts

| Contract | Address | Version | Tokens |
|---|---|---|---|
| BankrBetsPrediction V1 | `0xABADeb002247f2bd908Eeedb32918aEc304A0233` | v1 | CLAWD, MOLT, WCHAN, etc. |
| BankrBetsPrediction V2 | `0x2CD785Ba87e0841A8458141bc43d23a56a00557f` | v2 | AGBETS |
| USDC (Base) | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` | — | 6 decimals |

Pick the right contract: use `predictionContract` field from the Markets API, or route AGBETS→V2 and everything else→V1.

## Write Functions

### bet(address _token, uint256 _amount, uint8 _position)

Place a bet on a token's price direction. Requires prior USDC approval.

**Selector:** `0x37a02e62`

**Parameters:**
- `_token` — ERC20 token to bet on (e.g., AGBETS address)
- `_amount` — USDC amount in raw units (multiply by `1e6`). Minimum: `1_000_000` (1 USDC)
- `_position` — `0` = Bull (UP), `1` = Bear (DOWN)

**Behavior:**
- Transfers `_amount` USDC from caller to contract via `safeTransferFrom`
- Starts a new round if current round is settled/cancelled
- Reverts if token is not active, amount below minimum, round is locked, or pool cap exceeded

**Calldata encoding:**
```
0x37a02e62
+ padLeft32(token)      // 32 bytes
+ padLeft32(amount)     // 32 bytes
+ padLeft32(position)   // 32 bytes
```

### claim(address _token, uint256[] _epochs)

Claim winnings for one or more settled epochs.

**Selector:** `0x45718278`

**Parameters:**
- `_token` — token the bet was placed on
- `_epochs` — array of epoch numbers to claim

**Behavior:**
- Iterates epochs, transfers USDC winnings to caller
- Skips already-claimed epochs
- Reverts if any epoch is not claimable

**Calldata encoding (dynamic array):**
```
0x45718278
+ padLeft32(token)                        // 32 bytes
+ padLeft32(0x40)                         // offset to array (64)
+ padLeft32(epochs.length)                // array length
+ padLeft32(epoch[0])                     // first element
+ ...
```

## Read Functions

### claimable(address _token, uint256 _epoch, address _user) → bool

Returns true if `_user` has unclaimed winnings for `_token` at `_epoch`.

**Selector:** `0xd3c035fc`

## USDC Approval

Before `bet()`, approve the prediction contract to spend USDC:

```
ERC20.approve(predictionContract, amount)
```

**Selector:** `0x095ea7b3`

**Calldata:**
```
0x095ea7b3
+ padLeft32(predictionContract)   // spender
+ padLeft32(amount)                // uint256 amount in raw units
```

## Fees

Total fee: **2.1%** of losers' pool, split:
- 1.5% → treasury
- 0.5% → market creator
- 0.1% → settler (whoever calls `closeRound()`)

Winners split the remaining 97.9% pro-rata based on their bet size vs. winning-side total.

## Round Lifecycle

```
Open → Locked → Settled
         ↑         ↓
  (betWindow)   (next bet starts new round)
```

1. **Open** — `startTimestamp` to `lockTimestamp - 60s`. Bets accepted.
2. **Locked** — `lockRound()` called → `lockPrice` recorded
3. **Settled** — `closeRound()` called at `closeTimestamp` → winner determined
4. Next `bet()` call on this token starts a new round

## Constraints

- **Minimum bet:** `1_000_000` raw units = 1 USDC. Below this reverts with `BelowMinBet()`.
- **Max round pool (V2 only):** 10,000 USDC per round. Exceeding reverts with `ExceedsMaxRoundPool()`.
