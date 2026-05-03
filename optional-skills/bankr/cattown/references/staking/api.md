# Cat Town Staking API

Two public JSON endpoints served from `https://api.cat.town`. **No authentication** — plain GET, no headers required.

Use these when you need leaderboard or historical-deposit data without paying RPC costs. For live position reads (current stake, pending rewards, unlock state), go onchain — see [contract.md](contract.md).

---

## `GET /v2/revenue/staking/leaderboard`

Ranked list of all stakers with stake amount and pool share.

### Response

```json
{
  "success": true,
  "leaderboard": [
    {
      "user_address": "0x...",
      "staked_amount": "123456789000000000000",
      "share_percentage": "12.34",
      "active": true,
      "rank": 1,
      "basename": "someuser.base.eth"
    }
  ],
  "summary": {
    "total_stakers": 1234,
    "total_staked": "9876543210000000000000",
    "total_active_staked": "8765432100000000000000"
  }
}
```

### Field meanings

| Field                            | Type    | Notes                                                          |
|----------------------------------|---------|----------------------------------------------------------------|
| `leaderboard[].user_address`     | hex     | Staker's wallet.                                               |
| `leaderboard[].staked_amount`    | string  | KIBBLE amount in **wei** (18 decimals). Parse as BigInt / big-number. |
| `leaderboard[].share_percentage` | string  | Percentage of the active pool, as a decimal string (e.g. `"12.34"` = 12.34%). |
| `leaderboard[].active`           | bool    | `false` if the user is currently in an `unlock()` window (not earning). |
| `leaderboard[].rank`             | number  | 1-indexed rank. Sort by this.                                  |
| `leaderboard[].basename`         | string? | Optional Basename (human-readable handle).                     |
| `summary.total_stakers`          | number  | Distinct staker count.                                         |
| `summary.total_staked`           | string  | All KIBBLE in the contract, wei (includes unlocking users).    |
| `summary.total_active_staked`    | string  | KIBBLE currently earning rewards, wei.                         |

### Caching

Cat Town's frontend requests with `Cache-Control: no-cache, no-store, must-revalidate`. Treat this as near-live but not instant. For a second-accurate balance, use the onchain `getUserStaked` / `getTotalActiveStaked` reads.

---

## `GET /v2/revenue/deposits/{address}`

One user's history of fishing / gacha deposits, plus a summary.

### Response

```json
{
  "success": true,
  "deposits": [
    {
      "transaction_hash": "0x...",
      "block_number": 12345678,
      "source": "fishing",
      "deposit_timestamp": "2026-04-20T11:58:04.000Z",
      "deposit_amount": "500000000000000000000",
      "user_share_amount": "123456789000000000",
      "new_acc_reward_per_share": 123456,
      "created_at": "2026-04-20T11:58:10.000Z",
      "total_active_staked": "9876543210000000000000"
    }
  ],
  "summary": {
    "total_revenue": "5000000000000000000000",
    "total_user_share": "50000000000000000000",
    "user_wallet": "0x...",
    "current_total_staked": 9876543210000000000000,
    "current_total_active_staked": 8765432100000000000000
  }
}
```

### Field meanings

| Field                                     | Type    | Notes                                                          |
|-------------------------------------------|---------|----------------------------------------------------------------|
| `deposits[].transaction_hash`             | hex     | The `depositRevenue` tx hash — use as a stable key.            |
| `deposits[].block_number`                 | number  | Block the deposit landed in.                                   |
| `deposits[].source`                       | string  | `"fishing"`, `"gacha"`, or `"daycare"` (daycare not yet active). Case-insensitive; lowercase is canonical. |
| `deposits[].deposit_timestamp`            | ISO8601 | When the `RevenueDeposited` event fired.                       |
| `deposits[].deposit_amount`               | string  | Total KIBBLE added to the pool for this deposit, in wei.       |
| `deposits[].user_share_amount`            | string  | The share this specific user earned from this deposit, in wei. Zero if the user was unlocking or not staked at the time. |
| `deposits[].new_acc_reward_per_share`     | number  | Value of `accRewardPerShare` right after this deposit.         |
| `deposits[].total_active_staked`          | string  | Pool size (wei) at the moment of deposit — lets you verify `user_share_amount` math. |
| `summary.total_revenue`                   | string  | Sum of `deposit_amount` across all returned deposits, wei.     |
| `summary.total_user_share`                | string  | Sum of `user_share_amount` across all returned deposits, wei. This is the user's all-time earnings from staking. |
| `summary.current_total_staked`            | number  | Current pool total, raw (may drop precision; prefer onchain read for exact value). |
| `summary.current_total_active_staked`     | number  | Current active pool, raw (same precision caveat).              |

### Deduplication

If the same `transaction_hash` appears twice (can happen due to backend replays), prefer the row with non-zero `user_share_amount`.

### Sources

Per `RevenueShareSource` enum in the Cat Town frontend:
- `fishing` — weekly, deposited by 12:00 UTC Monday.
- `gacha` — weekly, deposited by 12:00 UTC Wednesday.
- `daycare` — not yet implemented; may appear in future responses.

---

## Typical agent use cases

- **"Where do I rank?"** → `GET /v2/revenue/staking/leaderboard`, find the user by `user_address`, report `rank` and `share_percentage`.
- **"How much have I earned?"** → `GET /v2/revenue/deposits/{address}`, return `summary.total_user_share` (wei → ÷ 1e18 for display).
- **"When was the last drop?"** → `GET /v2/revenue/deposits/{address}`, take `deposits[0].deposit_timestamp` (API returns newest-first; confirm before relying on it).
- **"Show me last month's fishing vs gacha split."** → filter `deposits` by `source` and sum `user_share_amount`.
