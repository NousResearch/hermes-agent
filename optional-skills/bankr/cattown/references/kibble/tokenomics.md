# KIBBLE Tokenomics

Cat Town's in-game NPC **Jasper** (Wealth & Whiskers Bank) surfaces the tokenomics users routinely ask about: % staked, % burned, and the current staking APY. This doc mirrors Jasper's exact math so Bankr answers with the same numbers the game shows.

Player-facing overview: https://docs.cat.town/economy/tokens/kibble, https://docs.cat.town/get-started/kibble-economy.

Source (in the Cat Town frontend): `hooks/interactions/Jasper/useJasperBankInteractions.tsx` and `hooks/game/staking/useStakingEstimatedAPY.tsx`.

## Fixed inputs

| Value                 | Source                                         | Notes                  |
|-----------------------|------------------------------------------------|------------------------|
| Total supply          | 1,000,000,000 KIBBLE                           | Hardcoded (18 decimals)|
| KIBBLE token          | `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`   | Base mainnet           |
| Burn sink             | `0x000000000000000000000000000000000000dEaD`   | Deflationary target    |
| RevenueShare contract | `0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1`   | Base mainnet           |
| Kibble price oracle   | `0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7`   | For USD conversions    |
| baronbot (APY ref)    | `0x8Ff7AcCCf73c515c1f62Fc7b64A63F17Ce99659e`   | rank-1 continuous staker |

## % Burned

```
totalBurned   = balanceOf(0xdEaD) on KIBBLE token   // in wei, then /1e18
burnedPercent = (totalBurned / totalSupply) * 100   // denominator is TOTAL supply
```

Note: the denominator here is **total supply** (1B), not circulating. That's how Jasper phrases it in the dialogue ("X% of the total supply, gone forever").

**Live at time of writing:** 663,398,178.50 KIBBLE burned → **66.34%** of total supply.

## % Staked

```
circulatingSupply = totalSupply − totalBurned
totalStaked       = RevenueShare.getTotalStaked()        // returns WHOLE KIBBLE (not wei) — see SKILL.md CRITICAL note
stakedPercent     = (totalStaked / circulatingSupply) * 100
```

Denominator is **circulating** supply (total − burned). This gives a realistic number — dividing by total would understate by ~3× because so much has been burned.

**Live at time of writing:** 80,826,993 KIBBLE staked ÷ 336,601,822 circulating → **24.01%** of circulating KIBBLE is staked.

## Staking APY at Wealth & Whiskers

The game doesn't publish a fixed APY; it's derived dynamically from the last 30 days of revenue deposits. The frontend uses **baronbot** (rank-1 continuous staker) as the reference because the KIBBLE return per staked KIBBLE is the same for every active staker.

### Formula (verbatim from `useStakingEstimatedAPY.tsx`)

```
1. Pull baronbot's last 30 days of deposits:
   GET https://api.cat.town/v2/revenue/deposits/0x8Ff7AcCCf73c515c1f62Fc7b64A63F17Ce99659e
   Keep deposits where deposit_timestamp >= now − 30 days.

2. period_revenue    = sum(deposit.user_share_amount over those 30 days)

3. Annualise from baronbot's staked amount (from the staking leaderboard API):
   earliest_deposit_date = min(deposit.date over those 30 days)
   days_since_first      = max(1, (now − earliest_deposit_date) in days)
   monthly_revenue       = period_revenue * (30 / days_since_first)
   monthly_rate          = monthly_revenue / baronbot.stakedAmount

4. Compound with caps:
   capped_monthly = min(monthly_rate, 0.50)             // 50%/month ceiling
   apy            = (1 + capped_monthly)^12 − 1
   apy_percent    = min(apy * 100, 1000)                // 1000% APY ceiling
```

Both caps are sanity guards — under normal revenue flow, neither hits. The monthly rate is proportional per KIBBLE, so the same APY applies to every active (non-unlocking) staker.

### Live worked example (at time of writing)

- baronbot staked: **8,000,000 KIBBLE** (rank 1, 10.01% of active pool)
- baronbot 30-day revenue: **171,957 KIBBLE** across 9 deposits
- Earliest deposit in window: 29.09 days ago
- monthly_revenue = 171,957 × (30 / 29.09) = **177,325 KIBBLE**
- monthly_rate = 177,325 / 8,000,000 = **2.217%**
- APY = (1.02217)^12 − 1 = **~30.1%**

So Wealth & Whiskers currently pays **~30% APY** to KIBBLE stakers. Expect it to drift with weekly fishing + gacha revenue — not a fixed rate.

## Recipe — "tell me about KIBBLE"

When a user asks about KIBBLE tokenomics, lead with the three headline numbers — burned, staked, APY — computed live. Brief example copy:

> **KIBBLE tokenomics (live):**
>
> - **~66.3% burned** (663.4M / 1B sent to `0xdEaD` via the 2.5%-per-fish mechanic)
> - **~24.0% of circulating KIBBLE is staked** in Wealth & Whiskers (80.8M KIBBLE in RevenueShare)
> - **Staking APY ~30%** — driven by weekly fishing + gacha revenue, not a fixed rate
>
> Want to stake some yourself? The lock period is 14 days.

Offer a natural follow-up: "Want me to walk you through staking?" or "Want the current top stakers?" (→ staking leaderboard).

## Other tokenomics Jasper surfaces (not yet wired here)

- **Community Pot** — KIBBLE held for task-completion rewards. Jasper quotes it as a % of circulating. Not yet covered in this skill.
- **Price / market cap** — Jasper doesn't quote prices. For USD conversion use the Kibble Price Oracle (`getKibbleUsdPrice()`) — see [../boutique/contract.md](../boutique/contract.md) for the exact formula (1e18-scaled, not Chainlink 1e8).
