# Cat Town Protocol Documentation

AI-readable reference for Cat Town — a Farcaster-native game world on Base. This doc is a higher-level companion to [SKILL.md](SKILL.md); for agent integration flows (trigger phrases, tx building, troubleshooting), start there.

- Game: https://cat.town
- Staking UI (Wealth & Whiskers Bank): https://cat.town/bank
- Public user docs: https://docs.cat.town
- Chain: Base mainnet (8453)

---

## Current coverage

This skill currently documents nine Cat Town surfaces:

1. **KIBBLE staking** — the RevenueShare contract, stake/claim/unlock/unstake flows, staking leaderboard, user deposit history.
2. **World state** — the GameData contract, current season/time-of-day/weather/weekend.
3. **Fishing drops** — the public item-truth catalog with world-state-conditioned drops (weather, season, time of day), plus the frontend's exact fishing filter.
4. **Fishing competition** — weekly Sat–Mon competition with live prize-pool math (10/80/10 split) and top-10 leaderboard.
5. **Fish raffle** — Paulie's weekly Fri 20:00 UTC draw with tier-based prize pool, chance-to-win math, free-ticket claim flow, and leaderboard.
6. **Boutique** — daily 3-item onchain shop with seasonal pools, plus the KIBBLE→USD price oracle.
7. **Gacha** — pay tx + async VRF-mint pattern with token-id-ordering result polling, daily limit, USD-denominated pricing, seasonal pool filter.
8. **Selling items** — vendor flow for Treasures + Collectibles minted by the V2 minter, with catalog-value math (cents → KIBBLE via oracle) and a 5% tax.
9. **KIBBLE tokenomics** — Jasper's answers: % staked, % burned, live staking APY.

Future revisions will add the boutique purchase flow, the paid-ticket (fish-burn) raffle path, legacy (V1-minter) sell support, seasonal events via `/v1/seasonal/*`, daily rewards, and the community-pot surface Jasper also touches. Each will land under its own `references/<feature>/` subdirectory.

---

## KIBBLE token

| Property       | Value                                                       |
|----------------|-------------------------------------------------------------|
| Address (Base) | `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`                |
| Decimals       | 18                                                          |
| Total supply   | 1,000,000,000 (fixed)                                       |
| Burn sink      | `0x000000000000000000000000000000000000dEaD`                |

**Deflationary mechanic:** 2.5% of every fish identified is burned. The burn balance is already substantial (~66% at the last read). When quoting any "% of supply" stat, always compute circulating = `totalSupply − balanceOf(0xdEaD)`, not just `totalSupply`. SKILL.md's "KIBBLE circulating supply" section has the full formula and a concrete example.

---

## KIBBLE staking (RevenueShare)

| Property       | Value                                                       |
|----------------|-------------------------------------------------------------|
| Contract       | `0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1` on Base        |
| Pattern        | Masterchef-style single-pool pro-rata                       |
| Reward token   | KIBBLE (in, out)                                            |
| Revenue sources | fishing (weekly, ≤12:00 UTC Mon), gacha (weekly, ≤12:00 UTC Wed) |
| Unlock wait    | **14 days** (`LOCK_PERIOD` = 1,209,600s, snapshotted per-user at `unlock()`) |

### The non-obvious bit — unit convention

`stake(uint256)` and `unstake(uint256)` take **integer KIBBLE** (not wei). The contract multiplies by `10^18` internally before calling `transferFrom`. `approve()` on KIBBLE is standard ERC-20 and takes wei. This asymmetry is the single most common integration failure on this contract — see the **⚠️ CRITICAL** section at the top of [SKILL.md](SKILL.md) before building any stake calldata.

### Exit flow

1. `unlock()` — sets `isUnlocking[user] = true`, writes `unlockEndTime[user] = now + LOCK_PERIOD`. User's stake is removed from `totalActiveStaked` → **pool share drops to 0%** → they stop accruing rewards. Tell the user this up front.
2. Wait until `block.timestamp >= unlockEndTime(user)` — 14 days.
3. `unstake(N)` — reverts if called before the wait ends. `N` is integer KIBBLE.

`relock()` cancels the wait and returns the user to the active pool at any point.

### Reads (all in integer KIBBLE, not wei)

- `getUserStaked(user)` — currently staked
- `pendingRewards(user)` — claimable right now
- `getPoolShareFraction(user) / 1e18 * 100` — user's current pool %
- `isUnlocking(user)` + `unlockEndTime(user)` — exit status + ETA
- `getTotalStaked()` / `getTotalActiveStaked()` — pool sizes

Full reference: [references/staking/contract.md](references/staking/contract.md).

### Offchain API (public, unauthenticated)

- `GET https://api.cat.town/v2/revenue/staking/leaderboard` — top stakers with rank + pool-share %.
- `GET https://api.cat.town/v2/revenue/deposits/{address}` — one user's historical fishing/gacha deposits with per-tx share.

Response shapes + field meanings: [references/staking/api.md](references/staking/api.md).

---

## World state (GameData)

| Property       | Value                                                       |
|----------------|-------------------------------------------------------------|
| Contract       | `0x298c0d412b95c8fc9a23FEA1E4d07A69CA3E7C34` on Base        |
| Reads          | `getGameState()` returns `(season, timeOfDay, isWeekend, worldEvent, weather)` |
| Granularity    | Live, updates continuously; reads are cheap                  |

Enums: Season `0..3` (Spring/Summer/Autumn/Winter); TimeOfDay string (`"Morning"`, `"Daytime"`, `"Evening"`, `"Nighttime"`); Weather `0..6` (None/Sun/Rain/Wind/Storm/Snow/Heatwave).

World state drives fishing and gacha drop tables (different fish appear in different weather/seasons). Fishing drop tables are documented in [references/fishing/drops.md](references/fishing/drops.md); gacha pools are covered in the gacha section below.

Full function table, selectors, live sample: [references/world/contract.md](references/world/contract.md).

---

## Fishing drops (item catalog + world filter)

| Property       | Value                                                       |
|----------------|-------------------------------------------------------------|
| Endpoint       | `GET https://api.cat.town/v2/items/master?limit=1000`       |
| Auth           | Public (no headers)                                         |
| Catalog size   | ~430 active items                                           |
| Caching        | Safe to cache ≥1 hour (frontend caches indefinitely)        |

Each item carries optional `dropConditions` keyed by `events`, `seasons`, `timesOfDay`, `weathers`. The frontend's fishing filter keeps only `source=Fishing` + `itemType ∈ {Fish, Treasure}`, matches the requested axis, drops event-exclusive items unless the event is active, and sorts by rarity DESC → name ASC. This returns items **exclusive** to that axis value (e.g. 3 snow-only drops, not 400+ weather-agnostic items).

Weather changes most frequently (minutes-to-hours), so weather-exclusive drops are the most rotational — highest-value thing to surface.

Full recipe + live weather→drops table: [references/fishing/drops.md](references/fishing/drops.md).

---

## Boutique (daily rotation, onchain)

| Property        | Value                                                       |
|-----------------|-------------------------------------------------------------|
| Contract        | `0xf9843bF01ae7EF5203fc49C39E4868C7D0ca7a02` on Base        |
| KIBBLE Oracle   | `0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7`                |
| Rotation cycle  | Every 00:00 UTC, 3 items from current season's pool         |
| Primary read    | `getTodaysRotationDetails() → ShopItemView[]`               |
| Pricing         | `price` in KIBBLE wei; convert to USD via oracle            |

Seasonal doc pages (public): [shops/boutique](https://docs.cat.town/shops/boutique), [spring-fashion](https://docs.cat.town/boutique/spring-fashion), [summer-fashion](https://docs.cat.town/boutique/summer-fashion), [autumn-fashion](https://docs.cat.town/boutique/autumn-fashion), [winter-fashion](https://docs.cat.town/boutique/winter-fashion).

**USD conversion:** `getKibbleUsdPrice()` returns USD-per-KIBBLE scaled by **`10^18`** (note: `getEthUsdPrice()` on the same contract uses `10^8`). Formula: `usd = (price_wei * rawKibbleUsdPrice) / 10^36`. Live at time of writing: ~$0.0009487 per KIBBLE.

Full reference: [references/boutique/contract.md](references/boutique/contract.md).

---

## Fishing competition (weekly, Sat 00:00 UTC → Mon 00:00 UTC)

| Property       | Value                                                       |
|----------------|-------------------------------------------------------------|
| Contract       | `0x62a8F851AEB7d333e07445E59457eD150CEE2B7a` on Base        |
| Leaderboard API | `GET https://api.cat.town/v1/fishing/competition/leaderboard` (public) |
| Cycle          | Saturday 00:00 UTC → Monday 00:00 UTC (48h window)          |
| Host NPC       | Isabella                                                    |

Prize-pool math (mirroring the frontend exactly):

```
prizePool                             // API field: total volume of KIBBLE spent on fish IDs during comp
leaderboardShare = prizePool * 0.10   // top-10 prize pool, further split 30/20/10/8/8/7/5/4/4/4
treasureShare    = prizePool * 0.80   // returned to fishers as treasures
stakersRevenue   = prizePool * 0.10   // flows to KIBBLE stakers via RevenueShare
```

When active: lead with running time / weather / participants / prize pool / top 10. When inactive: compute next Saturday 00:00 UTC, offer a reminder, and offer to narrate the last completed competition (the API returns it when `isActive=false`).

Full reference: [references/fishing/competition.md](references/fishing/competition.md).

---

## Fish raffle (weekly, Fri 20:00 UTC draw)

| Property           | Value                                                       |
|--------------------|-------------------------------------------------------------|
| FishRaffle         | `0x5E183eBc7CA4dF353170C35b4D69Ea9f42317b28` on Base        |
| FreeToPlayPool     | `0x131E680dc7A146F00b282FBD7d6261c5B38c4Fa6` on Base        |
| Leaderboard API    | `GET https://api.cat.town/v1/tickets/leaderboard` (public)  |
| Winners API        | `GET https://api.cat.town/v1/tickets/winners` (public)      |
| Cycle              | Mon 00:00 UTC open → Fri 19:50 UTC close → Fri 20:00 UTC draw |
| Winners per draw   | 5 (equal split of the prize pool)                           |
| Free ticket        | 1 per wallet per ISO week; `canClaimFreeTicket(user)` → `claimFreeTicket()` |

Prize pool is `poolBalance * tier.bps / 10000` where the tier is picked by the round's `totalTickets`:

| minTickets | bps |
|-----------:|----:|
| 0 / 250 / 500 / 850 / 1,400 / 2,200 / 3,500 / 5,500 | 30 / 40 / 50 / 60 / 70 / 80 / 90 / 100 |

Chance-to-win approximation: `min(1, 5 * userTickets / totalTickets)`. Live at time of writing: round 31, 2,855 tickets sold, 80-bps tier, ~47,742 KIBBLE pool → ~9,548 KIBBLE per winner.

Full reference: [references/fish-raffle/contract.md](references/fish-raffle/contract.md), [references/fish-raffle/api.md](references/fish-raffle/api.md).

---

## Gacha (async VRF pulls, pay now → receive later)

| Property       | Value                                                         |
|----------------|---------------------------------------------------------------|
| Contract       | `0xAD0ee945B4Eba7FB8eB7540370672E97eB951F1a` on Base          |
| Daily cap      | 100 pulls / wallet / UTC day                                  |
| Cost           | `capsulePriceUSD()` in US cents (live: **$0.50** ≈ 527 KIBBLE) |
| Result poll    | `GET https://api.cat.town/v2/items/capsule/<user>` (public)   |
| Batch model    | N sequential `purchaseAndOpenCapsule()` txs (no onchain batch)|
| Result detection | Token-id ordering: capture `latestId` pre-pull, poll for items with `id > latestId` |

The pay tx submits a VRF randomness request; the NFT mints in a separate tx seconds later. Agents must either poll the capsule API for new items (if Bankr supports async polling) or return "ask me again in ~30 s" and re-check later. For multi-pulls, wait until `count(newItems) >= N` before reporting.

Every pull is uniformly weighted against the current season's pool — no pity, no streaks. Full pattern + oracle math + 500-on-cold-wallet quirk: [references/gacha/contract.md](references/gacha/contract.md), [references/gacha/api.md](references/gacha/api.md).

---

## Selling items (V2 minter, vendor)

| Property       | Value                                                         |
|----------------|---------------------------------------------------------------|
| SellItems      | `0x49936db5Dcbc906D682CFa2dcfAb0788e3ee5808` on Base          |
| V2 minter      | `0x7b65ec82cB4600Bc1dCc5124a15594976f19eA14` (only supported source) |
| Payout token   | KIBBLE                                                        |
| Tax            | **5%** (`taxRateInBps() = 500`)                               |
| Batch cap      | 25 items per tx (frontend gate; no onchain cap)               |
| Inventory API  | `GET https://api.cat.town/v2/inventory/<address>/paginated?hasSellValue=true` (public) |

Sellable types: Treasure + Collectible only (Cosmetics/Fish/Equipment aren't sellable here). `sellValue` in the item catalog is in **US cents**; convert to KIBBLE for display via the oracle, then apply the 5% tax for payout.

Write call (single, batched): `sellMultipleNFTsToContract(address[] nftContracts, uint256[] tokenIds, uint256[] amounts)`. Requires `setApprovalForAll(sellContract, true)` on the V2 minter, once per wallet.

Full reference: [references/sell-items/contract.md](references/sell-items/contract.md).

---

## KIBBLE tokenomics

| Property   | Value                                                              |
|------------|--------------------------------------------------------------------|
| Inputs     | `balanceOf(0xdEaD)`, `RevenueShare.getTotalStaked()`, baronbot's 30-day revenue-share history |
| % burned   | `balanceOf(0xdEaD) / totalSupply` × 100 (live: ~66%)               |
| % staked   | `totalStaked / (totalSupply − burned)` × 100 (live: ~24%)          |
| Staking APY | Derived from baronbot's 30-day deposits + stake (live: ~30%)       |

Mirrors Jasper's NPC answers in the Wealth & Whiskers Bank. The % burned uses total supply as the denominator; % staked uses circulating (total − burned). APY is dynamic and uncapped until 1000% APY / 50% monthly rate sanity limits.

Full reference: [references/kibble/tokenomics.md](references/kibble/tokenomics.md).

---

## Weekly cadence

Cat Town runs on a fixed weekly UTC cycle. Only the **bold** rows directly affect staking rewards; the others feed other surfaces (raffle, competition, boutique) that this skill already covers.

| Day       | Event                             | Time                       | Host              |
|-----------|-----------------------------------|----------------------------|-------------------|
| Monday    | **Fishing revenue deposit**       | by 12:00 (often earlier)   | Theodore / Cassie |
| Mon–Fri   | Fish raffle ticket sales open     | —                          | Paulie            |
| Mon–Fri   | Weekday fishing                   | —                          | Skipper           |
| Wednesday | **Gacha revenue deposit**         | by 12:00 (often earlier)   | Theodore / Cassie |
| Friday    | **Fish raffle draw**              | 20:00                      | Paulie            |
| Sat–Sun   | **Weekly fishing competition**    | Sat morning → Sun night    | Isabella          |

Full calendar with revenue-split details and NPC cheat-sheet: [references/world/calendar.md](references/world/calendar.md).

---

## Not covered yet

Cat Town's codebase exposes additional public surfaces that are out of scope for the current revision: gacha spins + the async VRF receive pattern, the boutique purchase flow (approve + `purchaseItem`), the paid fish-raffle path (burning 20 kg of caught fish per ticket via `buyTickets`), seasonal events via `/v1/seasonal/*`, daily rewards, and the community pot Jasper references. When they're added, each will land in a new `references/<feature>/` subdirectory without disturbing existing integrations.
