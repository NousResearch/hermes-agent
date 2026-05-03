---
name: cattown
description: Interact with Cat Town — a Farcaster-native game world on Base. Covers KIBBLE staking (stake, claim, unlock, unstake, leaderboard, deposit history); live world state (season, weather, time of day, weekend flag); fishing drops filtered by world state; Isabella's weekend fishing competition with live prize-pool math; Paulie's weekly fish raffle (free-ticket claim, tier-based prize pool, odds, last winners); the daily 3-item boutique with KIBBLE→USD conversion; gacha spins (async VRF pay-then-receive, 100/day cap, seasonal pools); item valuation plus batch selling via the V2 vendor (5% tax); and KIBBLE tokenomics (% burned, % staked, live APY). Use when the user mentions Cat Town, KIBBLE, Wealth & Whiskers, Jasper, Isabella, Paulie, Skipper, Theodore, Cassie, RevenueShare, fishing, gacha, raffle, boutique, vendor, prize pools, drop tables, or any read/write on the Cat Town contracts.
---

# Cat Town — Agent Overview

Cat Town is a Farcaster-native game world on Base. Players fish, collect, and earn KIBBLE; a share of town revenue is streamed weekly to KIBBLE stakers. This skill lets agents read Cat Town state and submit the transactions needed to participate.

The town's NPCs run each activity and are worth naming when talking to players:

- **Wealth & Whiskers Bank** — where KIBBLE staking happens. **Theodore** works the day shift, **Cassie** takes over in the evening.
- **Paulie** — runs the weekly fish raffle.
- **Skipper** — the weekday fishing NPC.
- **Isabella** — hosts the weekend fishing competition.

Current coverage:

- **KIBBLE staking** (RevenueShare) — stake, claim, claim-and-restake, unlock, relock, unstake, plus staking leaderboard and deposit history.
- **World state** (GameData) — live season, time of day, weather, weekend flag.
- **Fishing drops** — the public item-truth catalog filtered by world state (weather/season/time).
- **Fishing competition** (Isabella, Sat–Mon) — live prize-pool math, top-10 leaderboard, active/inactive response patterns.
- **Fish raffle** (Paulie, Fri 20:00 UTC draw) — free-ticket claim flow, tier-based prize pool, chance-to-win, leaderboard + last winners.
- **Boutique** — daily 3-item onchain rotation with KIBBLE→USD conversion via the Kibble Price Oracle.
- **KIBBLE tokenomics** — Jasper's math for % burned, % staked, and live staking APY.

Each surface has its own subdirectory under `references/` for the deep reference. The weekly calendar below is the shared timing reference — many sections link back to it.

Links:
- Game: https://cat.town
- Bank (staking UI): https://cat.town/bank
- Docs: https://docs.cat.town

---

## Weekly Calendar (all times UTC)

Cat Town runs on a fixed weekly cadence. Use these timings when setting user expectations ("your next fishing drop is Monday") or scheduling follow-ups.

| Day       | Event                             | Time                       | Host              | Affects staking rewards? |
|-----------|-----------------------------------|----------------------------|-------------------|--------------------------|
| Monday    | **Fishing revenue deposit**       | by 12:00 (often earlier)   | Theodore / Cassie | Yes                      |
| Mon–Fri   | Fish raffle ticket sales open     | —                          | Paulie            | No                       |
| Mon–Fri   | Weekday fishing                   | —                          | Skipper           | No                       |
| Wednesday | **Gacha revenue deposit**         | by 12:00 (often earlier)   | Theodore / Cassie | Yes                      |
| Friday    | **Fish raffle draw**              | 20:00                      | Paulie            | No                       |
| Sat–Sun   | **Weekly fishing competition**    | Sat morning → Sun night    | Isabella          | Indirect*                |

*During the weekend fishing competition (Sat–Sun), 10% of every fish identification feeds the KIBBLE stakers pool. Weekday fishing (Skipper) does **not** feed stakers. This is why weekend activity sizes the following Monday's fishing-revenue deposit. See [references/world/calendar.md](references/world/calendar.md) for the full revenue split.

Deposits are triggered by the Cat Town backend calling `depositRevenue(amount, source)` on RevenueShare, with `source` in `"fishing"` or `"gacha"`. Watch the `RevenueDeposited(string source, uint256 depositTimestamp, uint256 depositAmount, uint256 newAccRewardPerShare)` event to know the exact moment a drop lands.

---

## ⚠️ CRITICAL: stake/unstake take INTEGER KIBBLE, not wei — read this first

This contract breaks the 99%-of-ERC-20 default. If you reflexively wei-encode the stake/unstake amount because "that's how tokens work," the tx reverts. This is the single failure mode you will hit on this contract. Stop here and internalize the table before touching any write path.

| Call                                      | Amount unit               | Example for 1 KIBBLE              |
|-------------------------------------------|---------------------------|-----------------------------------|
| `kibble.approve(revenueShare, …)`         | **wei** (standard ERC-20) | `1000000000000000000` (= 1 × 10¹⁸) |
| `revenueShare.stake(uint256 amount)`      | **integer KIBBLE**        | `1`                               |
| `revenueShare.unstake(uint256 amount)`    | **integer KIBBLE**        | `1`                               |

Reads are also integer KIBBLE: `getUserStaked`, `pendingRewards`, `getTotalStaked`, `getTotalActiveStaked`.

### Raw calldata — right vs. wrong

```
✅ stake(1)     → 0xa694fc3a0000000000000000000000000000000000000000000000000000000000000001
❌ stake(1e18)  → 0xa694fc3a0000000000000000000000000000000000000000000000000de0b6b3a7640000
```

The second form reverts with `ERC20: transfer amount exceeds balance` because the contract multiplies your argument by `10^18` internally, turning `1e18` into `1e36` wei. Verified via simulation against the deployed contract on Base.

### Pre-submit validation (run this before every stake/unstake)

- Is the amount `< 1,000,000`? → probably correct (integer KIBBLE).
- Is the amount `≥ 10^15`? → almost certainly wrong — you wei-encoded by reflex.
- Sanity check: `stake(1)` = 1 KIBBLE. `stake(100)` = 100 KIBBLE. `stake(10000)` = 10,000 KIBBLE.
- `approve` is the OPPOSITE — it is wei. Staking `N` KIBBLE requires `approve(revenueShare, N * 10^18)`.

**Signer = holder.** The address that signs `stake` must be the same address that holds the KIBBLE and signed the `approve`.

---

## KIBBLE Staking

### Addresses (Base, chain id 8453)

- **RevenueShare**: `0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1`
- **KIBBLE token (ERC-20, 18 decimals)**: `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`

Base Sepolia addresses and the full ABI surface are in [references/staking/contract.md](references/staking/contract.md). User-facing overview: https://docs.cat.town/economy/staking.

### Core flows

Single pool, single reward token — KIBBLE in, KIBBLE out. No reward-token selection, no per-user lock duration, no multipliers. One global `accRewardPerShare` accumulator updated on each `depositRevenue`.

**1. Stake** (mixed units — re-read the CRITICAL section above if uncertain)
1. `kibble.approve(revenueShare, amount_wei)` — `amount_wei = N * 10^18` where `N` is the KIBBLE count. Required once if `allowance(user, revenueShare) < amount_wei`.
2. `revenueShare.stake(uint256 N)` — **`N` is the integer KIBBLE count, NOT wei.** If this reverts with `ERC20: transfer amount exceeds balance`, you wei-encoded — pass the plain integer instead. Emits `Staked(user, amount)`.

**2. Claim** (after each fishing/gacha deposit)
- `revenueShare.claim()` — transfers `pendingRewards(user)` to the user. Emits `Claimed(user, amount)`.
- `revenueShare.claimAndRestake()` — claims and auto-adds to the user's stake in one tx. Emits `ClaimedAndRestaked(user, restakedAmount, totalStakedNow)`.

**3. Exit (unlock → wait → unstake)**
1. `revenueShare.unlock()` — emits `UnlockInitiated(user, unlockEndTime)`. Sets `isUnlocking[user] = true`. **Always tell the user two things when they unlock:** (a) the wait is **14 days** (`LOCK_PERIOD` = 1,209,600 seconds, snapshotted so later changes don't affect them), and (b) their pool share just dropped from whatever-it-was to **0%** — they won't earn fishing or gacha deposits during the wait. Read the pre-unlock share first via `getPoolShareFraction(user) / 1e18 * 100`.
2. Wait until `block.timestamp >= unlockEndTime(user)`. The 14-day value is safe to quote at the point of unlock. Read `LOCK_PERIOD()` live only if you want defensive protection against future upgrades (the contract is UUPS-upgradeable).
3. `revenueShare.unstake(uint256 N)` — **`N` is the integer KIBBLE count, same convention as stake.** Reverts before the wait ends. Emits `Unstaked(user, amount)`.
- `revenueShare.relock()` — at any time during the wait, cancels the unlock and puts the user back into the earning pool. Emits `Relocked(user, amount)`.

#### Checking remaining unlock time

When a user asks "how long until I can withdraw?", compute from `unlockEndTime(user)`:

```
remaining_seconds = max(0, unlockEndTime(user) - current_unix_time)
```

- `isUnlocking(user) == false` → not unlocking, nothing to wait on.
- `remaining_seconds > 0` → still waiting. Convert to days/hours for the reply.
- `remaining_seconds == 0` → `unstake(N)` is callable now.

No dedicated contract method for "time left" — just subtract. Use the latest block's timestamp if you want to avoid clock-skew with the user's device.

#### Mapping "unstake" / "withdraw" / "exit" to the right call

Users say "unstake" colloquially to mean the whole exit, not literally the onchain `unstake()` function. Before acting, read three values: `isUnlocking(user)`, `unlockEndTime(user)`, and the user's current pool share (`getPoolShareFraction(user) / 1e18 * 100` as a percentage). Then route:

| State                                                    | What to call | What to tell the user                                                                 |
|----------------------------------------------------------|--------------|---------------------------------------------------------------------------------------|
| `isUnlocking(user) == false`                             | `unlock()`   | "Started your unlock. Wait is **14 days** — ready at `<unlockEndTime>`. Your pool share just dropped from **Y%** to **0%**; you won't earn revenue deposits during the wait. Call `relock()` any time to cancel and restore your share." |
| `isUnlocking == true` **and** `now < unlockEndTime`      | *(no tx)*    | "Already unlocking. ~X days Y hours left until you can withdraw. Your pool share is **0%** until you either `unstake()` after the wait or `relock()` now."                                                  |
| `isUnlocking == true` **and** `now >= unlockEndTime`     | `unstake(N)` | "Withdrew N KIBBLE." (Or remaining balance if partial.)                                |

Same routing for "withdraw," "exit," "pull my KIBBLE out," "get my stake back." **Never call the onchain `unstake()` as the first step** — it reverts unless the user has already completed an unlock wait.

Why the share drop matters: while `isUnlocking == true`, the user's stake is **removed from `totalActiveStaked`**, so they do not earn any fishing or gacha revenue deposits that land during the 14-day wait. Surfacing the pre-unlock share (Y%) makes the opportunity cost explicit.

### Unlock state machine — the gotcha to warn users about

```
 [staking, earning] ──unlock()──▶ [unlocking, NOT earning] ──wait LOCK_PERIOD──▶ [unstake available]
         ▲                               │                                              │
         └────────── relock() ───────────┘                                              │
         │                                                                              │
         └────────────────────── unstake(amount) ◀──────────────────────────────────────┘
```

While `isUnlocking[user] == true`:
- The user's balance is in `totalStaked()` but **excluded from `totalActiveStaked()`**.
- Reward math divides by `totalActiveStaked`, so the user **does not accrue rewards** during the unlock window.
- `unstake()` reverts until `unlockEndTime(user)` has passed.

Recommend users `claim()` any pending rewards first, then `unlock()`, then `unstake()` once the wait is over. If they change their mind mid-wait, `relock()` is free and returns them to the earning pool instantly.

### Reading a user's position

KIBBLE-denominated reads return **whole KIBBLE** (not wei). See the Amount units section above.

| Call                               | Returns / unit                                     | Meaning                                                            |
|------------------------------------|----------------------------------------------------|--------------------------------------------------------------------|
| `getUserStaked(address)`           | whole KIBBLE                                       | Currently staked KIBBLE                                            |
| `pendingRewards(address)`          | whole KIBBLE                                       | Claimable KIBBLE right now                                         |
| `isUnlocking(address)`             | `bool`                                             | True if user has called `unlock()` and not yet unstaked/relocked   |
| `unlockStartTime(address)`         | unix seconds                                       | When `unlock()` was called                                         |
| `unlockEndTime(address)`           | unix seconds                                       | When `unstake()` becomes callable                                  |
| `getPoolShareFraction(address)`    | fraction × 1e18                                    | User's share of the active pool                                    |
| `getTotalActiveStaked()`           | whole KIBBLE                                       | Total KIBBLE earning rewards right now                             |
| `getTotalStaked()`                 | whole KIBBLE                                       | Total KIBBLE in the contract (includes unlocking users)            |
| `LOCK_PERIOD()`                    | seconds                                            | Unlock wait duration                                               |
| `accRewardPerShare()`              | accumulator × 1e18                                 | Global reward accumulator                                          |

Full function-by-function reference: [references/staking/contract.md](references/staking/contract.md).

### KIBBLE circulating supply — always subtract the burn address

When quoting "what % of KIBBLE is staked" (or any % of supply), compute against **circulating supply**, not `totalSupply`. KIBBLE has a deflationary burn mechanic: 2.5% of every fish identified is sent to `0x000000000000000000000000000000000000dEaD`, and this compounds. The burned portion is already substantial — dividing by `totalSupply` materially undercounts the staked share (typically by ~3×).

```
totalSupply       = 1,000,000,000 KIBBLE                         // fixed, read via totalSupply() on KIBBLE
burned            = balanceOf(0x000000000000000000000000000000000000dEaD) on KIBBLE   // read live
circulating       = totalSupply − burned
percentStaked     = getTotalStaked() / circulating × 100          // reads are whole-KIBBLE integers
```

Representative recent values (re-read live — the burn keeps growing):

- `totalSupply` ≈ 1,000,000,000 KIBBLE
- `balanceOf(0xdEaD)` ≈ 663M KIBBLE burned (~66%)
- circulating ≈ 337M KIBBLE
- `getTotalStaked()` ≈ 81M KIBBLE → **~24% of circulating KIBBLE is staked**

`balanceOf(0x0)` on KIBBLE is `0`; the protocol burns to `0xdEaD` only. If you must be exhaustive, check both, but `0xdEaD` is where the number lives.

### Staking leaderboard & user deposit history

Two public JSON endpoints on `https://api.cat.town`, **no auth required**. Use these whenever the user wants their rank, their share of the pool, or their weekly earnings history without paying RPC costs.

- `GET /v2/revenue/staking/leaderboard` — ranked stakers with stake amount and pool-share %.
- `GET /v2/revenue/deposits/{address}` — one user's historical `fishing` / `gacha` deposits, per-tx amounts, and the share that landed for that user.

Full shapes, field meanings, and example responses: [references/staking/api.md](references/staking/api.md).

---

## World state

Cat Town's live world state (season, time of day, weather, weekend flag) lives on a single onchain contract — **GameData** at `0x298c0d412b95c8fc9a23FEA1E4d07A69CA3E7C34` on Base. Fully read-only from an agent's perspective.

The one call you usually want is **`getGameState()`** → `(season, timeOfDay, isWeekend, worldEvent, weather)`. One RPC, every field:

- **Season** (`uint8`): `0=Spring`, `1=Summer`, `2=Autumn`, `3=Winter`
- **TimeOfDay** (`string`): `"Morning"`, `"Daytime"`, `"Evening"`, `"Nighttime"`
- **Weather** (`uint8`): `0=None`, `1=Sun`, `2=Rain`, `3=Wind`, `4=Storm`, `5=Snow`, `6=Heatwave`
- **isWeekend** (`bool`): true on Sat/Sun UTC (the fishing-competition window)
- **worldEvent** (`uint8`): event code — detailed event decoding is out of scope for this skill revision

World state drives fishing and gacha drop tables — different fish appear in different weather/seasons. Fishing drop tables are documented in the **Fishing drops** section below; gacha pools are planned for a future revision.

Full function table, selectors, raw calldata, live sample response, and historical-lookup fns (`getSeasonForDate`, `getWeatherForDate`): [references/world/contract.md](references/world/contract.md).

For the fixed weekly cadence (fishing/gacha revenue deposits, Paulie's raffle, Isabella's weekend competition), see [references/world/calendar.md](references/world/calendar.md).

---

## Fishing drops — "what can I catch in this weather?"

When a user asks "what's catchable in the rain?", "what's exclusive to Winter evenings?", or "what drops in a Storm?", combine live world state (from GameData above) with Cat Town's public item catalog:

```
GET https://api.cat.town/v2/items/master?limit=1000        // public, no auth
```

Each item has optional `dropConditions: { events?, seasons?, timesOfDay?, weathers? }`. The frontend's fishing filter (ported verbatim from `utils/helpers/fishingHelpers.tsx`) is four steps:

1. Keep only `isActive == true`, `source == "Fishing"`, `itemType ∈ {"Fish", "Treasure"}`.
2. Match the user-asked axis — `weathers` / `seasons` / `timesOfDay` — including `axis_value` in the item's condition array.
3. Drop items that require a seasonal event unless that event is currently active (Halloween items are invisible outside Halloween).
4. Sort by rarity DESC, then name ASC.

**This returns items *exclusive* to that axis value.** `getFishingItemsForWeather("Snow")` → 3 snow-only drops, not the 400+ weather-agnostic items. That matches how the frontend surfaces "special drops this weather."

**Enum mismatch to normalize:** GameData contract returns `timeOfDay` as `"Daytime"` / `"Nighttime"`; the item API uses `"Afternoon"` / `"Night"`. Weather and season strings match (after lowercasing).

Live example — weather=Storm: Misty Duck (Rare), Lovely Duck (Rare), King Snapper (Rare Fish), **Elusive Marlin (Legendary Fish)**. Weather is the most rotational axis (minutes-to-hours), so weather-exclusive drops are the highest-value thing to surface to a user deciding *when* to fish.

### Response pattern — lead with big-ticket specials, then offer the standard drops

When a user asks "what can I catch today / right now?", listing only the 3-4 axis-exclusive items feels incomplete. Answer in two tiers. Within each tier, lead with the **big-ticket items** so the reply opens with the most interesting catches.

#### Big-ticket sort (apply to every list you surface)

1. **Rarity DESC** — Legendary → Epic → Rare → Uncommon → Common. This is the primary signal for fish (weight data isn't in the item API — see below).
2. **`sellValue` DESC** — useful tiebreaker within a rarity. `sellValue` is in **cents USD** (not KIBBLE). Real examples from the catalog: Legendary time-of-day rings (Solar, Dawnbreak, Moonlight, Twilight) sell at 25,000¢ = $250; Diamond and Frozen Tusk at 10,000¢ = $100; Gilded Sundial at 5,500¢.
3. Name ASC as final tiebreaker.

#### Two tiers

1. **Lead with the special drops** — weather-exclusive and timeOfDay-exclusive items for the current state. Sort with the big-ticket order above — the Legendary goes first in the reply, not last.
2. **Count the "standard drops" also catchable today** — items with NO `weathers` and NO `timesOfDay` conditions, whose season + event gates still pass. Baseline is ~26 per season.
3. **Offer the deep dive.** End with a prompt like: *"There are X other standard drops you can also catch today — want me to list them?"*

Concrete filter for standard drops (note: "standard" = not rotating on weather/time, as opposed to "special" — has nothing to do with the `Common` *rarity* tier):

```
standard_drops(current_season, current_event):
  for item in catalog:
    require item.isActive
    require item.source == "Fishing"
    require item.itemType in {"Fish", "Treasure"}
    require item.dropConditions has no `weathers` array
    require item.dropConditions has no `timesOfDay` array
    if item.dropConditions.seasons is set:
      require current_season in item.dropConditions.seasons
    if item.dropConditions.events is set:
      require current_event in item.dropConditions.events
```

Example reply for Storm / Spring / no active event — note Legendary leads:

> Storm weather right now brings out 4 special drops, headed by **Elusive Marlin** (Legendary Fish). The rest: **King Snapper** (Rare Fish), **Misty Duck** (Rare Treasure), **Lovely Duck** (Rare Treasure).
>
> You can also catch **~26 other standard Spring drops** today, led by **Alligator Gar** (Legendary Fish), **Diamond** ($100 Epic Treasure), and **Jade Figurine** ($40 Epic). Want me to list the rest?

#### Fish weight data — not in the API, cross-reference the public docs

Per-species fish weight ranges are **not** returned by `/v2/items/master`. If a user asks about the heaviest fish or typical weights, cross-reference Cat Town's public docs (unauthenticated, human-readable):

- Fish weights + conditions: https://docs.cat.town/items/fishing/fish
- Treasure details: https://docs.cat.town/items/fishing/treasures

For quick programmatic answers, lean on rarity + `sellValue`. For "what's the biggest {species}", point the user at those docs pages.

Full recipe, complete weather→drops table, and live-sweep counts: [references/fishing/drops.md](references/fishing/drops.md). Player-facing context: https://docs.cat.town/fishing/start-fishing, https://docs.cat.town/fishing/hot-streaks, https://docs.cat.town/fishing/upgrades.

---

## Fishing competition (weekly, Isabella hosts)

The **FishingCompetition** contract at `0x62a8F851AEB7d333e07445E59457eD150CEE2B7a` (Base) runs a weekly competition **Saturday 00:00 UTC → Monday 00:00 UTC**. When a user asks about it, **lead with live data, not with generic rules.** The skeleton differs based on whether one is currently running.

### Is one running?

- Onchain: `isCompetitionActive()` → `(bool active, bytes32 eventId)`
- API (public, no auth): `competition.isActive` in `GET https://api.cat.town/v1/fishing/competition/leaderboard`

### Prize-pool math (mirror the frontend exactly)

The API's `prizePool` is **total volume** (all KIBBLE spent identifying fish during the competition). The frontend splits it three ways:

```
prizePool                             // total volume
leaderboardShare = prizePool * 0.10   // top-10 prize pool ("Prize Pool" in UI)
treasureShare    = prizePool * 0.80   // treasures returned to fishers
stakersRevenue   = prizePool * 0.10   // flows to KIBBLE stakers via RevenueShare
```

Top-10 distribution of `leaderboardShare` (from `fishingLeaderboardShareForRank`): **30%, 20%, 10%, 8%, 8%, 7%, 5%, 4%, 4%, 4%**. `Math.floor` to whole KIBBLE.

### If active — response pattern

Pull the API response once, then pick 3–5 of these to feature (keep it conversational, don't dump everything):

- **Running time** — `now - startTime` ("14 hours in, 34 hours to go")
- **Weather** — from `GameData.getGameState()` (drives which special fish appear)
- **Participants** — `totalPlayers`
- **Leaderboard prize pool** — `prizePool * 0.10` in KIBBLE **+ USD conversion** via the oracle
- **Treasures returned to players** — `prizePool * 0.80`
- **Stakers revenue generated** — `prizePool * 0.10`
- **Top 10** — rank, basename (or short addr), fishName (+ shiny flag), weight in kg, expected payout

### If NOT active — response pattern

1. Say it clearly: "No fishing competition is running right now."
2. Compute next start = **next Saturday 00:00 UTC**; express as "starts in X days Y hours."
3. **Offer a reminder**: "Want me to ping you when it kicks off?"
4. **Ask the follow-up**: "Do you want to hear about last week's competition?"
5. If the user says yes, the same API response (even when inactive) carries the most recent completed competition — narrate winner, top-3 prizes, total volume, total participants.

### Example reply — inactive with offer

> There's no fishing competition running right now. The next one starts **Saturday 00:00 UTC — about 2 days 14 hours away**.
>
> Want me to ping you when it kicks off? I can also tell you about **last weekend's competition** — 100 fishers, **3.06M KIBBLE total volume**, and **bitcoinbov.base.eth won** with a 46.36 kg Elusive Marlin (~91,700 KIBBLE, ~$87).

### Example reply — active, leading with live data

> Fishing competition is live — **12 hours in, 36 hours to go**. Weather's **Storm** 🌧️ (Elusive Marlin's biting).
>
> - **27 fishers** competing
> - **Leaderboard prize pool**: ~41,200 KIBBLE (~$39) — 1st takes 30% (~12,360 KIBBLE)
> - Currently leading: **alice.base.eth** with a 42.8 kg Alligator Gar
> - Also generating **~33k KIBBLE for KIBBLE stakers** and **~264k returned to fishers as treasures** this weekend
>
> Want the full top 10?

Full ABI surface, per-rank payout worked example at current oracle rate, and the complete leaderboard response shape: [references/fishing/competition.md](references/fishing/competition.md). Player-facing overview: https://docs.cat.town/fishing/weekly-competition.

---

## Boutique — daily 3-item shop

The boutique is a fully onchain daily shop. Every day at **00:00 UTC** the Boutique contract surfaces **3 items** deterministically selected from the current season's pool. No offchain API — all state is readable directly on Base.

### Addresses

- **Boutique**: `0xf9843bF01ae7EF5203fc49C39E4868C7D0ca7a02`
- **Kibble Price Oracle** (for USD conversion): `0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7`

### Primary read — `getTodaysRotationDetails()`

Single call returns today's 3 items as `ShopItemView[]`. Each item carries `price` (in KIBBLE **wei**, divide by `10^18`), `stockRemaining`, `maxSupply`, `isPurchasableNow`, and a `traitNames`/`traitValues` parallel pair that encodes Name, Rarity, Slot, Image. Parse those into a dict to render.

### KIBBLE → USD conversion (the game UI doesn't do this — we should)

The in-game boutique shows KIBBLE prices only. To give users a USD readout, read the Kibble Price Oracle:

- `getKibbleUsdPrice()` → `uint256` USD per 1 KIBBLE, scaled by **`10^18`** (not 1e8 — **don't confuse with `getEthUsdPrice()` which is `10^8` Chainlink style**).
- Formula: `usd_value = (price_wei * rawKibbleUsdPrice) / 10^36`
- Live example: raw = `948,723,424,083,878` → **$0.0009487 per KIBBLE** → 10,000 KIBBLE ≈ **$9.49**.

### Response pattern — "what's in the boutique today?"

1. Parallel reads: `getTodaysRotationDetails()` + `getKibbleUsdPrice()`.
2. For each of the 3 items: parse the trait arrays (Name/Rarity/Slot), compute KIBBLE and USD price, check stock.
3. Sort big-ticket first — **rarity DESC** (Legendary → Common), then **KIBBLE price DESC**, then name ASC.
4. Flag `stockRemaining == 0` as "Sold Out"; otherwise format as `"{stockRemaining} of {maxSupply} remaining"` — **stockRemaining first, maxSupply second**. Sanity check: if your first number is larger than the second, you've swapped them — reread the struct fields. `stockRemaining` can never exceed `maxSupply`.
5. Open the reply with the current season; close with the matching `docs.cat.town/boutique/…-fashion` link for fuller context.

The collection name (e.g. `"Spring Fashion"`) is on the item itself as the **`Collection`** trait — surface it at the top of the reply so the user knows which collection is currently rotating.

Example reply (real data from today's rotation) — note the **"N of M remaining"** phrasing:

> **Boutique today — Spring Fashion collection:**
>
> 1. **White Longsleeve** — Rare Body — **12,500 KIBBLE (~$11.86)** — 1 of 1 remaining
> 2. **Royal Blue Varsity** — Uncommon Body — **6,000 KIBBLE (~$5.69)** — 2 of 2 remaining
> 3. **Classic Academic Blouse** — Uncommon Body — **6,000 KIBBLE (~$5.69)** — 1 of 2 remaining
>
> Browse the other seasonal collections:
> - Spring: https://docs.cat.town/boutique/spring-fashion
> - Summer: https://docs.cat.town/boutique/summer-fashion
> - Autumn: https://docs.cat.town/boutique/autumn-fashion
> - Winter: https://docs.cat.town/boutique/winter-fashion
> - Overview: https://docs.cat.town/shops/boutique

Include all four season links in every response — a user interested in the current collection will often want to peek at others.

Full ABI surface, trait schema (real keys: `Item Name`, `Rarity`, `Item Type`, `Source`, `Slot`, `Sprite`, `imageUrl`, `Collection`, etc.), preview future rotations, and the complete oracle math: [references/boutique/contract.md](references/boutique/contract.md).

**Purchase flow is out of scope for this revision** — this skill currently reads the boutique only.

---

## Paulie's fish raffle (weekly, Fri 20:00 UTC draw)

**FishRaffle** at `0x5E183eBc7CA4dF353170C35b4D69Ea9f42317b28` (Base). Weekly ISO-week rounds: tickets sell Mon 00:00 UTC → Fri ~19:50 UTC, 5 winners drawn Fri 20:00 UTC via Chainlink VRF. Paid tickets burn 20 kg of caught fish each. Every wallet gets 1 **free ticket per ISO week**.

A second contract, **FreeToPlayPool** at `0x131E680dc7A146F00b282FBD7d6261c5B38c4Fa6`, holds the prize pool balance and the tier table.

### Claim the weekly free ticket

Preflight read, then write:

```
canClaimFreeTicket(address user) → bool      // gate check
claimFreeTicket()                             // no args, msg.sender inferred
```

Always call `canClaimFreeTicket(user)` first. If false, surface the reason:

- Already claimed this week → "You've already claimed your free ticket this week. Next one resets **Monday 00:00 UTC**."
- Sales closed → "Sales closed at Fri 19:50 UTC. Winners draw at 20:00 UTC."
- Paused → "The raffle is paused."

Emits `FreeTicketClaimed(user, roundId)` on success. No token approval needed.

### Current state — lead with live numbers

Read in parallel:

- `currentRoundId()`, `currentISOWeek()`, `paused()`, `salesClosed()`
- `FreeToPlayPool.poolBalance()` + `FreeToPlayPool.getTiers()`
- `GET https://api.cat.town/v1/tickets/leaderboard` (public, no auth) — provides `totalTickets` and the top buyers

### Prize pool math (tier-based, not linear)

The prize pool is a fraction of `poolBalance`, set by the tier the round's `totalTickets` crosses into. **5 winners get an equal split.**

```
tier          = tiers.findLast(t => totalTickets >= t.minTickets)   // highest threshold crossed
prize_pool    = poolBalance * tier.bps / 10000                      // in KIBBLE wei
per_winner    = prize_pool / winnersPerDraw                         // equal split, NOT ranked
```

Live tier table:

| minTickets | bps | % of pool |
|-----------:|----:|----------:|
| 0          | 30  | 0.30%     |
| 250        | 40  | 0.40%     |
| 500        | 50  | 0.50%     |
| 850        | 60  | 0.60%     |
| 1,400      | 70  | 0.70%     |
| 2,200      | 80  | 0.80%     |
| 3,500      | 90  | 0.90%     |
| 5,500      | 100 | 1.00%     |

Live example (captured during writing): 2,855 tickets → 80 bps tier → `5,967,812 × 0.008 = ~47,742 KIBBLE` prize pool → **~9,548 KIBBLE per winner** (~$9). Last week's draw confirmed equal payouts of 9,363.86 KIBBLE to each of the 5 winners.

### Chance to win

Use the proportional approximation in replies (accurate enough for small ticket counts):

```
chance ≈ min(1, winnersPerDraw * userTickets / totalTickets)
```

Exact form (C = binomial coefficient): `1 − C(totalTickets − userTickets, 5) / C(totalTickets, 5)`. Use the exact form only if the user asks for precision.

For a single free-ticket claimant in a 2,855-ticket round: `5 * 1 / 2855 ≈ 0.175%`. For the current leader with 399 tickets: `~70%`.

### Leaderboard + last winners

- `GET https://api.cat.town/v1/tickets/leaderboard` — current-round `{ roundId, totalTickets, leaderboard[] }` with per-buyer `totalCount`, basename, equipment.
- `GET https://api.cat.town/v1/tickets/winners` — most recent completed draw with `roundId`, `timestamp`, 5 winners (all with the **same** `prizeAmount`, despite the `rank` field).

### Response pattern — "tell me about the fish raffle"

1. `canClaimFreeTicket(user)` — so you can close with a relevant CTA.
2. Pull leaderboard + pool balance + current tier.
3. Lead with live prize pool and per-winner split, then participant count, then top 3.
4. Close with the user's status: "You've got N tickets → ~X% chance" or "Claim your free ticket? I can do it now."

Example reply (live state):

> **Paulie's raffle is open** — round 31, draws **Friday 20:00 UTC (about 2 days out)**.
>
> - **~47,742 KIBBLE prize pool** (~$45), split equally among 5 winners → **~9,548 KIBBLE each**
> - **2,855 tickets** sold across **204 fishers**; 645 more tickets unlock the 90-bps tier
> - Top 3: `0xef05…` (399 tickets, ~70% chance), **bitcoinbov.base.eth** (364, ~64%), `0xdc6a…` (310, ~54%)
>
> You haven't claimed your free ticket this week — want me to grab it?

Full ABI surface, write paths, tier math, live-worked chance calcs: [references/fish-raffle/contract.md](references/fish-raffle/contract.md). API response shapes: [references/fish-raffle/api.md](references/fish-raffle/api.md). Player-facing overview: https://docs.cat.town/fishing/fish-raffle.

**Paid tickets (buying with caught fish) are out of scope for this revision** — free claim + reads only.

---

## Gacha — async VRF pulls

**GachaMachine** at `0xAD0ee945B4Eba7FB8eB7540370672E97eB951F1a` (Base) pays out seasonal items. Pulls are **asynchronously fulfilled via VRF** — one tx pays, a separate tx mints the NFT a few seconds later. Agents must account for the delay when answering "what did I get?".

### Basics

- **Daily cap:** 100 pulls per wallet, `dailyUsageLimit()` constant; resets at 00:00 UTC. Per-user remaining: `getPlaysLeftForToday(user)`.
- **Cost:** USD-denominated. `capsulePriceUSD()` returns cents (currently **50 = $0.50 per pull**). Convert to KIBBLE at pull time using the Kibble Price Oracle — same oracle as the boutique (`0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7`, scale `10^18`). At current rates: **~527 KIBBLE per pull**.
- **Drops are flat random.** Every pull is independent, uniformly weighted against the current season's pool. No pity, no streaks, no reroll. Two pulls back-to-back are statistically identical.
- **Seasonal pool** is filterable via `/v2/items/master?limit=1000` (public): `source == "Gacha"` + `dropConditions.seasons` includes the current season (from `GameData.getCurrentSeason()`).

### Write path — `purchaseAndOpenCapsule()` (payable)

Single pull per tx. Multi-pulls are **N sequential txs** — there's no onchain batch call.

```
Preconditions:
  - kibble.allowance(user, gacha) >= kibble_cost        (standard ERC-20, wei)
  - msg.value = VRF fee (small ETH amount, per pull)
  - getPlaysLeftForToday(user) > 0
  - user holds enough ETH on Base for VRF fee + gas (see "ETH preflight" below)

Effect of the pay tx:
  - pulls KIBBLE from user
  - submits VRF randomness request
  - does NOT mint the NFT — that happens in a separate tx on VRF callback
```

### ETH preflight — check before pulling, suggest a swap if low

Gacha is the only Cat Town write in this skill that sends `msg.value`, so users who normally hold only KIBBLE can trip on it silently. **Before building any pull tx, read the user's ETH balance on Base.** If it's thin:

```
if user.ethBalance < ~ $0.50 USD:
  # not enough headroom for VRF fee + gas across a few pulls
  surface to user:
    "You're low on ETH on Base ($X). Gacha pulls need a bit of ETH for
     the VRF fee and gas. Want me to swap ~$1 of KIBBLE to ETH so you're
     topped up?"
```

Rules for the swap suggestion:

- **Prioritise KIBBLE as the source token.** Most Cat Town users hold it already, and swapping a sliver back to ETH is the least-disruptive path. Fall back to other tokens only if KIBBLE balance is also insufficient.
- **Default target: ~$1 of ETH.** Enough for ~10+ pulls with comfortable gas headroom. Scale up for bigger batches (roughly `max($1, $0.08 × N)` for N pulls).
- **Offer, don't auto-execute.** Present it as a confirmation before running the swap, unless the user explicitly said "just do it."
- Bankr's built-in swap (via the `trails` or `symbiosis` skill) handles the actual swap — this skill just triggers the suggestion at the right moment.

If the user declines the swap but still has *some* ETH, proceed with whatever pulls that ETH covers and tell them how many before the wallet runs dry.

### Reading the result — the token-id ordering trick

Because the pay tx and the mint tx are decoupled, the frontend correlates them by **capsule token id** (mirrored here). Process for one or many pulls:

```
1. Before pulling:
     latestId = max( item.id for item in GET /v2/items/capsule/<user> )
     (If the user has never pulled, the endpoint returns a 500 — treat it as empty, latestId = 0.)

2. Submit N pay txs.

3. After confirmations, poll GET /v2/items/capsule/<user> every 1–2 s:
     newItems = [ item for item in response if item.id > latestId ]
     if len(newItems) >= N:   return newItems (these are your results)
4. Time out at ~60 s. If not all N have landed, surface a "still pending" reply.
```

If the user spins 10 times, you must wait for 10 items with `id > latestId`. Partial results are fine to preview, but be explicit about how many are still pending. Don't assume pull-1's result has a smaller id than pull-2's — VRF callbacks can interleave.

### Response patterns

**Can Bankr poll for results?** If yes, use the loop above and report when all N have landed. If not, submit the pay tx(s), return immediately with "Spin submitted — ask me again in ~30 seconds to see what dropped" and let the user re-prompt. When they come back, pull `/v2/items/capsule/<user>` and show items with `id > latestId` (where `latestId` was cached in the original turn).

### Always quote the item's value + offer a sell

When reporting a gacha result, look up the item's `sellValue` (in US cents) from `/v2/items/master`, convert to KIBBLE via the oracle, and surface both numbers. Then offer to sell — this skill has a vendor flow (see **Selling items** below).

Format: `"<Rarity> <Name>" (<Type>, <Collection>) worth ~<X> KIBBLE (~$<Y>)`. Example for Fern: `"Common Fern (Collectible, Plant Minis) worth ~105 KIBBLE (~$0.10)"`.

### Example replies

**Polling path (Bankr can wait):**

> Spinning once… paid ~527 KIBBLE. Waiting on VRF…
>
> 🎉 You pulled a **Common Fern** (Collectible, Plant Minis collection) worth **~105 KIBBLE (~$0.10)**. You have **99 pulls left today**.
>
> Want me to sell it for you? After the 5% vendor fee, you'd get ~100 KIBBLE.

**Non-polling path (no async support):**

> Submitted 5 pulls (~2,635 KIBBLE total). VRF needs a few seconds to mint each one. Ask me "what did I get?" in ~30 seconds and I'll check — I can also sell the results right away if you want.

### Reads cheat-sheet

| Call                                                   | Use case                            |
|--------------------------------------------------------|-------------------------------------|
| `dailyUsageLimit()`                                    | Global 100/day cap                  |
| `getPlaysLeftForToday(user)`                           | Remaining pulls for this wallet today |
| `capsulePriceUSD()`                                    | Cost per pull in US cents           |
| `getAllItemConfigs()` / `getItemConfig(index)`         | Onchain pool definitions            |
| `GET /v2/items/capsule/<user>`                         | Result polling target               |
| `GET /v2/items/master?limit=1000`                      | Full catalog; filter `source=Gacha` |

Full contract signatures, VRF event names, oracle math, and the capsule API quirks (500 for cold wallets, etc.): [references/gacha/contract.md](references/gacha/contract.md), [references/gacha/api.md](references/gacha/api.md). Player-facing overview + pool archive: https://docs.cat.town/shops/gacha, https://docs.cat.town/items/gacha/archive.

---

## Selling items (vendor, V2 minter only)

Players sell **Treasures** and **Collectibles** (including gacha pulls) to the **SellItems** contract at `0x49936db5Dcbc906D682CFa2dcfAb0788e3ee5808` for KIBBLE, minus a **5% merchant fee**.

This skill revision supports **only items minted by the V2 minter** (`0x7b65ec82cB4600Bc1dCc5124a15594976f19eA14`). Legacy V1-minted items must be filtered out in the preflight.

### Value math

Each sellable item has a `sellValue` in the public item catalog — **US cents**, not KIBBLE, not wei:

```
GET https://api.cat.town/v2/items/master?limit=1000
  → items[].sellValue  (cents, e.g. 10 = $0.10)
```

Convert to KIBBLE for display via the Kibble Price Oracle:

```
usd              = sellValue / 100
kibble_value     = usd / (rawKibbleUsdPrice / 10^18)
payout_after_tax = kibble_value * 0.95                  // 5% vendor fee
```

A freshly minted NFT (e.g. a gacha pull) also carries a `Sell Value (KIBBLE)` trait with the pre-computed KIBBLE amount. Prefer the trait when available; fall back to the catalog formula.

### Write flow

Single function, batched up to **25 items per call**:

```
SellItems.sellMultipleNFTsToContract(
  address[] nftContracts,   // V2 minter address repeated, one per item
  uint256[] tokenIds,       // token ids to sell
  uint256[] amounts         // 1 per item (ERC-1155)
)
```

Preflight:

1. **Approval** — check `V2Minter.isApprovedForAll(user, sellContract)`. If false, submit `setApprovalForAll(sellContract, true)` first. One-time per wallet.
2. **V2 filter** — only include items whose source nftContract is the V2 minter. Skip V1, tell the user how many were skipped.
3. **Ownership** — `V2Minter.balanceOf(user, tokenId) >= 1` for each item.
4. **Vendor liquidity** — `KIBBLE.balanceOf(sellContract)` must exceed total payout; otherwise reverts `KibbleTransferFailed` ("vendor is out of KIBBLE").

Tax rate is read from `taxRateInBps()` (currently 500 = 5%, rounded from chain on the frontend).

### Inventory API — "what can I sell?"

```
GET https://api.cat.town/v2/inventory/<address>/paginated?hasSellValue=true&sortBy=kibble&sortOrder=desc
```

Public, no auth. `hasSellValue=true` filters out unsellable types automatically. Sort by `kibble` to surface the highest-value items first — mirrors how the frontend's vendor modal opens.

### Response pattern — "sell my items"

1. Pull inventory via the API above.
2. Filter to V2-minted items only.
3. Sum expected payout (`sellValue` summed, converted to KIBBLE, × 0.95).
4. Confirm with the user: *"I'll sell N items for ~X KIBBLE (~$Y) after the 5% fee. Go ahead?"*
5. Run the approval if needed, then `sellMultipleNFTsToContract(...)`.
6. After confirmation, refetch inventory + KIBBLE balance and report the actual payout.

Example reply after a gacha pull:

> You pulled a Common **Fern** (~105 KIBBLE, $0.10). Want me to sell it right away? That'd net ~100 KIBBLE after the 5% fee.

Or, for a batch:

> You've got 12 V2-minter items worth selling, totaling ~3,420 KIBBLE after the 5% fee. (Skipping 2 legacy items.) Want me to sell all 12, or cherry-pick?

Full ABI surface, approval detail, inventory-API query params, revert catalogue, and the batch recipe: [references/sell-items/contract.md](references/sell-items/contract.md). Player-facing overview: https://docs.cat.town/shops/sell-items.

---

## KIBBLE tokenomics (Jasper's answers)

When a user asks about KIBBLE — "how much is staked?", "how much burned?", "what's the APY?" — mirror the numbers the NPC **Jasper** quotes at the Wealth & Whiskers Bank. Three headline stats, each from live reads:

### % Burned (of TOTAL supply)

```
burnedPercent = balanceOf(0xdEaD on KIBBLE) / 1,000,000,000 × 100
```

Denominator is total supply (1B), not circulating — that's how Jasper phrases it. **Live at time of writing: ~66.3% of supply already burned.**

### % Staked (of CIRCULATING supply)

```
circulating   = totalSupply − balanceOf(0xdEaD)
stakedPercent = RevenueShare.getTotalStaked() / circulating × 100
```

Denominator is **circulating** (total minus burned), so users get a realistic number after the deflationary burn. **Live at time of writing: ~24.0% of circulating KIBBLE is staked.**

### Staking APY at Wealth & Whiskers

Derived dynamically from **baronbot** (`0x8Ff7AcCCf73c515c1f62Fc7b64A63F17Ce99659e`, rank-1 continuous staker) because the return per KIBBLE is the same for every active staker. Formula:

```
1. GET /v2/revenue/deposits/<baronbot> — keep last 30 days of deposits
2. monthly_revenue = period_revenue * (30 / days_since_first_deposit)
3. monthly_rate    = monthly_revenue / baronbot.stakedAmount
4. apy             = min(((1 + min(monthly_rate, 0.50))^12 − 1) * 100, 1000)
```

**Live at time of writing: ~30% APY** — not a fixed rate; drifts with weekly fishing + gacha revenue.

### Example reply

> **KIBBLE tokenomics (live):** ~66% of supply has been burned, ~24% of circulating is staked in Wealth & Whiskers, and staking currently pays ~30% APY. Want me to walk you through staking? The lock period is 14 days.

Full formulas, APY caps, and the live worked example: [references/kibble/tokenomics.md](references/kibble/tokenomics.md). Player-facing KIBBLE economy overview: https://docs.cat.town/economy/tokens/kibble, https://docs.cat.town/get-started/kibble-economy.

---

## Executing transactions via Bankr

For any write call (`approve`, `stake`, `claim`, `claimAndRestake`, `unlock`, `relock`, `unstake`):

Natural-language Bankr agent prompt:

```bash
bankr agent prompt "Stake 1000 KIBBLE in Cat Town"
```

Or encode calldata and submit directly:

```bash
bankr wallet submit --to 0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1 --data <encoded-calldata> --chain base
```

Remember: submit the ERC-20 `approve` on the KIBBLE token (`0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`, target = RevenueShare) before `stake` if the current allowance is insufficient.

---

## Pitfalls

- **Forgetting the approval.** `stake` reverts cleanly but wastes a user's tx. Read `allowance(user, revenueShare)` first; only approve if low.
- **Unstaking while unlocking.** Reverts. Check `isUnlocking(user)` and `unlockEndTime(user)` before constructing an `unstake` tx.
- **Assuming continuous rewards.** `pendingRewards` is a step function — it only goes up when the backend calls `depositRevenue`. Between deposits, polling will show no change, and that is correct. Use the calendar above to set expectations.
- **Stale `LOCK_PERIOD` assumptions.** Currently **14 days** (1,209,600 seconds); safe to quote at the point of unlock because `unlockEndTime` is snapshotted per-user. Read `LOCK_PERIOD()` live only if you want defensive protection against UUPS upgrades.
- **Using the legacy contract.** An older staking contract (`0xc3398Ae89bAE27620Ad4A9216165c80EE654eE96`) exists but is deprecated. Do not send new stakes there.

---

## Troubleshooting

### `ERC20: transfer amount exceeds balance` on `stake`

**99% certainty: you wei-encoded the `stake` argument.** RevenueShare takes `amount` in whole KIBBLE and multiplies by `10^18` internally. If you pass `N × 10^18` thinking it's wei, the contract attempts to pull `N × 10^36` tokens from your balance, which trivially exceeds any balance.

Fix: pass the whole-KIBBLE integer. To stake 100 KIBBLE, call `stake(100)`, not `stake(100000000000000000000)`.

The KIBBLE `approve()` call is the opposite — it's a standard ERC-20 call and *does* take wei. So the correct 100-KIBBLE flow is:

```
kibble.approve(revenueShare, 100_000000000000000000)   // 100 × 10^18 wei
revenueShare.stake(100)                                // whole KIBBLE
```

Confirmed by onchain simulation against `0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1`:

| Call | Expected behaviour |
|---|---|
| `stake(1)` with ≥1 KIBBLE allowance | succeeds (stakes 1 KIBBLE) |
| `stake(100)` with =100 KIBBLE allowance | succeeds, hits cap exactly |
| `stake(101)` with 100 KIBBLE allowance | reverts `transfer amount exceeds allowance` |
| `stake(1e18)` with 100 KIBBLE allowance | reverts `transfer amount exceeds balance` ← the mistake |

### `ERC20: transfer amount exceeds allowance` on `stake`

`stake(N)` requires `allowance(signer, revenueShare) ≥ N × 10^18` on the KIBBLE token. Call `approve(revenueShare, N × 10^18)` from the same signer first.

### `unstake` reverts with no obvious reason

Check `isUnlocking(signer)`. If `true`, `unstake` reverts until `block.timestamp >= unlockEndTime(signer)`. Either wait out the window or call `relock()` to cancel the unlock and return to the earning pool.

### Diagnostic no-arg write tests

To verify the signer + contract are wired up without any amount-encoding risk:

- `unlock()` — no args. Succeeds even with 0 staked (sets `isUnlocking = true`). Follow with `relock()` immediately to avoid side effects.
- `claim()` — no args. No-ops cleanly when `pendingRewards(signer) == 0`.
