# Cat Town Weekly Fishing Competition

Isabella's weekend fishing competition runs **Saturday 00:00 UTC → Monday 00:00 UTC** (a 48-hour window, loosely "Sat morning → Sun night"). This doc covers live state detection, prize-pool math mirroring the frontend's exact numbers, and leaderboard reads.

Player-facing overview: https://docs.cat.town/fishing/weekly-competition.

## Addresses

| Contract | Address |
|---|---|
| FishingCompetition (Base)    | `0x62a8F851AEB7d333e07445E59457eD150CEE2B7a` |
| FishingCompetition (Sepolia) | `0xc6d6a4C530f84bd742FEb5c6Bdc2a15c0C3148ec` |

ABI: `/abi/internal/Fishing/FishingCompetitionAbi.json`

## Is a competition running?

Two equivalent sources of truth:

- **Onchain**: `isCompetitionActive()` → `(bool active, bytes32 eventId)`
- **API**: `competition.isActive` in the leaderboard response (below)

If `isActive == false`, the `getCurrentCompetition()` call still returns the **most recent completed** competition — use it for "tell me about the last event" queries.

### Next competition start

Competitions run weekly. No contract method returns the next start time directly — compute locally:

```
next_start = next Saturday 00:00 UTC
time_until = next_start - now
```

Round the delta to days + hours for user-facing copy.

## Prize-pool math — the frontend's exact numbers

The contract's `prizePool` field is the **total volume** of KIBBLE spent on fish identifications during the competition (the UI labels it "Total Volume"). It splits three ways, all percentages hardcoded in the frontend (`components/organisms/Scenes/Fishing/Competition/Modals/FishingSatchel/FishingCompetitionOverview.tsx` lines 38–60):

```
prizePool                             // total volume
leaderboardShare = prizePool * 0.10   // top-10 prize pool ("Prize Pool" in UI)
treasureShare    = prizePool * 0.80   // treasures returned to players
stakersRevenue   = prizePool * 0.10   // feeds RevenueShare for KIBBLE stakers
```

These three sum to 100% exactly.

> **Documentation drift to know about.** docs.cat.town breaks the 80% down further as 70% treasures / 7.5% treasury / 2.5% burn. The frontend rolls those into a single 80% "supermarket" share for display. **Mirror the frontend's 10/80/10 when answering "where does the prize pool go?"** — that's what users see in the game UI.

### Top-10 distribution (of `leaderboardShare`)

From `fishingLeaderboardShareForRank()` in `hooks/api/useFishingLeaderboard.tsx:9-21`:

| Rank  | Share of `leaderboardShare` |
|-------|-----------------------------|
| 1     | 30%                         |
| 2     | 20%                         |
| 3     | 10%                         |
| 4     | 8%                          |
| 5     | 8%                          |
| 6     | 7%                          |
| 7     | 5%                          |
| 8     | 4%                          |
| 9     | 4%                          |
| 10    | 4%                          |
| 11+   | 0%                          |

Percentages sum to exactly 100%. No dust redistribution when fewer than 10 entrants (the unspent portion is simply not paid out). `Math.floor` to whole KIBBLE at display time.

```
prize_for_rank(rank, prizePool):
  if rank > 10: return 0
  leaderboard_share = prizePool * 0.10
  return floor(leaderboard_share * share_for_rank(rank))
```

## Worked example — the most recent completed competition

Live pull at time of writing (eventId `0x1bee…`, 100 participants, prizes distributed):

```
prizePool    = 3,057,133.66 KIBBLE         (total volume)
leaderboard  = 305,713.37 KIBBLE  (10%)     → split among top 10
treasures    = 2,445,706.93 KIBBLE (80%)    → returned to players as treasures
stakers      = 305,713.37 KIBBLE  (10%)     → to RevenueShare stakers (pushed Monday ≤12:00 UTC)
```

Top 10 payouts (USD at `getKibbleUsdPrice() ≈ $0.000949`):

| Rank | %   | KIBBLE  | USD      |
|------|-----|---------|----------|
| 1    | 30% | 91,714  | ~$87.02  |
| 2    | 20% | 61,143  | ~$58.01  |
| 3    | 10% | 30,571  | ~$29.01  |
| 4    | 8%  | 24,457  | ~$23.20  |
| 5    | 8%  | 24,457  | ~$23.20  |
| 6    | 7%  | 21,400  | ~$20.30  |
| 7    | 5%  | 15,286  | ~$14.50  |
| 8    | 4%  | 12,229  | ~$11.60  |
| 9    | 4%  | 12,229  | ~$11.60  |
| 10   | 4%  | 12,229  | ~$11.60  |

Winner: **bitcoinbov.base.eth** — **Elusive Marlin at 46.364 kg** (46,364 grams), ~91,714 KIBBLE payout.

## Leaderboard — public API

`GET https://api.cat.town/v1/fishing/competition/leaderboard` — no auth, plain GET. Server-side cache 10s; frontend refetches every 120s. Returns top 50 max.

### Response shape

```ts
{
  competition: {
    eventId: string            // bytes32 event id, hex
    name: string               // e.g. "Fishing Competition"
    startTime: number          // unix seconds
    endTime: number            // unix seconds
    prizePool: string          // KIBBLE wei as a decimal string (BigInt)
    prizesDistributed: boolean
    isActive: boolean
    totalPlayers: number       // participant count
  }
  leaderboard: Array<{
    rank: number               // 1-indexed
    player: string             // 0x address
    size: string               // fish weight in GRAMS (parse int; /1000 for kg)
    timestamp: number          // unix seconds of the catch
    fishName: string           // e.g. "Elusive Marlin"
    tokenId: string            // NFT id of the caught fish
    isShiny: boolean
    basename?: string          // e.g. "bitcoinbov.base.eth"
    equipment?: {
      hat?: string; body?: string; eyewear?: string; companion?: string
      whiskers?: string; neck?: string; cat?: string; eyeColor?: string
      background?: string
    }
  }>
}
```

### Scoring formula

**Biggest single fish** by weight in grams. Ties broken by earliest timestamp. Your *one* best catch determines your rank — not total weight and not count.

### "Off-season" behaviour

When `isActive == false`, the API still returns the **most recent completed** competition (`prizesDistributed: true`, full final leaderboard). Use this to answer "tell me about the last fishing event."

## Onchain reads

| Function                          | Returns                                               | Notes                                     |
|-----------------------------------|-------------------------------------------------------|-------------------------------------------|
| `isCompetitionActive()`           | `(bool active, bytes32 eventId)`                      | Quickest active/not-active check          |
| `getCurrentCompetition()`         | `(bytes32, string, uint256, uint256, uint256, bool)`  | eventId, name, startTime, endTime, prizePool, prizesDistributed |
| `getLeaderboard()`                | array of 50 entries                                   | player, size, timestamp, fishName, tokenId, isShiny |
| `getParticipants()`               | `address[]`                                           | every participant of the current comp     |
| `getCompetition(bytes32 eventId)` | same tuple as `getCurrentCompetition()`               | historical lookup by eventId              |

All reads are cheap. Prefer the API unless you want second-accurate numbers.

## Live snapshot captured during writing

- `isCompetitionActive()` → **false**
- `getCurrentCompetition()` → Fishing Competition, 100 players, 3.06M KIBBLE total volume, prizes distributed
- Winner: **bitcoinbov.base.eth** with **46.364 kg Elusive Marlin**, took ~91,714 KIBBLE (~$87)
