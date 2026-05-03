# Cat Town Weekly Calendar

All times UTC. Cat Town runs on a fixed weekly cycle; agents should use this table to set user expectations and schedule follow-ups.

| Day       | Event                             | Time                        | Host              | Related skill surface         |
|-----------|-----------------------------------|-----------------------------|-------------------|-------------------------------|
| Monday    | **Fishing revenue deposit**       | by 12:00 (often earlier)    | Theodore / Cassie | KIBBLE staking                |
| Mon–Fri   | Fish raffle ticket sales open     | all day                     | Paulie            | Fish raffle (future)          |
| Mon–Fri   | Weekday fishing                   | all day                     | Skipper           | Fishing (future)              |
| Wednesday | **Gacha revenue deposit**         | by 12:00 (often earlier)    | Theodore / Cassie | KIBBLE staking                |
| Friday    | **Fish raffle draw**              | 20:00                       | Paulie            | Fish raffle (future)          |
| Sat–Sun   | **Weekly fishing competition**    | Sat morning → Sun night     | Isabella          | Fishing competition (future)  |

### Cat Town NPC cheat-sheet

- **Theodore** — day-shift teller at the Wealth & Whiskers Bank. Point daytime staking prompts at him.
- **Cassie** — evening-shift teller at the Wealth & Whiskers Bank. Use her in evening copy.
- **Paulie** — host of Paulie's Fish Raffle. Tickets Mon–Fri, draw Friday 20:00 UTC.
- **Skipper** — weekday fishing NPC. Weekday catches do **not** feed the stakers pool.
- **Isabella** — weekend fishing-competition host. Competition catches **do** feed the stakers pool.

Daycare (a third revenue source visible in the RevenueShare contract as `source = "daycare"`) is **not yet implemented** — no deposits of that source should be expected.

---

## Fishing revenue → KIBBLE stakers

**Only weekend-competition catches feed stakers.** Fish identified Monday–Friday with Skipper do **not** contribute to the stakers pool.

During the Sat–Sun fishing competition (hosted by Isabella), each fish a player identifies splits as:

| Portion | Destination                    |
|---------|--------------------------------|
| 70%     | Treasure catches (player pool) |
| 10%     | Weekend competition leaderboard pool |
| **10%** | **KIBBLE stakers**             |
| 7.5%    | Town Treasury                  |
| 2.5%    | Burned                         |

The stakers' 10% accumulates across the weekend and is pushed onchain as one `depositRevenue(amount, "fishing")` call **by 12:00 UTC Monday** (often earlier). It shows up instantly as `pendingRewards` for every staker whose `isUnlocking == false` at the time of the call.

Practical implication for agents: Monday morning UTC is the natural moment to prompt users to `claim()` or `claimAndRestake()` — that's when their pending rewards jump after the weekend's competition revenue lands.

## Gacha revenue → KIBBLE stakers

Gacha revenue aggregates across the week and is pushed onchain as `depositRevenue(amount, "gacha")` **by 12:00 UTC Wednesday** (often earlier). Same onchain mechanics as fishing — the only difference is `source`.

## Fish raffle (Friday 20:00 UTC draw) — Paulie

Paulie's Fish Raffle runs weekly.

- Tickets bought with caught fish (20 kg per ticket, subject to change).
- Ticket sales window: Monday through Friday UTC.
- One free ticket per week per player.
- Draw: Friday 20:00 UTC; sales close ~10 minutes earlier at 19:50 UTC.
- **5 winners per draw**, chosen via Chainlink VRF. Prize pool is split **equally** among the 5 — ranks are ordering only, not prize weighting.
- Prize pool is a tier-based fraction of the `FreeToPlayPool` balance. 8 tiers keyed to `totalTickets` sold that round, paying 30 bps (0.3%) at 0 tickets up to 100 bps (1.0%) once sales cross 5,500. More tickets sold → bigger cut of the pool paid out.

Full contract + API reference, free-ticket claim flow, tier table, and chance-to-win math: [../fish-raffle/contract.md](../fish-raffle/contract.md), [../fish-raffle/api.md](../fish-raffle/api.md).

## Weekly fishing competition (Sat–Sun) — Isabella

Isabella hosts the weekend fishing competition. Only catches made during her window feed the KIBBLE stakers pool; Skipper's weekday catches do not.

- Automatic entry for every fish caught during the window (biggest single catch wins).
- Top 10 by biggest fish — per-rank prize split (verified in frontend code):

  | Rank    | Prize share |
  |---------|-------------|
  | 1st     | 30%         |
  | 2nd     | 20%         |
  | 3rd     | 10%         |
  | 4th     | 8%          |
  | 5th     | 8%          |
  | 6th     | 7%          |
  | 7th     | 5%          |
  | 8th     | 4%          |
  | 9th     | 4%          |
  | 10th    | 4%          |

- The leaderboard is only meaningful Saturday–Monday; outside that window it reflects the prior week's results.

Full contract + API reference, prize-pool math (10% top-10 / 80% treasures / 10% stakers), and active/inactive response patterns: [../fishing/competition.md](../fishing/competition.md).

---

## Source of truth

- Official Cat Town docs: https://docs.cat.town
  - Staking: https://docs.cat.town/economy/staking.html
  - Fish raffle: https://docs.cat.town/fishing/fish-raffle.html
  - Weekly competition: https://docs.cat.town/fishing/weekly-competition.html
- Cat Town app: https://cat.town (staking UI at https://cat.town/bank)
