# Paulie's Fish Raffle — contract reference

Cat Town's weekly raffle. Free ticket for every player every week, plus paid tickets (burn 20 kg of caught fish per ticket). Draw is Friday 20:00 UTC — 5 winners selected via Chainlink VRF, pool split equally.

Player-facing overview: https://docs.cat.town/fishing/fish-raffle.

Two contracts are involved:

| Contract        | Address                                       | Role                             |
|-----------------|-----------------------------------------------|----------------------------------|
| FishRaffle      | `0x5E183eBc7CA4dF353170C35b4D69Ea9f42317b28`  | Tickets, rounds, draws, prizes   |
| FreeToPlayPool  | `0x131E680dc7A146F00b282FBD7d6261c5B38c4Fa6`  | Pool balance + tier table        |

ABIs: `/abi/internal/FishRaffle/FishRaffleAbi.json`, `/abi/internal/FreeToPlayPool/FreeToPlayPoolAbi.json`.

## Game configuration — `getGameConfiguration()`

Single tuple read; all the constants for this revision:

```ts
(
  kgPerTicket: uint256,            // 20 — grams of fish burned per paid ticket
  drawHour: uint256,               // 20 — UTC hour of the draw (20:00 = 8pm)
  salesCutoff: uint256,            // 600 — seconds before draw when sales close (sales end ~19:50 UTC Fri)
  perWalletWeeklyLimit: uint256,   // 200 — max paid tickets a wallet can buy per week
  winnersPerDraw: uint256,         // 5 — number of winners picked
  nextDrawTime: uint256            // unix seconds of the next Fri 20:00 UTC draw
)
```

Example live read at time of writing: `(20, 20, 600, 200, 5, 1777060800)` → next draw **2026-04-24 20:00 UTC**.

## Free ticket — the claim flow

Every wallet gets **one free ticket per ISO week**. The gate is onchain: the contract compares `lastFreeTicketRound[user]` with `currentISOWeek()`.

### Preflight read

```
canClaimFreeTicket(address user) → bool
```

Returns true only when `lastFreeTicketRound[user] != currentISOWeek() && !salesClosed() && !paused()`. Always call this before building a claim tx — skips a wasted revert.

### Write

```
claimFreeTicket()   // no args, msg.sender inferred
```

Emits `FreeTicketClaimed(address indexed claimer, uint256 indexed roundId)`. No token approval needed.

### Known reverts

| Error                        | Meaning                                     | User-facing response                                         |
|------------------------------|---------------------------------------------|--------------------------------------------------------------|
| `FreeTicketAlreadyClaimed`   | User already claimed this ISO week          | "You've already claimed your free ticket this week. Next one resets Monday 00:00 UTC." |
| `SalesClosed`                | Sales window ended (≥10 min before draw)    | "Sales are closed. Winners draw Friday 20:00 UTC."           |
| `EnforcedPause`              | Contract paused by owner                    | "The raffle is paused — try again later."                    |
| `MaxParticipantsReached`     | Participant cap hit (rare)                  | "Raffle is at max participants this round."                  |

## Paid tickets — `buyTickets(uint256[] tokenIds)`

Burns fish NFTs owned by `msg.sender`. Each 20 kg of total weight in the provided token IDs mints 1 ticket (floor division). Multiple tickets in one tx — pass more fish.

| Constraint            | Enforcement                                    |
|-----------------------|------------------------------------------------|
| Min per ticket        | 20 kg total weight across the provided ids     |
| Per-wallet weekly cap | `perWalletWeeklyLimit` (200)                   |
| Sales window          | Mon 00:00 UTC → Fri 19:50 UTC (`salesClosed == false`) |

Emits `TicketPurchased(address indexed buyer, uint256 indexed roundId, uint256 count, uint256[] burnedIds)`. Approval of the fish NFT contract may be required depending on its standard — this skill revision does not build the paid-buy path (free claim only).

Reverts: `InsufficientWeight`, `WalletCapExceeded`, `SalesClosed`, `MaxParticipantsReached`, `EnforcedPause`.

## Prize pool math — pool balance × tier bps

The prize pool for a given round is a **tier-based fraction of the FreeToPlayPool balance**. Total tickets sold that round picks the tier; higher ticket volume unlocks higher basis points.

### Read the tier table from the pool contract

```
FreeToPlayPool.getTiers() → Array<{ minTickets: uint256, bps: uint256 }>
FreeToPlayPool.poolBalance() → uint256   // KIBBLE in wei
```

Live tier table at time of writing:

| Tier | minTickets | bps | % of pool |
|------|------------|-----|-----------|
| 1    | 0          | 30  | 0.30%     |
| 2    | 250        | 40  | 0.40%     |
| 3    | 500        | 50  | 0.50%     |
| 4    | 850        | 60  | 0.60%     |
| 5    | 1,400      | 70  | 0.70%     |
| 6    | 2,200      | 80  | 0.80%     |
| 7    | 3,500      | 90  | 0.90%     |
| 8    | 5,500      | 100 | 1.00%     |

### Formula

```
current_tier    = tiers.findLast(t => totalTickets >= t.minTickets)   // highest threshold crossed
prize_pool_wei  = poolBalance * current_tier.bps / 10000
per_winner_wei  = prize_pool_wei / winnersPerDraw                     // equal split — NOT ranked
```

Divide by `10^18` for KIBBLE; multiply by `getKibbleUsdPrice() / 10^18` for USD (see [../boutique/contract.md](../boutique/contract.md) for the oracle formula).

### Live worked example (at time of writing)

- `totalTickets = 2,855` → tier 6 (≥2,200 threshold), **80 bps**
- `poolBalance = 5,967,812 KIBBLE`
- `prize_pool = 5,967,812 × 0.008 = ~47,742 KIBBLE` (~$45.30)
- `per_winner = 47,742 / 5 = ~9,548 KIBBLE` each (~$9.06)

Last week's draw paid **9,363.86 KIBBLE to each of 5 winners** — consistent with the formula. Tickets grow the pool toward the next tier, which is at 3,500 tickets (90 bps).

## Chance to win

5 winners per draw, drawn via Chainlink VRF from the ticket pool **without replacement** (one wallet can win at most one of the 5 slots). Each individual ticket is equally weighted.

### Formulas

```
per_ticket_chance     = 1 / totalTickets                    // probability any single ticket wins a given slot
approximate_win_chance(tickets) = min(1, 5 * tickets / totalTickets)   // first-order approximation
exact_win_chance(tickets, totalTickets) = 1 − C(totalTickets − tickets, 5) / C(totalTickets, 5)
```

For small `tickets / totalTickets` the approximation is accurate to within ~1–2%. Use the approximation in chat responses; use the exact form if the user specifically asks.

### Live examples

| Player                      | Tickets | Approx chance of winning any of 5 slots |
|-----------------------------|--------:|----------------------------------------:|
| rank 1 (`0xef05…`)          | 399     | 5 × 399 / 2855 = **~69.9%**             |
| rank 2 (`bitcoinbov`)       | 364     | **~63.7%**                              |
| single-free-ticket claimant | 1       | 5 × 1 / 2855 = **~0.175%**              |

If a user asks "what's my chance?", call the `/v1/tickets/leaderboard` endpoint (see [./api.md](./api.md)), find their address, read `totalCount`, and compute against `totalTickets`.

## Claiming winnings — `claimPrize()`

Won prizes sit in `pendingPrizes[winner]` (KIBBLE wei) until claimed.

```
pendingPrizes(address user) → uint256      // read to check
claimPrize()                                // write, no args
```

Emits `PrizeClaimed(address indexed winner, uint256 amount)`. Reverts `NoPendingPrize` if nothing to claim.

## State reads

| Function                               | Returns          | Notes                                                 |
|----------------------------------------|------------------|-------------------------------------------------------|
| `currentRoundId()`                     | `uint256`        | Active round id                                       |
| `currentISOWeek()`                     | `uint256`        | ISO 8601 week index; gates free ticket claims         |
| `ticketsThisWeek(address user)`        | `uint256`        | User's tickets for the active round                   |
| `lastFreeTicketRound(address user)`    | `uint256`        | Week index of user's last free-ticket claim           |
| `salesClosed()`                        | `bool`           | Sales window closed, draw not yet run                 |
| `paused()`                             | `bool`           | Owner pause                                           |
| `cleanupInProgress()`                  | `bool`           | Draw done, prizes being distributed                   |
| `participants()`                       | `address[]`      | All participants in the current round                 |
| `maxParticipants()`                    | `uint256`        | Safety cap                                            |

### State machine

```
[SalesOpen]  ── Fri 19:50 UTC ──▶  [SalesClosed]  ── draw() w/ VRF ──▶  [WinnersChosen]  ── cleanup ──▶  [next round]
   Mon 00:00 UTC                      sales gate                          pendingPrizes set
```

## Live snapshot (captured during writing)

- `currentRoundId` = 31
- `currentISOWeek` = 2938
- Tickets sold this round: **2,855** (tier 6, 80 bps)
- Participants: **204**
- `poolBalance` = 5,967,812 KIBBLE → prize pool ≈ **47,742 KIBBLE** (~$45)
- Per-winner (5 winners): ~9,548 KIBBLE (~$9)
- Next draw: Friday 2026-04-24 20:00 UTC
- `paused`: false
