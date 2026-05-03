# Fish raffle — public API

Two unauthenticated JSON endpoints on `https://api.cat.town` for leaderboard and past-draw data. Use these whenever a read would otherwise mean scanning `participants()` + `ticketsThisWeek()` across 200+ addresses — the API aggregates it cheaply.

## `GET /v1/tickets/leaderboard`

Current-round ticket counts per buyer, with `totalTickets` for the round.

### Response

```ts
{
  roundId: number                     // active round id
  totalTickets: number                // sum across all buyers this round (use this as the denominator for chance-to-win)
  leaderboard: Array<{
    rank: number                      // 1-indexed, sorted by totalCount desc
    buyer: string                     // 0x address (lowercase)
    roundId: number                   // same roundId as outer
    totalCount: number                // this buyer's tickets this round (free + paid)
    firstAt: string                   // ISO8601 — first ticket this round
    lastAt: string                    // ISO8601 — most recent ticket
    txCount: number                   // onchain txs this buyer submitted (1 for pure free-claimers)
    basename?: string                 // base.eth name if set
    equipment?: { hat?, body?, eyewear?, companion?, whiskers?, neck?, cat?, eyeColor?, background? }
  }>
}
```

### Typical size

A few hundred entries once sales have been open for a day or two. Live at time of writing: 204 participants, 2,855 total tickets.

### Use cases

- **Chance-to-win** — find the user's `buyer` entry, then `5 * totalCount / totalTickets` for approximate odds.
- **"Who's leading?"** — `leaderboard[0..4]` with basename, equipment, and ticket count.
- **"How big is this week's pool?"** — combine `totalTickets` with the tier math in `contract.md` (tier bps × `FreeToPlayPool.poolBalance()` / 10000).

## `GET /v1/tickets/winners`

Most recent completed draw. No historical list — just the last round's winners.

### Response

```ts
{
  roundId: string                     // the round these winners came from (one behind the active round)
  blockNumber: number
  transactionHash: string
  timestamp: string                   // ISO8601 of the draw tx
  winners: Array<{
    rank: number                      // 1–5 but all 5 get the same prize (see below)
    address: string                   // 0x winner
    prizeAmount: string               // KIBBLE wei as a string (BigInt)
    basename?: string
    equipment?: { … }                 // same shape as leaderboard
  }>
}
```

### Equal payouts — not a ranked split

Despite the `rank` field, all 5 winners receive **the same KIBBLE amount** (prize pool divided equally by `winnersPerDraw`). Live example from the last draw (round 30):

- All 5 winners: `prizeAmount = 9,363,857,476,887,126,627,062` wei = **9,363.86 KIBBLE each** (~$8.88)
- Total paid out: 46,819 KIBBLE (~$44.4)
- The `rank` is just the VRF ordering; it carries no prize weight.

If a user asks "who won last week?", list all 5 with basename/address and the single equal payout.

## Caching

Both endpoints hit `Cache-Control: no-cache` but CDN/server caches are typically 30–60 s. Refetch every 1–2 min if you're building a live dashboard; once per request is fine for chat.

## What's not here

- **Historical draws beyond the latest.** The API returns only the most recent. If a user asks about round 27, 28, etc., the public API can't answer — you'd need to scan onchain events (`WinnersDrawn(uint256 indexed roundId, address[] winners, uint256[] prizes)`).
- **Per-wallet free-claim status.** Use the onchain reads (`canClaimFreeTicket(user)`, `lastFreeTicketRound(user)`) — the API doesn't expose this.
