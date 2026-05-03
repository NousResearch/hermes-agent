# Gacha Machine — contract reference

Cat Town's gacha machine pays out seasonal items. Every pull is independent, uniformly weighted against the season's item table, and **asynchronously fulfilled via Chainlink/Supra VRF** — the pay tx submits a randomness request, and the NFT mints in a separate tx on VRF callback. Agents must account for the latency between submit and mint when answering the user.

Player-facing overview: https://docs.cat.town/shops/gacha. Pool archive (all items ever available): https://docs.cat.town/items/gacha/archive.

## Address

**Base mainnet (chain 8453):** `0xAD0ee945B4Eba7FB8eB7540370672E97eB951F1a`

ABI: `/abi/internal/GachaMachine/GachaMachineAbi.json`. Payment token: **KIBBLE** (`0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`).

## Daily limit

| Read                              | Returns  | Notes                                     |
|-----------------------------------|----------|-------------------------------------------|
| `dailyUsageLimit()`               | `uint256`| Global per-wallet cap. Current: **100**.  |
| `getPlaysLeftForToday(address)`   | `uint256`| Remaining pulls for the user today.       |

The "day" boundary is **00:00 UTC** (`SECONDS_IN_DAY = 86400`, day index = `block.timestamp / 86400`). Counter resets automatically at midnight UTC.

## Cost per pull

Price is USD-denominated; payment is in KIBBLE computed against the Kibble Price Oracle at pull time.

| Read                  | Returns  | Notes                                              |
|-----------------------|----------|----------------------------------------------------|
| `capsulePriceUSD()`   | `uint256`| Price **in US cents**. Current: `50` → **$0.50**.  |

KIBBLE cost per pull:

```
cents         = capsulePriceUSD()                                         // e.g. 50
rawKibbleUsd  = KibblePriceOracle.getKibbleUsdPrice()                     // USD × 10^18 per 1 KIBBLE
kibble_cost   = (cents / 100) / (rawKibbleUsd / 10^18)                    // simple form
             = (cents * 10^18) / (rawKibbleUsd * 100)                      // BigInt-safe
```

Live example: `50 cents / $0.0009487 per KIBBLE ≈ **527 KIBBLE per pull** ≈ 52,700 KIBBLE to hit the daily cap.

The pay tx also requires an **ETH value** for the VRF fee (fetched client-side from the `TokenPriceProvider`; agents submitting raw calldata should include a small ETH value per pull).

### ETH preflight + swap recipe

This is the only write in the skill that sends `msg.value`, so users who hold only KIBBLE will trip on it. Before constructing any pull, read the user's ETH balance on Base. If it's below ~$0.50 worth:

1. Don't silently fail. Surface the low-balance state to the user.
2. Offer a **KIBBLE → ETH swap** (prefer KIBBLE as source because most users hold it; fall back to other tokens only if KIBBLE balance is also short).
3. Default target: **~$1 of ETH** — enough for ~10 pulls with comfortable gas headroom. Scale: `max($1, $0.08 × N)` for N planned pulls.
4. Execute via Bankr's built-in swap skills (`trails`, `symbiosis`, etc.) — this skill doesn't need to implement the swap, just trigger the suggestion.

If the user declines but still has *some* ETH, proceed with as many pulls as the ETH covers and quote the number before they run dry.

## Seasonal drop pool

Drops are filtered by the current season. Two equivalent ways to enumerate the pool:

- **Onchain (authoritative):** `getAllItemConfigs() → ItemConfig[]`, where each `ItemConfig` exposes `traitNames`, `traitValues`, `probability`, and `availableSeasons[]`. Filter by `GameData.getCurrentSeason()`.
- **Offchain (friendlier):** `GET https://api.cat.town/v2/items/master?limit=1000` (public) — filter `source == "Gacha"` and `dropConditions.seasons` includes the current season. Same items, richer metadata (image URLs, flavor text, rarity strings).

Probability is a raw `uint256` weighted against a `PROBABILITY_SCALE = 1,000,000` constant. Higher weight = more likely. Every pull is independent — no pity, no streak. Two pulls back-to-back are statistically identical.

## Write path — `purchaseAndOpenCapsule()`

Single write for one pull. No onchain batch call; multi-pulls are N sequential txs.

```solidity
function purchaseAndOpenCapsule() external payable
```

Preconditions:

1. `kibble.allowance(user, gacha) >= kibble_cost` (standard ERC-20; **wei**, not whole KIBBLE).
2. `user` sends the VRF fee as `msg.value` (ETH).
3. `getPlaysLeftForToday(user) > 0`.

Effects:

- Pulls KIBBLE via `transferFrom`.
- Submits VRF randomness request (asynchronous — **NFT does not mint in this tx**).
- Emits internal tracking events; the NFT mint happens later in a separate tx.

### Events

| Event                                             | Fires                | Notes                                              |
|---------------------------------------------------|----------------------|----------------------------------------------------|
| (pay tx) VRF request event                        | pay tx mines         | Marks the pull as in-flight; no NFT yet.           |
| `ItemMinted(address user, uint256 itemId, string[] traitNames, string[] traitValues)` | VRF-callback tx      | The result. Carries all display metadata.          |

The pay tx will **not** have the result. The agent must read the result from `ItemMinted` (event log) or from the capsule API (below) after the callback tx lands.

## Async result model — the token-id ordering trick

Because the two txs are decoupled, the frontend correlates them by **token-id ordering**, not by VRF request id. Pattern (mirrored from `useGachaItemPolling` in the frontend):

```
1. BEFORE the pay tx, read the user's current max capsule token id:
     latestId = max( item.id for item in GET /v2/items/capsule/<user> )
     (If the user has never pulled, latestId = 0.)

2. SUBMIT N pay txs (one per pull).

3. AFTER each tx confirms, poll GET /v2/items/capsule/<user> every 1–2 s.
     newItems = [ item for item in response if item.id > latestId ]
     if len(newItems) >= N:
         # all N results have landed
         return newItems                              // newest first
     # otherwise keep polling

4. Give up after a reasonable window (e.g. 60 s) with a "results pending"
   message — VRF is fast on Base but network delays happen.
```

Important: items are minted in the order the VRF callbacks land, which is roughly (but not exactly) the order the pay txs were submitted. **Don't assume pull 1's result has a smaller id than pull 2's.** Always compare against the pre-spin `latestId` and count how many have landed.

If the user pulls 10 times, wait for 10 items with `id > latestId` before reporting results. Partial results are OK to preview but state clearly that the remaining pulls haven't confirmed yet.

## Capsule history API

```
GET https://api.cat.town/v2/items/capsule/<address>
```

Public, no auth. Returns the user's capsule NFTs (not a global feed). Used as the polling target by the frontend's `useGachaItemPolling` hook (1-second interval).

Response shape (array of items): `[{ id, name, rarity, imageUrl, traitNames, traitValues, ... }, ... ]`. The `id` is the key field for the ordering trick. A 500 response can mean the address has never pulled; treat it as an empty array.

Full API shape notes: [./api.md](./api.md).

## Reads cheat-sheet

| Call                                  | Returns                        | Notes                                       |
|---------------------------------------|--------------------------------|---------------------------------------------|
| `dailyUsageLimit()`                   | `uint256`                      | 100 (global)                                |
| `getPlaysLeftForToday(address)`       | `uint256`                      | Per user, resets 00:00 UTC                  |
| `capsulePriceUSD()`                   | `uint256`                      | Price in US cents                           |
| `getAllItemConfigs()`                 | `ItemConfig[]`                 | Pool definitions                            |
| `getItemConfig(uint256 index)`        | `ItemConfig`                   | Single pool item                            |
| `balanceOf(address, uint256 id)`      | `uint256`                      | ERC-1155 ownership of a minted item         |

## Admin (reference only — not user-callable)

`finishOpenCapsule(uint256 nonce, uint256[] randomNumbers)` is the VRF-callback entrypoint used by the keeper to mint the NFT after randomness arrives. Users don't call this; it's listed for completeness.

## Live snapshot at time of writing

- `dailyUsageLimit = 100`
- `capsulePriceUSD = 50` cents ($0.50)
- `getKibbleUsdPrice ≈ 948,723,424,083,878` → ~$0.0009487 per KIBBLE
- Cost per pull: **~527 KIBBLE** (ceiling-rounded, frontend uses `Math.ceil`)
- Max daily spend per wallet at cap: ~52,700 KIBBLE (~$50)
