# Cat Town Boutique — contract + KIBBLE oracle reference

The boutique is a fully onchain daily shop on Base. Every day at **00:00 UTC** the contract surfaces **3 items** selected deterministically from the current season's pool. No offchain API is needed — items, prices, stock, and rotation are all readable directly from the Boutique contract.

This doc covers the **Boutique** contract (rotation + item state) and the **KIBBLE price oracle** (for USD conversion — the in-game UI shows KIBBLE only).

## Addresses (Base, chain 8453)

| Contract            | Address                                      |
|---------------------|----------------------------------------------|
| Boutique            | `0xf9843bF01ae7EF5203fc49C39E4868C7D0ca7a02` |
| Kibble Price Oracle | `0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7` |
| KIBBLE token        | `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb` |

## Rotation model

- Each day starts at **00:00 UTC**; `getCurrentDayNumber()` returns days since Unix epoch (`block.timestamp / 86400`).
- Rotation is deterministic from `(dayNumber, currentSeason)` — same day + same season = same 3 items.
- `itemsPerDay()` = **3** (constant).
- Season boundaries follow `GameData.getCurrentSeason()` (see [../world/contract.md](../world/contract.md)); each season has its own pool.
- The matching human-readable doc pages:
  - Top-level shop: https://docs.cat.town/shops/boutique
  - Spring: https://docs.cat.town/boutique/spring-fashion
  - Summer: https://docs.cat.town/boutique/summer-fashion
  - Autumn: https://docs.cat.town/boutique/autumn-fashion
  - Winter: https://docs.cat.town/boutique/winter-fashion

## Primary read — `getTodaysRotationDetails()`

Returns today's 3 items as `ShopItemView[]` — full details in one call. Selector: `0x36362553`, no args.

### `ShopItemView` fields

| Field              | Type        | Notes                                                                 |
|--------------------|-------------|-----------------------------------------------------------------------|
| `itemId`           | `uint256`   | Unique id for the shop item                                           |
| `traitNames`       | `string[]`  | Parallel array of trait keys, e.g. `["Name","Rarity","Image","Slot","Shiny"]` |
| `traitValues`      | `string[]`  | Parallel array of values in the same order                            |
| `paymentToken`     | `address`   | Always the KIBBLE token                                               |
| `price`            | `uint256`   | KIBBLE in **wei** (18 decimals) — divide by `10^18` for display       |
| `stockRemaining`   | `uint256`   | Units still purchasable. `0` → sold out                               |
| `totalPurchased`   | `uint256`   | Units sold so far                                                     |
| `maxSupply`        | `uint256`   | Total ever available. `type(uint256).max` → uncapped                  |
| `startTime`        | `uint64`    | Unix seconds (0 = always available)                                   |
| `endTime`          | `uint64`    | Unix seconds (0 = no end)                                             |
| `availableSeasons` | `uint8`     | Bitmask: `1=Spring`, `2=Summer`, `4=Autumn`, `8=Winter`               |
| `isActive`         | `bool`      | Enabled by admin                                                      |
| `isPurchasableNow` | `bool`      | Passes time + season gates                                            |
| `isInTodaysRotation` | `bool`    | In today's 3-item set                                                 |

### Parsing the trait arrays

`traitNames` and `traitValues` are parallel. Real trait keys on a live boutique item:

| Trait key     | Example value                                        | Notes                                                      |
|---------------|------------------------------------------------------|------------------------------------------------------------|
| `Item Name`   | `"White Longsleeve"`                                 | Display name                                               |
| `Rarity`      | `"Rare"`                                             | `Common` / `Uncommon` / `Rare` / `Epic` / `Legendary`      |
| `Item Type`   | `"Cosmetic"`                                         | Almost always `Cosmetic` for boutique                      |
| `Source`      | `"Boutique"`                                         | Distinguishes from `Fishing`/`Gacha` in a joined view      |
| `Slot`        | `"Body"`                                             | `Hat` / `Body` / `Eyewear` / `Companion` / etc.            |
| `Sprite`      | `"white-longsleeve"`                                 | Internal asset id                                          |
| `imageUrl`    | `https://cdn.cat.town/nft/equipment/body/...`        | Display image                                              |
| **`Collection`** | `"Spring Fashion"`                                | **Collection label** — use this to tell the user which collection is currently rotating |
| `Flavor Text` | `"Clean and crisp like fresh spring linens."`        | Optional color                                             |
| `Sell Value`  | `"0"`                                                | Usually 0 for boutique (these aren't meant to be resold)   |
| `coreId`      | `"cmlz9n8f30008kz04flhruq6t"`                        | Internal database id                                       |

Boutique metadata is **onchain via the trait arrays** — don't cross-reference `/v2/items/master`. The `ShopItemView.traitNames`/`traitValues` are the source of truth.

## Other useful reads

| Function                              | Returns                       | Notes                                          |
|---------------------------------------|-------------------------------|------------------------------------------------|
| `getTodaysRotation()`                 | `uint256[]`                   | Just today's 3 item ids (cheaper)              |
| `getCurrentDayNumber()`               | `uint256`                     | Days since Unix epoch                          |
| `getCurrentSeason()`                  | `uint8`                       | `0=Spring, 1=Summer, 2=Autumn, 3=Winter`       |
| `getShopItem(itemId)`                 | `ShopItemView`                | One item by id                                 |
| `getAllShopItems()`                   | `ShopItemView[]`              | Full catalog, active + inactive                |
| `getItemsBySeason(season)`            | `ShopItemView[]`              | Season-specific pool                           |
| `previewRotationForDay(day, season)`  | `uint256[]`                   | Future rotation preview (deterministic)        |
| `getItemStock(itemId)`                | `(max, purchased, remaining)` | Stock only                                     |
| `dailyRotationEnabled()`              | `bool`                        | Is daily rotation on (expected: true)          |
| `itemsPerDay()`                       | `uint8`                       | Currently 3                                    |
| `defaultPaymentToken()`               | `address`                     | KIBBLE                                         |

## KIBBLE → USD conversion

### Oracle reads

| Function              | Selector     | Returns                        | Scale       |
|-----------------------|--------------|--------------------------------|-------------|
| `getKibbleUsdPrice()` | `0x00cbfbce` | `uint256` USD per 1 KIBBLE     | **× 10^18** |
| `getEthUsdPrice()`    | `0xa0a8045e` | `uint256` USD per 1 ETH        | × 10^8 (Chainlink) |
| `getKibbleEthPrice()` | `0x47bb71e5` | `uint256` ETH per 1 KIBBLE     | × 10^18     |

**Watch the scale mismatch:** `getKibbleUsdPrice()` is `10^18`, but `getEthUsdPrice()` is `10^8`. Easy to mix up — use the right divisor per call.

### Formula

Boutique `price` is in KIBBLE wei (18 decimals). Oracle returns USD × `10^18` per 1 KIBBLE:

```
kibble_human    = price / 10^18                           # KIBBLE count
usd_per_kibble  = rawKibbleUsdPrice / 10^18               # USD per 1 KIBBLE
usd_value       = kibble_human * usd_per_kibble
                = (price * rawKibbleUsdPrice) / 10^36     # BigInt-safe form
```

For integer cents: `usd_cents = (price * rawKibbleUsdPrice) / 10^34`.

### Live example (captured during writing)

- `getKibbleUsdPrice()` = `948,723,424,083,878` → **$0.0009487 per KIBBLE**
- 1,000 KIBBLE ≈ $0.95
- 10,000 KIBBLE ≈ $9.49
- 100,000 KIBBLE ≈ $94.87

The oracle tracks KIBBLE's real market price; re-read at least every few minutes if you care about accuracy.

## Response pattern — "what's in the boutique today?"

1. Read in parallel: `getTodaysRotationDetails()` (single call, 3 items) and `getKibbleUsdPrice()`.
2. For each `ShopItemView`:
   - Parse `traitNames`/`traitValues` into a dict → pull `Name`, `Rarity`, `Slot`.
   - `kibble_price = price / 10^18`
   - `usd_price = (price * rawKibbleUsdPrice) / 10^36`
   - Stock: if `stockRemaining == 0` → **"Sold Out"**; otherwise format as **`"{stockRemaining} of {maxSupply} remaining"`** — stockRemaining first, maxSupply second. The order matters: `stockRemaining` ≤ `maxSupply` always, so if the first number ever exceeds the second you've swapped them. Reread the struct fields if unsure.
3. Sort with the big-ticket order: **rarity DESC** (Legendary → Common), then **KIBBLE price DESC**, then name ASC.
4. Open the reply with the current season, and end with a link to the matching `docs.cat.town/boutique/...-fashion` page.

### Example response (real data from today's rotation)

> **Boutique today — Spring Fashion collection (Day 20566):**
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

## Notes

- **Purchase flow is out of scope for this revision.** It involves `approve(boutique, price_wei)` on KIBBLE, then `purchaseItem(itemId)` which mints an NFT and returns `mintedTokenId`. A future skill update will add the write path (watch the integer-vs-wei convention on `purchaseItem`'s `itemId` — likely raw uint256 not scaled).
- **Caching:** `getTodaysRotationDetails()` is stable within a UTC day — cache freely. The oracle moves with market — 1–5 min cache is reasonable.
- **Future rotations:** `previewRotationForDay(day, season)` supports "what's in the boutique tomorrow?" queries without waiting.
- **Season mismatch:** if `GameData.getCurrentSeason()` disagrees with `Boutique.getCurrentSeason()`, trust Boutique's for rotation questions (they should match, but boutique may lag a block).
