# Sell Items — V2 minter path

Cat Town's vendor flow: players sell treasures and collectibles to the SellItems contract for KIBBLE at a fixed catalog price minus a 5% merchant fee. This skill only supports selling items minted by the **V2 minter** — legacy V1-minted items are out of scope.

Player-facing overview: https://docs.cat.town/shops/sell-items.

## Addresses (Base, chain 8453)

| Contract            | Address                                       | Role                             |
|---------------------|-----------------------------------------------|----------------------------------|
| SellItems           | `0x49936db5Dcbc906D682CFa2dcfAb0788e3ee5808`  | Receives NFTs, pays out KIBBLE   |
| CentralizedMinter V2 | `0x7b65ec82cB4600Bc1dCc5124a15594976f19eA14` | Source of sellable NFTs (ERC-1155) — **use this everywhere in calldata** |
| KIBBLE token        | `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`  | Payout token                     |

Legacy V1 minter (`0x408C186C1fFCc78592cbdae9B04da8a64A975550`) also exists. Do **not** include V1-minted items in sell calldata for this revision — filter them out in the preflight.

ABI: `/abi/internal/Vendor/SellItemsAbi.json`.

## What's sellable

- `itemType == "Treasure"` ✓
- `itemType == "Collectible"` ✓ (gacha pulls end up here — Fern, Cactus, etc.)
- Everything else (Cosmetic, Fish, Equipment) — **not sellable** via this contract.
- The inventory API filters with `hasSellValue=true`, which implicitly drops unsellable types.

## Catalog value — `sellValue` (cents, USD)

The item-truth catalog (`GET /v2/items/master?limit=1000`, public, no auth) lists a `sellValue` on every sellable item. **Units: US cents, not KIBBLE, not wei.** This matches the frontend's internal mapping to `centValue`.

| Example item            | `sellValue` | USD    |
|-------------------------|-------------|--------|
| Fern (Common Collectible, Plant Minis)  | 10    | $0.10 |
| Blooming Cactus (Common Collectible)    | 25    | $0.25 |
| Solar Pearl (Uncommon Treasure, Sun)    | (varies)| —     |
| Solar Ring (Legendary Treasure)         | 25,000| $250.00|
| Diamond (Epic Treasure)                 | 10,000| $100.00|

### Convert catalog value to KIBBLE for display

Use the Kibble Price Oracle (`0xE97B7ab01837A4CbF8C332181A2048EEE4033FB7`, scale `10^18`):

```
usd_dollars      = sellValue / 100
kibble_value     = usd_dollars / (rawKibbleUsdPrice / 10^18)
                 = (sellValue * 10^16) / rawKibbleUsdPrice          // BigInt-safe
payout_after_tax = kibble_value * 0.95                              // 5% merchant fee
```

Live example: Fern `sellValue=10` → $0.10 → ~105 KIBBLE → ~100 KIBBLE after the 5% tax.

### NFT trait fallback (for freshly minted items)

When the agent is holding a *minted* NFT in hand (e.g. right after a gacha pull), the NFT's metadata traits carry a `Sell Value (KIBBLE)` field directly — already denominated in KIBBLE, no oracle conversion needed. Prefer this when available; fall back to the catalog formula above if the trait isn't set.

## Write path — `sellMultipleNFTsToContract`

```solidity
function sellMultipleNFTsToContract(
    address[] nftContracts,   // one entry per item — always the V2 minter for this skill
    uint256[] tokenIds,       // the item ids being sold
    uint256[] amounts         // ERC-1155 quantities (1 per token id for our usage)
) external
```

All three arrays must be the same length (`InputArrayLengthMismatch` revert otherwise). The frontend caps at 25 items per call — mirror that.

### Preflight

1. **Approval.** The seller must approve the SellItems contract to transfer their NFTs:
   ```
   V2_Minter.setApprovalForAll(sellItemsContract, true)
   ```
   One-time per wallet. Check with `isApprovedForAll(seller, sellItemsContract)` on the V2 minter before submitting — if already true, skip the approval tx.

2. **Filter to V2-minted items.** For each item the user wants to sell, confirm the source nftContract is the V2 minter (`0x7b65ec82cB4600Bc1dCc5124a15594976f19eA14`). Drop anything sourced from V1; surface that explicitly ("skipping 2 legacy items").

3. **Ownership.** The seller must own each token id with amount ≥ 1 (`balanceOf(seller, tokenId)`). Revert `InsufficientNFTBalance` otherwise.

4. **Vendor liquidity.** The contract pays out KIBBLE from its own balance. Check `KIBBLE.balanceOf(sellItemsContract)` — if the vendor is short, the sell reverts `KibbleTransferFailed`. The frontend gates the whole sell UI when this happens and shows "Vendor is out of KIBBLE!" — agents should surface the same.

### Effects

- All `(nftContract, tokenId, amount)` tuples are pulled from `msg.sender` via ERC-1155 `safeTransferFrom`.
- Total payout = `sum(item.sellValue_in_KIBBLE) * (1 − taxRateBps/10000)`, transferred in one KIBBLE transfer.
- The contract reads `taxRateInBps()` — current value **500** (5%), so effective payout multiplier is **0.95**.

Emits `NFTPurchased(address indexed seller, uint256 tokenId, uint256 amount)` — one event per item sold.

### Known reverts

| Error                        | Cause                                         |
|------------------------------|-----------------------------------------------|
| `InputArrayLengthMismatch`   | The three arrays differ in length             |
| `InsufficientNFTBalance`     | Seller doesn't own one of the token ids       |
| `NFTContractNotSupported`    | One of the nftContracts isn't whitelisted     |
| `KibbleTransferFailed`       | Vendor KIBBLE balance too low for the payout  |

Standard ERC-1155 approval errors also apply if `setApprovalForAll` wasn't run first.

## Inventory API — finding sellable items

```
GET https://api.cat.town/v2/inventory/<address>/paginated?hasSellValue=true&sortBy=kibble&sortOrder=desc
```

Public, no auth. Returns paginated items with `hasSellValue=true` filter. Sort by `kibble` to put highest-value items first — matches what the frontend does when you open the vendor modal.

Use this as the "what can this user sell" lookup. Cross-reference with the source contract to drop V1-minted items.

## Batch sell recipe (V2 only)

```
1. Check approval:
     approved = V2_Minter.isApprovedForAll(seller, sellItemsContract)
     if not approved:
         submit V2_Minter.setApprovalForAll(sellItemsContract, true)

2. Pull inventory:
     GET /v2/inventory/<seller>/paginated?hasSellValue=true&sortBy=kibble&sortOrder=desc
     Filter to items whose nftContract == V2_Minter.

3. Slice to ≤ 25 items (contract accepts more, UI caps at 25 — mirror that).

4. Compute expected payout:
     total_kibble = sum(item_value_in_kibble for each item)
     payout       = total_kibble * 0.95

5. Confirm with user ("You'll get ~N KIBBLE after the 5% fee.").

6. Submit sellMultipleNFTsToContract(
      nftContracts = [V2_Minter] * len(items),
      tokenIds     = [item.tokenId for item in items],
      amounts      = [1] * len(items)
   )

7. After confirmation, refetch inventory + KIBBLE balance.
```

## Live snapshot (at time of writing)

- Tax rate: **5%** (`taxRateInBps() = 500`).
- Batch cap (frontend): **25 items/tx** (no onchain limit; mirror the UI).
- Fern payout example: `10 ¢ → ~105 KIBBLE → ~100 KIBBLE` after tax.
- Vendor KIBBLE balance should be checked live — if the vendor is drained, sells revert.
