# Claiming and Managing Hydrex Rewards

Hydrex distributes rewards as **oHYDX** (options HYDX) — a token that can be converted into a veHYDX voting position or liquid HYDX. Rewards accrue across multiple campaigns and are claimed via Merkle proof in a single `claimMultiple` transaction.

**Hydrex Incentive Distributor:** `0x8604d646df5A15074876fc2825CfeE306473dD45` (Base)
**oHYDX Token:** `0xA1136031150E50B015b41f1ca6B2e99e49D8cB78` (Base)

## How It Works

1. Rewards accrue across one or more active campaigns (voting incentives, liquidity incentives, etc.)
2. Fetch your Merkle proofs from the incentives API
3. Call `claimMultiple` on the Incentive Distributor to collect oHYDX
4. Call `exerciseVe` on the oHYDX contract to convert oHYDX into a veHYDX lock position

## Checking Your Rewards

### Natural Language

```bash
bankr agent "Check my Hydrex rewards"
bankr agent "How much oHYDX have I earned on Hydrex?"
bankr agent "Show my unclaimed Hydrex incentives"
```

### Rewards API

Fetch Merkle proofs for an address:

```
GET https://incentives-api.hydrex.fi/campaigns/proofs/YOUR_ADDRESS
```

**Key fields per proof:**

| Field | Description |
|-------|-------------|
| `campaignId` | Unique identifier for the rewards campaign (bytes32) |
| `root` | Merkle root for this batch (bytes32) |
| `amount` | **Total cumulative amount earned** across all batches (wei) |
| `proof` | Array of bytes32 Merkle proof hashes |
| `rewardToken` | Token being distributed (oHYDX: `0xA1136031150E50B015b41f1ca6B2e99e49D8cB78`) |
| `claimedAmount` | Amount already claimed from this campaign (wei) |

**Calculating unclaimed rewards per campaign:**

```
unclaimed = amount - claimedAmount
```

If `unclaimed == 0`, skip — already fully claimed.

## Claiming Rewards

### Natural Language

```bash
bankr agent "Claim my Hydrex rewards"
bankr agent "Claim all my unclaimed Hydrex incentives"
```

### Steps Bankr Executes

1. **Fetch proofs**: `GET https://incentives-api.hydrex.fi/campaigns/proofs/userAddress`
2. **Filter eligible proofs**:
   - Exclude proofs where `amount == claimedAmount` (already fully claimed)
   - Exclude campaigns listed in `warnings`
3. **Construct and submit** `claimMultiple` on the Incentive Distributor

### `claimMultiple` Call

**Function**: `claimMultiple(...)` on Incentive Distributor
**Contract**: `0x8604d646df5A15074876fc2825CfeE306473dD45` (Base)

Each element in the call maps directly to one proof object from the API:

| Parameter | Source | Type |
|-----------|--------|------|
| `campaignId` | `proof.campaignId` | `bytes32` |
| `batchIndex` | `proof.batchIndex` | `uint256` |
| `root` | `proof.root` | `bytes32` |
| `amount` | `proof.amount` | `uint256` (total cumulative, not unclaimed) |
| `proof` | `proof.proof` | `bytes32[]` |

```
Send transaction to 0x8604d646df5A15074876fc2825CfeE306473dD45 on Base calling claimMultiple with the eligible proof data from https://incentives-api.hydrex.fi/campaigns/proofs/[USER_ADDRESS]
```

**Important**: Pass the full `amount` (not the unclaimed delta). The contract tracks claimed amounts internally and only distributes the difference.

**Result**: oHYDX tokens are transferred to the caller's address.

## Converting oHYDX to veHYDX

oHYDX is an options token — exercising it converts it into a locked veHYDX position. You pay a discounted price (in ETH/WETH) and receive a veHYDX NFT.

### Natural Language

```bash
bankr agent "Convert my oHYDX to veHYDX on Hydrex"
bankr agent "Exercise my Hydrex oHYDX rewards into veHYDX"
bankr agent "How much does it cost to exercise my oHYDX on Hydrex?"
```

### `exerciseVe` Call

**Function**: `exerciseVe(uint256 amount, uint256 maxPaymentAmount, address recipient)`
**Contract**: `0xA1136031150E50B015b41f1ca6B2e99e49D8cB78` (oHYDX, Base)

```
Send transaction to 0xA1136031150E50B015b41f1ca6B2e99e49D8cB78 on Base calling exerciseVe with amount [OHYD_AMOUNT_IN_WEI], maxPaymentAmount [MAX_ETH_IN_WEI], recipient [USER_ADDRESS]
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `amount` | Amount of oHYDX to exercise (wei) |
| `maxPaymentAmount` | Maximum ETH willing to pay (wei) — set higher than expected to avoid slippage reverts |
| `recipient` | Address to receive the new veHYDX NFT |

**Before calling:**
1. Check oHYDX balance: `balanceOf(userAddress)` on `0xA1136031150E50B015b41f1ca6B2e99e49D8cB78`
2. Check the current exercise cost (discount rate varies) — ask Bankr or query the contract
3. Approve the oHYDX contract to spend the payment token if required

**Result**: oHYDX is burned, payment is taken, and a new veHYDX NFT (rolling lock) is minted to `recipient`.

## Function Reference

| Function | Contract | Parameters | Returns |
|----------|----------|------------|---------|
| `claimMultiple(...)` | Incentive Distributor | Array of campaign proofs | — |
| `exerciseVe(uint256,uint256,address)` | oHYDX | amount, maxPayment, recipient | tokenId |
| `balanceOf(address)` | oHYDX | user address | uint256 oHYDX balance |

To read oHYDX balance directly — `balanceOf(address)` selector `0x70a08231`, encode user address as 32-byte padded hex and call `eth_call` on `0xA1136031150E50B015b41f1ca6B2e99e49D8cB78`.

## Complete Workflow

```bash
# 1. Check what rewards you've earned
bankr agent "Check my Hydrex rewards"

# 2. Claim all unclaimed oHYDX
bankr agent "Claim my Hydrex oHYDX rewards"

# 3. Check your oHYDX balance
bankr agent "What's my oHYDX balance on Base?"

# 4. Convert oHYDX into veHYDX (locking for voting power + fee earnings)
bankr agent "Exercise my oHYDX into veHYDX on Hydrex"
```

## Implementation Guide for Bankr

When a user requests reward claiming or checking:

### Check Rewards Flow

```
1. GET https://incentives-api.hydrex.fi/campaigns/proofs/userAddress
2. For each proof:
     unclaimed = parseInt(proof.amount) - parseInt(proof.claimedAmount)
3. Filter out: warnings list, unclaimed == 0
4. Sum total unclaimed across all eligible proofs (convert from wei for display)
5. Show: total unclaimed oHYDX, number of campaigns, estimated USD value
```

### Claim Flow

```
1. GET https://incentives-api.hydrex.fi/campaigns/proofs/userAddress
2. Filter eligible proofs (exclude warnings, exclude fully claimed)
3. Build claimMultiple calldata with all eligible proofs:
     - campaignId, batchIndex, root, amount (full cumulative), proof[]
4. Submit to 0x8604d646df5A15074876fc2825CfeE306473dD45 on Base
```

### Exercise Flow

```
1. eth_call balanceOf(userAddress) on 0xA1136031150E50B015b41f1ca6B2e99e49D8cB78 → oHYDX balance
2. Get current exercise price from contract or display to user
3. Call exerciseVe(oHYDXBalance, maxPaymentAmount, userAddress) on oHYDX contract
   - Requires ETH payment — confirm with user before submitting
```

## Notes

- **oHYDX is not HYDX** — it's an options token that must be exercised to become veHYDX or HYDX
- **Exercising costs ETH** — the exercise price is a 30% discount to spot HYDX price (set by the protocol)
- **veHYDX from exercising** is a permalock (Type 2), automatically maintained for maximum voting power
- **amount is cumulative** — always pass the full `amount` from the proof, not the delta
