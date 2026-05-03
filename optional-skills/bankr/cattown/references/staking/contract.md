# RevenueShare Contract Reference

Single-pool, single-reward staking for KIBBLE on Base. Deployed as a UUPS proxy (upgradeable) — always read values like `LOCK_PERIOD()` from chain rather than hardcoding.

Player-facing overview: https://docs.cat.town/economy/staking.

## Addresses

| Chain             | Chain ID | RevenueShare                                   | KIBBLE (ERC-20, 18 decimals)                   |
|-------------------|----------|------------------------------------------------|------------------------------------------------|
| Base mainnet      | 8453     | `0x9e1Ced3b5130EBfff428eE0Ff471e4Df5383C0a1`   | `0x64cc19A52f4D631eF5BE07947CABA14aE00c52Eb`   |
| Base Sepolia      | 84532    | `0x04ef20ca98d65d4c659a805daa57e0ff4b44f46f`   | `0x7C1059Bdcf44BC2BC1452c67e6A50cE1AB69C49C`   |

Legacy (deprecated) staking contract on Base: `0xc3398Ae89bAE27620Ad4A9216165c80EE654eE96` — do not route new stakes there.

## Reward accounting model

Masterchef-style single pool:

```
pendingRewards(user) = (userStaked(user) * (accRewardPerShare - rewardDebt(user))) / 1e18
```

- `accRewardPerShare` is updated on each `depositRevenue(amount, source)` call:
  ```
  accRewardPerShare += (amount * 1e18) / totalActiveStaked
  ```
- `rewardDebt(user)` is snapshotted on every user action (`stake`, `claim`, `claimAndRestake`, `unlock`, `relock`, `unstake`).
- Users in an `unlock()` window are subtracted from `totalActiveStaked`, so they neither earn nor dilute active stakers.

## Write functions

**Unit convention (important and non-obvious):** `stake(uint256)` and `unstake(uint256)` take `amount` in **whole KIBBLE units** (no decimals), not wei. The contract multiplies by `10^18` internally before calling `transferFrom` on the KIBBLE token. To stake 100 KIBBLE, call `stake(100)`. If you pass `100 * 10^18`, the internal multiplication produces `100 * 10^36`, which exceeds any balance and reverts `ERC20: transfer amount exceeds balance`.

The `approve()` call on the KIBBLE token is a separate, standard ERC-20 call and **is** wei-denominated: to allow staking 100 KIBBLE, approve `100 * 10^18 = 100000000000000000000`.

The address that submits a write must be the same address that (a) holds the KIBBLE balance and (b) granted the allowance. If a smart wallet executes the tx, the balance must sit on the smart wallet — not on an attached EOA.

### `stake(uint256 amount)`

`amount` is in **whole KIBBLE units** (not wei). The contract scales it by `10^18` and pulls `amount * 10^18` wei from `msg.sender` via `transferFrom` on the KIBBLE token (requires prior ERC-20 approval of at least that many wei). Emits `Staked(indexed address user, uint256 amount)`.

Reverts if:
- Allowance on KIBBLE < `amount * 10^18` → `ERC20: transfer amount exceeds allowance`
- Balance on KIBBLE < `amount * 10^18` → `ERC20: transfer amount exceeds balance` (most common cause: caller wei-encoded the argument)

### `unstake(uint256 amount)`

`amount` is in **whole KIBBLE units** (not wei). The contract transfers `amount * 10^18` wei of KIBBLE back to `msg.sender`. Emits `Unstaked(indexed address user, uint256 amount)`.

Reverts if:
- `isUnlocking[user] == true` and `block.timestamp < unlockEndTime[user]`
- `amount > getUserStaked(user)` (both compared in whole KIBBLE units)

### `unlock()`

Starts the exit wait. Sets `isUnlocking[user] = true`, `unlockStartTime[user] = block.timestamp`, `unlockEndTime[user] = block.timestamp + LOCK_PERIOD()`. Removes the user's stake from `totalActiveStaked`. Emits `UnlockInitiated(indexed address user, uint256 unlockEndTime)`.

### `relock()`

Cancels an in-progress unlock. Sets `isUnlocking[user] = false` and returns the user to `totalActiveStaked`. Emits `Relocked(indexed address user, uint256 amount)`.

### `claim()`

Transfers `pendingRewards(user)` KIBBLE to `msg.sender` and updates `rewardDebt`. Emits `Claimed(indexed address user, uint256 amount)`.

### `claimAndRestake()`

Computes `pendingRewards(user)` and adds it directly to the user's stake (no token transfer out, no ERC-20 approval needed). Emits `ClaimedAndRestaked(indexed address user, uint256 restakedAmount, uint256 totalStakedNow)`.

### Admin / owner-only

| Function                                                           | Notes                                  |
|--------------------------------------------------------------------|----------------------------------------|
| `depositRevenue(uint256 amount, string source)`                    | Cat Town backend only; updates `accRewardPerShare`. Emits `RevenueDeposited(string source, uint256 depositTimestamp, uint256 depositAmount, uint256 newAccRewardPerShare)`. `source` ∈ `"fishing"`, `"gacha"`, `"daycare"` (daycare not yet active). |
| `setKibbleToken(address)`                                          | Owner only.                            |
| `recoverERC20(address tokenAddress, uint256 tokenAmount)`          | Owner only; for stuck non-KIBBLE tokens. |
| `initialize(address _owner, address _kibbleToken)`                 | Proxy init; not callable post-deploy.  |
| `upgradeToAndCall(address newImplementation, bytes data)`          | UUPS upgrade; owner only.              |
| `transferOwnership(address newOwner)` / `renounceOwnership()`      | Standard OZ Ownable.                   |

## Read functions

KIBBLE-denominated values are returned in **whole KIBBLE units** (consistent with the unit used by `stake`/`unstake`, NOT wei).

| Call                                           | Returns / unit                | Meaning                                                        |
|------------------------------------------------|-------------------------------|----------------------------------------------------------------|
| `getUserStaked(address user)`                  | whole KIBBLE                  | Currently staked KIBBLE (same as `userStaked(user)`).          |
| `userStaked(address)`                          | whole KIBBLE                  | Raw mapping read; equivalent to `getUserStaked`.               |
| `pendingRewards(address user)`                 | whole KIBBLE                  | Claimable KIBBLE right now.                                    |
| `getPoolShareFraction(address user)`           | fraction × 1e18               | User's share of the active pool, scaled by 1e18.               |
| `isUnlocking(address)`                         | `bool`                        | True iff user has an open `unlock()` they haven't closed.      |
| `unlockStartTime(address)`                     | unix seconds                  | When `unlock()` was called.                                    |
| `unlockEndTime(address)`                       | unix seconds                  | When `unstake()` becomes callable.                             |
| `getTotalStaked()` / `totalStaked`             | whole KIBBLE                  | All staked KIBBLE (includes users currently unlocking).        |
| `getTotalActiveStaked()` / `totalActiveStaked` | whole KIBBLE                  | KIBBLE earning rewards right now.                              |
| `accRewardPerShare()`                          | accumulator × 1e18            | Global reward accumulator.                                     |
| `rewardDebt(address)`                          | accumulator × 1e18            | Per-user snapshot used in the `pendingRewards` formula.        |
| `LOCK_PERIOD()`                                | seconds                       | Unlock wait duration. Read live — contract is upgradeable.     |
| `kibbleToken()`                                | `address`                     | KIBBLE ERC-20 address.                                         |
| `owner()`                                      | `address`                     | Current owner (Cat Town deployer).                             |

Note: the contract's actual KIBBLE holdings read via `balanceOf(revenueShare)` on the KIBBLE token contract are in wei as usual — that's a standard ERC-20 call, not a RevenueShare read. If you want a sanity check, `balanceOf(revenueShare) / 10^18` should approximately equal `getTotalStaked()` (plus some buffer for reward deposits that have been pushed but not yet claimed).

## Events

| Event                                                                                        | When                               |
|----------------------------------------------------------------------------------------------|------------------------------------|
| `Staked(indexed address user, uint256 amount)`                                               | On `stake`.                        |
| `Unstaked(indexed address user, uint256 amount)`                                             | On `unstake`.                      |
| `UnlockInitiated(indexed address user, uint256 unlockEndTime)`                               | On `unlock`.                       |
| `Relocked(indexed address user, uint256 amount)`                                             | On `relock`.                       |
| `Claimed(indexed address user, uint256 amount)`                                              | On `claim`.                        |
| `ClaimedAndRestaked(indexed address user, uint256 restakedAmount, uint256 totalStakedNow)`   | On `claimAndRestake`.              |
| `RevenueDeposited(string source, uint256 depositTimestamp, uint256 depositAmount, uint256 newAccRewardPerShare)` | On `depositRevenue` (backend only). Watch this to detect the weekly drops. |
| `OwnershipTransferred(indexed address previousOwner, indexed address newOwner)`              | Standard OZ Ownable.               |
| `Initialized(uint64 version)` / `Upgraded(indexed address implementation)`                   | Proxy lifecycle.                   |

## Transaction-building checklist

For any stake flow, before constructing calldata:

1. Keep two values straight: `human_amount` (e.g. `100`, the whole-KIBBLE count) and `wei_amount = human_amount * 10^18` (for the ERC-20 approval only).
2. Pass `stake(human_amount)` — the whole-KIBBLE integer. **Not** wei.
3. Read `allowance(user, revenueShare)` on the KIBBLE token. If `< wei_amount`, include an `approve(revenueShare, wei_amount)` tx first.
4. Read `balanceOf(user)` on the KIBBLE token (standard ERC-20, returns wei). Confirm `wei_amount ≤ balanceOf(user)`. Don't submit a tx that will revert.
5. Confirm the tx sender address equals the user address holding KIBBLE. A smart-wallet / relay setup that changes `msg.sender` will cause reverts even when the "user" has tokens.
6. For `unstake`: read `isUnlocking(user)` and `unlockEndTime(user)`. Abort with a clear error if the wait isn't over. Note `unstake` takes whole KIBBLE too.
7. For any write, `getUserStaked(user)` and `pendingRewards(user)` are cheap reads that help you give an accurate pre-tx preview — they also return whole KIBBLE.
