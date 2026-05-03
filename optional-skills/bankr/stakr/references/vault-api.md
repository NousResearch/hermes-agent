# Stakr Vault API Reference

Use this file when implementing calls, writing tests, or debugging. It summarizes the main Stakr contract interfaces and behavior.

---

## StakrVaultFactory

| Function               | Signature                                                                                                                         | Notes                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `createStakrVault`     | `createStakrVault(address underlying, string _name, string _symbol, string _description, address _owner) returns (address vault)` | Deploys a new vault. `_owner` can be `address(0)` for permissionless add/modify rewards. |
| `availableStakrVaults` | `availableStakrVaults(address _underlying) view returns (uint256)`                                                                | Number of vaults for that underlying.                                                    |
| `stakrVaultByIndex`    | `stakrVaultByIndex(address _underlying, uint256 _index) view returns (address)`                                                   | Vault address at index.                                                                  |
| `setProtocolFee`       | `setProtocolFee(uint256 _feeOnAddReward, uint256 _feeOnLock, address _feeCollector)`                                              | Requires `FEE_MANAGER_ROLE`.                                                             |

---

## StakrVault — Reward API

### Structs

```solidity
struct Settings {
    uint256 startTime;  // Start of distribution
    uint256 endTime;    // End of distribution
}
```

### addRewardToken

```solidity
function addRewardToken(address token, uint256 amount, Settings calldata _settings) external
```

- **Modifiers**: `nonReentrant`, `checkOwnable` (owner or anyone if owner is `address(0)`).
- **Reverts**: `StakrVaultInvalidReward()` if `token == address(this)`, reward already active for `token`, or `_rewardsCount >= MAX_REWARDS_COUNT` (25). `StakrVaultInvalidSettings()` if `block.timestamp > startTime` or `startTime >= endTime`.
- **Side effects**: Pulls `token` from `msg.sender` (after optional fee to factory `feeCollector`). Cannot withdraw rewards later; use `modifyRewardToken` to add amount or extend.

### modifyRewardToken

```solidity
function modifyRewardToken(address token, uint256 amount, Settings calldata _settings) external
```

- **Modifiers**: `nonReentrant`, `checkOwnable`.
- **Reverts**: `StakrVaultInvalidReward()` if reward for `token` is not active. `StakrVaultInvalidSettings()` if `block.timestamp > _endTime` or `_startTime >= _endTime`, or (when reward has ended) if `block.timestamp > _startTime`.
- **Behavior**:
    - If reward **not** ended: only `endTime` can be updated (and extra `amount` added). `startTime` unchanged.
    - If reward **ended**: new `startTime`/`endTime` allowed; `accRewardsPerShare` reset.
- **Side effects**: Pulls `amount` of `token` from `msg.sender` (fee may apply). Updates `remainingAmount`, `amount`, and `settings.endTime` (and optionally `startTime`).

### Streaming rewards (for agents)

Agents can **stream** rewards by using `modifyRewardToken` repeatedly instead of funding one long window with `addRewardToken`:

- Start with **`addRewardToken(token, amount, settings)`** (e.g. a short initial window).
- Then call **`modifyRewardToken(token, amount, settings)`** on a schedule (e.g. weekly): pass **additional** `amount` and a **new `endTime`** that extends the distribution window. The reward stays active and linear; stakers see a continuous stream.
- **While the reward is active**: only `endTime` can be updated (and extra amount added). **After the reward has ended**: a new full window (`startTime`, `endTime`) can be set and more amount added, then the cycle can repeat.

This pattern lets agents fund incentives in chunks and extend the program over time without locking a large lump sum upfront.

### Execution via Bankr wallet

Use this file to build and validate function calldata (`createStakrVault`, `addRewardToken`, `modifyRewardToken`, etc.), then submit the transaction with Bankr wallet:

```bash
bankr wallet submit --to <vault-address-or-factory> --data <encoded-calldata> --chain base
```

For a natural-language execution flow, see the skill guide section in [../SKILL.md](../SKILL.md).

### Other reward-related

| Function           | Signature                                     | Notes                                                                  |
| ------------------ | --------------------------------------------- | ---------------------------------------------------------------------- |
| `rewardsCount`     | `rewardsCount() view returns (uint256)`       | Number of active rewards (max 25).                                     |
| `updateStakrVault` | `updateStakrVault() external`                 | Updates internal accRewardsPerShare state; anyone can call.            |
| `rewards`          | `rewards(address) view returns (RewardToken)` | Reward config per token.                                               |
| `rewardDebt`       | `rewardDebt(bytes32) view returns (uint256)`  | Per-user reward debt; ID = `keccak256(abi.encodePacked(user, token))`. |

---

## StakrVault — Lock / Harvest / ERC-4626

| Function            | Signature                                           | Notes                                                                                                    |
| ------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `lock`              | `lock(uint256 _shares, address _user)`              | Locks shares on behalf of `_user`; caller must transfer shares to vault first (or use `depositAndLock`). |
| `unlock`            | `unlock(uint256 _shares, address _user)`            | Unlocks and sends shares back to `msg.sender`.                                                           |
| `depositAndLock`    | `depositAndLock(uint256 assets, address _user)`     | Deposit underlying and lock resulting shares for `_user`.                                                |
| `unlockAndRedeem`   | `unlockAndRedeem(uint256 shares, address receiver)` | Unlock and redeem shares for underlying to `receiver`.                                                   |
| `harvest`           | `harvest(address _user)`                            | Sends all pending rewards to `_user`.                                                                    |
| `lockedShares`      | `lockedShares(address) view returns (uint256)`      | Locked share balance per user.                                                                           |
| `owner`             | `owner() view returns (address)`                    | Address allowed to add/modify rewards; `address(0)` = permissionless.                                    |
| `transferOwnership` | `transferOwnership(address _newOwner)`              | Owner only.                                                                                              |

Standard ERC-4626: `deposit`, `redeem`, `mint`, `withdraw`, `asset`, `totalAssets`, etc.

---

## Events

- `AddReward(address indexed token, uint256 amount, uint256 startDate, uint256 endDate)`
- `ModifyReward(address indexed token, uint256 amount, uint256 startDate, uint256 endDate)`
- `DeleteReward(address indexed token)` — when a reward is fully harvested.
- `Lock(address indexed user, uint256 amount, uint256 lastLockedTime)`
- `Unlock(address indexed user, uint256 amount, uint256 lastLockedTime)`
- `Harvest(address indexed user, address token, uint256 amount)`

---

## Errors

- `StakrVaultInvalidReward()` — Invalid or duplicate reward token, or max rewards reached.
- `StakrVaultInvalidSettings()` — Invalid start/end times.
- `StakrVaultInsufficientBalance()` — Unlock amount exceeds locked balance.
- `StakrVaultNotAllowed()` — Caller is not owner (when owner is set).
- `StakrVaultMathError()` — Internal math revert (e.g. trySub failure).
