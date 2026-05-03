---
name: stakr-protocol
description: Interact with the Stakr protocol — ERC-4626 vaults with multi-reward staking. Use when the user or an agent needs to work with Stakr vaults, add or modify rewards (addRewardToken, addRewards, modifyRewardToken, modifyReward), create or manage an "agent vault" or "own vault", fund incentive programs, configure reward schedules, or integrate Stakr in scripts or tooling. Prefer this skill whenever Stakr, StakrVault, rewards, staking, or vault ownership is mentioned.
---

# Stakr Protocol — Agent Overview

This skill gives agents the context to interact with the **Stakr** protocol: ERC-4626 tokenized vaults with **multi-reward staking** for any ERC-20 token. Use it when building integrations, scripts, or tooling that create vaults, add rewards, modify reward schedules, or let an agent operate its "own" vault.

---

## Protocol at a Glance

- **StakrVault**: Single-asset ERC-4626 vault. Users deposit underlying, get shares; they can **lock** shares to earn multiple reward tokens over configurable windows.
- **StakrVaultFactory**: Deploys vaults and holds protocol fee configuration. One factory per chain.
- **Rewards**: Up to 25 reward tokens per vault. Each reward has `startTime`, `endTime`, and total `amount`. Distribution is linear over the window; logic is Masterchef-style (accumulated rewards per share).
- **Ownership**: A vault can have an `owner` (address that can add/modify rewards) or `address(0)` for permissionless reward addition.

When an agent is said to have its "own vault", it means: the agent (or a controlled EOA/contract) is the vault **owner**, so it can call `addRewardToken` and `modifyRewardToken` to fund and adjust rewards without third-party permission.

---

## Emphasis: Adding and Modifying Rewards (Agent-Owned Vaults)

Agents that operate their own vault will use these two functions most:

### 1. `addRewardToken(token, amount, settings)`

**Purpose**: Start a new reward program for a given ERC-20 token.

- **Caller**: Vault **owner** (or anyone if `owner() == address(0)`).
- **Parameters**:
    - `token`: ERC-20 reward token address. Cannot be the vault’s share token `address(vault)`.
    - `amount`: Total amount of `token` to distribute. Tokens are pulled from `msg.sender`; protocol may take a fee (see factory `feeOnAddReward`).
    - `settings`: `Settings{ startTime, endTime }`. Distribution is linear from `startTime` to `endTime`; both must be in the future and `startTime < endTime`.
- **Effects**: Registers the reward, pulls tokens (minus fee) into the vault, and emits `AddReward`. Rewards cannot be withdrawn once added; they can only be modified (extended or topped up) via `modifyRewardToken`.
- **Limits**: No duplicate reward token; vault cannot have more than 25 active rewards.

Use this when the agent wants to **create a new reward** (e.g. launch an incentive program on its vault).

### 2. `modifyRewardToken(token, amount, settings)`

**Purpose**: Add more amount and/or extend (or reschedule) an existing reward.

- **Caller**: Same as `addRewardToken` (vault owner or permissionless if owner is zero).
- **Parameters**:
    - `token`: Address of an **already active** reward token.
    - `amount`: **Additional** amount of `token` to add. Pulled from `msg.sender`; fee may apply. Cannot reduce existing amount.
    - `settings`: `Settings{ startTime, endTime }`. Rules:
        - If the reward has **not** yet ended (`currentTime <= reward.settings.endTime`): you can only **extend** `endTime` (and add more amount). `startTime` cannot be changed.
        - If the reward **has** ended (`currentTime > reward.settings.endTime`): you can set a **new** window (`startTime`, `endTime`) and add amount; `accRewardsPerShare` is reset.
- **Effects**: Increases `remainingAmount` (and total amount) by `amount`, updates `endTime` (and possibly `startTime` if reward had ended), pulls tokens from `msg.sender`, and emits `ModifyReward`.

Use this when the agent wants to **top up** an existing reward or **extend** the distribution period (or reschedule after it has ended).

**Summary for agents**:

- New reward → `addRewardToken(token, amount, settings)`.
- More reward or longer duration (or new window after end) → `modifyRewardToken(token, amount, settings)`.
- Ensure the vault has been created via the factory and the agent (or its controlled address) is the vault owner to call these.

## Executing Transactions via Bankr

To submit any Stakr call (vault creation, `addRewardToken`, `modifyRewardToken`), first encode calldata, then submit the transaction via the Bankr wallet API.

Use a natural-language Bankr agent prompt:

```bash
bankr agent prompt "Call addRewardToken on vault 0x... with token 0x... amount 1000 USDC starting tomorrow for 7 days"
```

Or submit raw encoded calldata directly:

```bash
bankr wallet submit --to <vault-address> --data <encoded-calldata> --chain base
```

For calldata encoding help, see the [vault API reference](references/vault-api.md).

### 3. Streaming rewards (continuous incentives)

Agents can **stream** rewards over time instead of funding one large window up front:

- **Pattern**: Call `addRewardToken` once to start a reward (short amount if desired). Then call **`modifyRewardToken`** repeatedly to **add more amount** and **extend `endTime`**. Each call tops up the reward and pushes the end of the distribution window forward.
- **Why**: This avoids locking a huge amount for a long period. The agent (or a script/cron) can fund the vault in chunks and extend the window as needed, effectively creating a continuous reward stream.
- **Rules**: While the reward is active (`currentTime <= reward.settings.endTime`), `modifyRewardToken` only allows extending `endTime` and adding `amount`; `startTime` cannot be changed. After the reward has ended, the agent can set a brand‑new window with `modifyRewardToken` (new `startTime` and `endTime`) and keep streaming.

Use streaming when the agent wants to fund incentives on an ongoing basis (e.g. weekly top‑ups, or extending the program as budget allows) rather than committing to a single long window.

---

## Creating an Agent-Owned Vault

1. Get the **StakrVaultFactory** address for the chain.
    - On **Base mainnet**, the factory is deployed at `0x7Ef55108fa37472296DA59D2287FdA92cd21A0d0` ([view on BaseScan](https://basescan.org/address/0x7Ef55108fa37472296DA59D2287FdA92cd21A0d0)).
    - For an example Stakr vault implementation on Base, see `0x93125009209e23fBAFf2B78712029F7A7CdD23cD` ([example vault on BaseScan](https://basescan.org/address/0x93125009209e23fbaff2b78712029f7a7cdd23cd)).
2. Call **`createStakrVault(underlying, name, symbol, description, owner)`**:
    - `underlying`: ERC-20 underlying asset.
    - `name`, `symbol`: Vault share token name/symbol.
    - `description`: Short metadata (e.g. "Agent incentive vault").
    - `owner`: Set to the agent’s address (or the EOA/contract the agent controls). Use `address(0)` for permissionless reward add/modify by anyone.
3. Use the returned vault address for all subsequent calls (`addRewardToken`, `modifyRewardToken`, etc.).

---

## Core User Flows (for completeness)

- **Deposit only**: `deposit(assets, receiver)` (ERC-4626).
- **Deposit and lock**: `depositAndLock(assets, user)`.
- **Lock existing shares**: `lock(shares, user)` (vault must be approved for the shares).
- **Harvest**: `harvest(user)` to send pending rewards to `user`.
- **Unlock**: `unlock(shares, user)`; then optionally `unlockAndRedeem(shares, receiver)` to redeem shares for underlying.

---

## Settings and Types

- **Settings**: `struct Settings { uint256 startTime; uint256 endTime; }`
- **Validation**: `block.timestamp <= startTime` and `startTime < endTime` for new or rescheduled rewards.
- **Token**: Any ERC-20 except the vault’s share token. The underlying can be used as a reward.

---

For exact function signatures, revert reasons, and events, read the [vault API](references/vault-api.md) in this skill when implementing calls or debugging.
