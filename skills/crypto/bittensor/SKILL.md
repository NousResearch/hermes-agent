---
name: bittensor-wallet
description: Manage Bittensor wallets, check balances, stake TAO tokens, and query subnet info using btcli
version: 1.0.0
author: het4rk
platforms:
  - macos
  - linux
tags:
  - Crypto
  - Bittensor
  - TAO
  - Wallet
  - Staking
  - Subnet
requires_toolsets:
  - terminal
---

# Bittensor Wallet Management

Manage Bittensor wallets, monitor TAO balances, view staking positions, and query subnet information using `btcli` — the official Bittensor CLI.

## When to Use

- Check TAO wallet balance (coldkey or hotkey)
- View staking positions across subnets
- List active subnets and their metagraphs
- Query miner/validator emissions
- Transfer TAO between wallets
- Monitor subnet health and registrations

## Quick Reference

| Task | Command |
|------|---------|
| Check balance | `btcli wallet balance --wallet.name <name>` |
| View stake | `btcli stake show --wallet.name <name>` |
| List subnets | `btcli subnet list` |
| Query metagraph | `btcli subnet metagraph --netuid <N>` |
| Transfer TAO | `btcli wallet transfer --dest <ss58addr> --amount <TAO>` |
| Check emissions | `btcli subnet metagraph --netuid <N>` (see Emission column) |

## Procedure

### Check Wallet Balance

```bash
# List all wallets
btcli wallet list

# Check balance for a specific wallet
btcli wallet balance --wallet.name default

# Check balance for all wallets
btcli wallet balance --all
```

**What to look for:** `Free Balance` (spendable TAO) vs `Staked Balance` (locked in subnets).

---

### View Staking Positions

```bash
# Show all stakes for a wallet
btcli stake show --wallet.name <wallet_name>

# Show stakes with detailed hotkey info
btcli stake show --wallet.name <wallet_name> --all
```

**Output columns:** Hotkey address, Subnet UID, Stake (TAO), Emission rate (ρ/block).

---

### List Active Subnets

```bash
# List all subnets
btcli subnet list

# Filter to see specific fields (pipe through grep/awk)
btcli subnet list | grep -E "NETUID|Tempo|Emission"
```

**Key columns:** NETUID, Name, Tempo (blocks/epoch), Emission (TAO/block), Neurons registered.

---

### Query Subnet Metagraph

```bash
# Full metagraph for a subnet (e.g., subnet 1)
btcli subnet metagraph --netuid 1

# Metagraph for subnet 18 (cortex.t)
btcli subnet metagraph --netuid 18
```

**Output includes:** UID, Hotkey, Stake, Rank, Trust, Consensus, Incentive, Dividends, Emission, Last Update.

---

### Check Emissions

```bash
# View per-neuron emissions in a subnet
btcli subnet metagraph --netuid <N>

# View subnet-level emission allocations
btcli subnet list
```

Emissions are denominated in ρ (rho) per block. Multiply by ~7200 blocks/day for daily TAO.

---

### Transfer TAO

```bash
# Transfer TAO to another address
btcli wallet transfer \
  --wallet.name <wallet_name> \
  --dest <ss58_destination_address> \
  --amount <TAO_amount>
```

> **Warning:** Double-check the destination SS58 address. TAO transfers are irreversible.

---

## Pitfalls

- **Coldkey vs Hotkey confusion:** Coldkeys hold the actual TAO balance. Hotkeys are used for staking/mining operations. Never expose your coldkey.
- **Network fees:** Each transaction costs a small TAO fee. Ensure you have enough free balance.
- **Metagraph staleness:** The metagraph is updated each tempo (~100 blocks). Data may be up to ~20 minutes old.
- **dTAO subnets:** Subnets with dynamic TAO (dTAO) use subnet-specific tokens (αTokens). Use `btcli subnet metagraph` to see alpha stake vs TAO stake.
- **Registration costs:** Registering a hotkey on a subnet requires TAO burn or proof-of-work. Check current cost with `btcli subnet register --help`.
- **Wallet path:** Default wallet directory is `~/.bittensor/wallets/`. Use `--wallet.path` to override.

## Verification

After any wallet operation, verify success:

```bash
# Confirm balance after transfer
btcli wallet balance --wallet.name <wallet_name>

# Confirm stake after staking/unstaking
btcli stake show --wallet.name <wallet_name>

# Check transaction on-chain
# Visit: https://taostats.io or https://x.taostats.io
```
