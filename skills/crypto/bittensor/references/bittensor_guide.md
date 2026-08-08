# Bittensor Reference Guide

## Architecture Overview

Bittensor is a decentralized network that incentivizes the production and validation of machine intelligence. It runs on its own blockchain (Subtensor) and uses **TAO** as its native token.

### Core Components

| Component | Description |
|-----------|-------------|
| **Subtensor** | The Bittensor blockchain (Substrate-based) |
| **Subnet** | A specialized ML task network (netuid 0–64+) |
| **Miner** | Produces intelligence/outputs for a subnet |
| **Validator** | Scores miners and sets weights |
| **TAO** | Native token used for staking and emissions |

### Network Topology

```
Bittensor Network
├── Root Network (netuid 0) — controls subnet emissions
├── Subnet 1 (Text prompting)
├── Subnet 18 (Cortex.t — reasoning)
├── Subnet 19 (Vision)
└── ... (64+ subnets)
```

Each subnet is an independent incentive mechanism where miners compete to perform a specific ML task.

---

## Wallet Structure

Bittensor uses a **dual-key** architecture:

### Coldkey
- Master key that holds your TAO balance
- Used for transfers and high-security operations
- **Never expose the coldkey mnemonic**
- Stored at: `~/.bittensor/wallets/<name>/coldkey`

### Hotkey
- Operational key used for mining/validating
- Can be registered on subnets
- Lower security risk if compromised (no direct TAO access)
- Stored at: `~/.bittensor/wallets/<name>/hotkeys/<hotkey_name>`

### Wallet Directory Structure

```
~/.bittensor/wallets/
└── default/
    ├── coldkey          # encrypted coldkey
    ├── coldkeypub.txt   # public coldkey (SS58 address)
    └── hotkeys/
        ├── default      # default hotkey
        └── validator    # additional hotkeys
```

---

## Staking Mechanics

### How Staking Works

1. TAO is moved from your coldkey to a hotkey (staking)
2. The hotkey registers on a subnet
3. Validators/miners on the subnet earn emissions based on performance
4. Emissions flow back to the hotkey, then can be unstaked to coldkey

### Staking Commands

```bash
# Stake TAO to a hotkey
btcli stake add \
  --wallet.name <coldkey_name> \
  --wallet.hotkey <hotkey_name> \
  --amount <TAO>

# Unstake TAO from a hotkey
btcli stake remove \
  --wallet.name <coldkey_name> \
  --wallet.hotkey <hotkey_name> \
  --amount <TAO>

# View all stakes
btcli stake show --wallet.name <coldkey_name>
```

### Emission Distribution

- The root network (netuid 0) controls how TAO is distributed across subnets
- Within each subnet, validators set weights on miners
- Yuma Consensus converts weights → ranks → emissions
- ~7200 blocks per day × emission_rate = daily TAO earnings

---

## dTAO (Dynamic TAO)

dTAO is Bittensor's mechanism for decentralizing subnet emission control.

### Key Concepts

| Term | Description |
|------|-------------|
| **αToken** | Subnet-specific token (e.g., α18 for subnet 18) |
| **Alpha stake** | TAO staked into a subnet's liquidity pool |
| **τ/α price** | Exchange rate between TAO and the subnet's αToken |
| **Moving price** | EMA of the τ/α price used for emission calculation |

### How dTAO Works

1. Each subnet has its own token (αToken)
2. Stakers buy αTokens by depositing TAO into the subnet pool
3. The subnet's emission share is proportional to its αToken market cap
4. Validators stake αTokens (not raw TAO) to participate in a subnet

### dTAO Commands

```bash
# View subnet token info
btcli subnet list  # shows dTAO price and market cap columns

# Stake into a subnet's alpha pool
btcli stake add --netuid <N> --wallet.name <name> --amount <TAO>
```

---

## Common Workflows

### Onboarding a New Miner

1. Create a wallet: `btcli wallet new_coldkey --wallet.name miner1`
2. Create a hotkey: `btcli wallet new_hotkey --wallet.name miner1 --wallet.hotkey default`
3. Fund the coldkey (transfer TAO from exchange)
4. Register on a subnet: `btcli subnet register --netuid <N> --wallet.name miner1`
5. Start your miner process (subnet-specific)
6. Monitor: `btcli subnet metagraph --netuid <N>`

### Checking Profitability

```bash
# Get your UID's emission in a subnet
btcli subnet metagraph --netuid <N>
# Look for your hotkey's row → Emission column (ρ/block)

# Estimate daily TAO:
# daily_TAO = emission_per_block × 7200
```

### Recovering a Wallet

```bash
# Restore coldkey from mnemonic
btcli wallet regen_coldkey --wallet.name <name> --mnemonic "word1 word2 ..."

# Restore hotkey from mnemonic
btcli wallet regen_hotkey --wallet.name <name> --wallet.hotkey <hk_name> --mnemonic "..."
```

---

## Useful Resources

- **Bittensor Docs:** https://docs.bittensor.com
- **TAO Stats:** https://taostats.io
- **Subnet Explorer:** https://x.taostats.io
- **GitHub:** https://github.com/opentensor/bittensor
- **btcli Reference:** `btcli --help`
