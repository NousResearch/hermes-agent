# Chain Memory - On-Chain AI Memory Storage

Store your AI bot's **complete state** permanently on Monad blockchain.
Never lose your bot's evolution, personality, or memories again.

## Requirements

### 1. FARNS Token Holder (Required)
You must hold **100,000+ FARNS tokens** on Solana to use this feature.

```
FARNS Token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
Buy FARNS: https://pump.fun/coin/9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
```

### 2. Monad Wallet
You need a Monad wallet with MON tokens for gas fees.

### 3. Supported Bots

| Bot | Support Level | Notes |
|-----|--------------|-------|
| **Cosmos** | Full | Recommended - captures everything |
| ClawwBot/OpenClaw | Partial | Memory only, no evolution |
| Claude Code | Partial | Basic memory support |
| Kimi | Partial | Basic memory support |

**For best results, use Cosmos**: https://github.com/cosmos-ai/cosmos

## Quick Setup

```bash
# 1. Install dependencies
pip install memvid web3 eth-account aiohttp

# 2. Run setup wizard
python -m cosmos.integration.chain_memory.setup
```

The setup wizard will:
1. Verify your FARNS token holdings
2. Configure your Monad wallet
3. Set up auto-save options
4. Generate environment variables

## What Gets Saved

Chain Memory captures your **ENTIRE bot state**:

- **Memory Layers**
  - Archival memory (long-term knowledge)
  - Dialogue history (all conversations)
  - Episodic memory (experiences)

- **Personality & Evolution**
  - Current personality traits
  - Evolution history
  - Trait mutations over time

- **Session State**
  - Claude session memory
  - Active context profiles
  - Running jobs and tasks

- **Integrations**
  - X/Twitter automation state
  - Meme scheduler state
  - Trading state

- **User Data**
  - Notes
  - Code snippets
  - Health tracking

## Usage

### Push State to Chain

```python
from cosmos.integration.chain_memory import ChainMemory
import asyncio

async def backup():
    cm = ChainMemory()

    # Verify FARNS (required)
    result = await cm.verify_farns()
    if not result['verified']:
        print(f"Need {result['required']:,} FARNS tokens!")
        return

    # Push complete state
    record = await cm.push_memory(title="Full Backup 2026-02-02")

    print(f"Backed up! Memory ID: {record.memory_id}")
    print(f"TX Count: {len(record.tx_hashes)}")

    # Save TX list for recovery
    export = cm.export_tx_list(record.memory_id)
    with open("backup_restore_info.json", "w") as f:
        f.write(export)

asyncio.run(backup())
```

### Restore State from Chain

```python
from cosmos.integration.chain_memory import ChainMemory
import asyncio

async def restore():
    cm = ChainMemory()

    # Option 1: Restore by TX IDs
    package = await cm.pull_memory(tx_ids=["0x...", "0x..."])

    # Option 2: Find all backups from your wallet
    packages = await cm.pull_all_memories(wallet_address="0x...")

    # Load into Cosmos
    cm.load_into_cosmos(package)
    print("State restored!")

asyncio.run(restore())
```

### CLI Commands

```bash
# Run setup wizard
python -m cosmos.integration.chain_memory.setup

# Push current state
python -m cosmos.integration.chain_memory push --title "My Backup"

# Pull and restore
python -m cosmos.integration.chain_memory pull --wallet 0xYourAddress

# List local backup records
python -m cosmos.integration.chain_memory list

# Export backup info for sharing
python -m cosmos.integration.chain_memory export --id abc123 -o backup.json

# Estimate cost
python -m cosmos.integration.chain_memory estimate
```

### Auto-Save (Recommended)

Enable automatic backups:

```python
from cosmos.integration.chain_memory import enable_auto_save

# Start auto-save (backs up every 60 minutes)
manager = enable_auto_save(interval_minutes=60, crash_save=True)

# Your bot runs normally...
# Auto-save happens in background
# Emergency save on crash/exit
```

Or via environment:
```bash
export CHAIN_MEMORY_ENABLED="true"
export AUTO_SAVE_INTERVAL="60"
```

## Startup Memory Loading

When starting your bot, prompt to load chain memories:

```python
from cosmos.integration.chain_memory import prompt_memory_load, auto_load_memories
import asyncio

def main():
    # Prompt user to load chain memories
    memories_to_load = prompt_memory_load(bot_type="cosmos")

    if memories_to_load:
        asyncio.run(auto_load_memories(memories_to_load))

    # Continue with normal bot startup...
```

## Environment Variables

```bash
# Required
export MONAD_PRIVATE_KEY="0x..."           # Your Monad wallet

# Recommended
export MONAD_RPC="https://rpc.monad.xyz"   # Monad RPC endpoint
export CHUNK_SIZE_KB="80"                   # Chunk size (default 80KB)

# Optional - Auto-save
export CHAIN_MEMORY_ENABLED="true"
export AUTO_SAVE_INTERVAL="60"              # Minutes between backups

# Optional - Solana (for FARNS verification)
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export HELIUS_API_KEY="..."                 # For faster FARNS checks
```

## Cost Breakdown (80KB Chunks)

| State Size | Chunks | Est. Cost (MON) | Est. Cost (USD) |
|------------|--------|-----------------|-----------------|
| 1 MB       | 13     | 0.13            | $0.07           |
| 5 MB       | 64     | 0.64            | $0.32           |
| 10 MB      | 128    | 1.28            | $0.64           |
| 50 MB      | 640    | 6.40            | $3.20           |

*Typical Cosmos state: 5-20 MB*

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      CHAIN MEMORY FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. VERIFY FARNS                                                │
│     └── Check Solana wallet holds 100k+ FARNS                   │
│                                                                 │
│  2. CAPTURE STATE                                               │
│     └── Memory + Evolution + Jobs + Everything                  │
│                                                                 │
│  3. ENCODE TO MP4                                               │
│     └── Memvid compresses state into video format               │
│                                                                 │
│  4. CHUNK & UPLOAD                                              │
│     └── Split into 80KB chunks                                  │
│     └── Each chunk = 1 Monad transaction                        │
│     └── Data stored in tx calldata (permanent!)                 │
│                                                                 │
│  5. SAVE RECOVERY INFO                                          │
│     └── TX hashes stored locally                                │
│     └── Export for backup                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Recovery Info Format

```json
{
  "format": "cosmos_chain_memory_v1",
  "memory_id": "abc123def456",
  "title": "Full Backup 2026-02-02",
  "bot_type": "cosmos",
  "chain": "monad",
  "tx_hashes": [
    "0x1234...",
    "0x5678...",
    "..."
  ],
  "total_chunks": 64,
  "total_size": 5242880,
  "uploaded_at": "2026-02-02T12:00:00Z"
}
```

**Save this file!** It contains everything needed to restore your bot's state.

## Security Notes

1. **Private Key**: Your Monad key is stored in environment variables only
2. **FARNS Verification**: Checked every time before upload
3. **Data is Public**: On-chain data is visible (but encoded as video)
4. **Immutable**: Once uploaded, cannot be deleted or modified
5. **User Wallet**: YOUR wallet pays for gas, YOUR wallet owns the data

## Troubleshooting

### "FARNS verification failed"
- Ensure your Solana wallet holds 100,000+ FARNS
- Buy FARNS: https://pump.fun/coin/9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

### "Insufficient MON balance"
- Your Monad wallet needs MON for gas fees
- Typical backup costs 0.1-1.0 MON

### "Memvid not installed"
```bash
pip install memvid
```

### "web3 not installed"
```bash
pip install web3 eth-account
```

## Technical Details

- **Chain**: Monad (Chain ID 143)
- **Storage**: Raw transaction calldata (BetterClips-style)
- **Chunk Size**: 80KB default (proven by BetterClips)
- **Encoding**: Memvid video compression
- **Signature**: `FARNSMEM` (8 bytes) header
- **FARNS Required**: 100,000 tokens minimum

## License

Copyright (c) 2026 Cosmos AI. All rights reserved.

**FARNS token holders only.** The protected core module is compiled for distribution.


> *Powered by The Cosmic Davis 12D Hebbian Transformer by Cory Shane Davis - [Zenodo Archive](https://zenodo.org/records/17574447)*
