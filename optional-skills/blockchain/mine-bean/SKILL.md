---
name: mine-bean
version: 0.2.0
description: Mine $BEAN on Base from inside Hermes Agent. Round-based on-chain deployment, five strategy presets, autonomous cron mode, signed Gitlawb audit log.
author: MineBean
license: MIT
homepage: https://minebean.com
repository: https://github.com/damo-nu11/hermes-mine-bean
network: base-mainnet
audit_log: gitlawb://did:key:z6MkwVfgaAnuypajisEkJLkVbWPiPEBwceMkGutfXpEEYHKi/minebean-rounds
---

# mine-bean

MineBean ($BEAN) is a round-based mining game on Base. New round every 60 seconds. Pick a strategy, deploy ETH into blocks on a 5x5 grid, earn $BEAN rewards on each round close. Occasional beanpot jackpots hit ~1-in-777 rounds.

This skill gives any Hermes agent access to the live game through seven tools and a `/minebean` slash command. Run interactively from a chat, or schedule headless mining via the bundled `hermes-minebean-deploy` console script.

## Install

```bash
pip install hermes-mine-bean
hermes plugins enable minebean
```

Or with MCP support for Claude Desktop / Cursor:

```bash
pip install "hermes-mine-bean[mcp]"
```

## Configure

Add to `~/.hermes/.env`:

```
# Required for autonomous mining (broadcasting)
MINEBEAN_DEPLOYER_KEY=0x...your_dedicated_test_wallet_private_key...

# Or for Bankr-managed signing
# BANKR_API_KEY=bk_ptr_...
# MINEBEAN_MINER_ADDRESS=0x...

# Optional safety guards
MINEBEAN_MAX_DEPLOYS_PER_DAY=100
MINEBEAN_PER_BLOCK_WEI=2500000000000

# Optional RPC override
# BASE_RPC_URL=https://mainnet.base.org
```

For read-only inspection (no broadcasting):

```
MINEBEAN_MINER_ADDRESS=0x...the_address_you_want_to_observe...
```

## Tools

| Tool | What it does |
|---|---|
| `minebean_status` | Live round state, beanpot pool, caller's pending balances |
| `minebean_pending` | Pending ETH + BEAN for any address |
| `minebean_set_profile` | Save default strategy (sniper, anti-winner, beanpot-hunter, anti-loser, nostradamus) |
| `minebean_deploy` | Build deploy plan, optionally broadcast (dry-run default) |
| `minebean_claim` | Claim pending winnings (dry-run default) |
| `minebean_autostart` | Install autonomous mining cron job |
| `minebean_autostop` | Remove the cron job |

Slash command: `/minebean <subcommand>`. Try `/minebean status` first.

## Strategy presets

All five use the closed-form EV optimum `X* = sqrt(K × P × T) − T` where P is BEAN price in ETH, T is the deployment total (strategy-specific), and K = B / 0.105.

| Preset | Blocks | T source | K | At T ≥ threshold |
|---|---|---|---|---|
| anti-winner | 24 (excl. prev winner) | current grid | 10.476 (B=1.1) | Deploy minimum (beanpot eligibility) |
| nostradamus | All 25 | avg of last 3 rounds | 9.524 (B=1.0) | Skip |
| anti-loser | 24 (excl. coldest 100-rd) | max(grid, avg last 3) | 9.524 | Skip |
| sniper | All 25 | live grid at deploy | 9.524 | Skip |
| beanpot-hunter | All 25 | current grid | dynamic (B=1+pot/777) | Skip if pot < 62.16 BEAN |

Beanpot-hunter additionally scales size linearly from 0.5% of wallet at 62.16 BEAN to max at 700 BEAN. Sniper uses adaptive 2-10s timing offsets at broadcast.

## Autonomous mode

```bash
hermes cron add --name "MineBean deploy" --no-agent --script /path/to/wrapper.sh "every 60s"
```

Or use the built-in autostart: `/minebean autostart every 60s`.

The cron entry enforces `MINEBEAN_MAX_DEPLOYS_PER_DAY` and exits cleanly on ceiling hit (cron doesn't error-spam).

## Privacy mode

Configure your Hermes runtime to route LLM inference through Venice for end-to-end no-log privacy. The skill itself runs entirely on-chain reads + writes, so privacy mode is about your conversational layer with the agent, not the mining loop.

## Safety

- `dry_run=true` is the hard default on every deploy and claim call. Live broadcast requires `MINEBEAN_LIVE_BROADCAST_UNLOCKED=1`.
- Daily deploy ceiling enforced via env var
- Already-deployed-this-round guard prevents accidental double-deploys
- Readonly mode (just `MINEBEAN_MINER_ADDRESS`, no key) lets users inspect any address without exposing keys

## Verifiability

Every round is mirrored to `gitlawb://did:key:.../minebean-rounds` every 5 minutes. Signed, append-only, replicated across the Gitlawb network. Pull the latest window file to audit any round independently of minebean.com.

## Repo

Source, issues, releases: https://github.com/damo-nu11/hermes-mine-bean
