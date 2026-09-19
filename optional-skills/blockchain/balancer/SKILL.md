---
name: balancer
description: "Compare Balancer vs Aave V3 flash loan costs by amount."
version: 0.1.0
author: Ghost (ghost), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [flash-loan, balancer, aave, comparator, web3]
    related_skills: [web3-deployment-workflow, institutional-mev-rust-bot]
---

# Balancer Flash Loan Comparator

Compares Balancer V2/V3 flash loans (0% fee) against Aave V3 (0.05% fee) for any loan size. Recommends cheapest source of liquidity, factoring swap fees, slippage, gas, and optional bridge costs.

## When to Use

- Evaluating which protocol to borrow from for an arbitrage or liquidation flash loan
- Deciding Balancer vs Aave V3 for a specific loan amount and network
- Checking cross-chain arb profitability including bridge fees
- AETHER pipeline needs real-time flash loan cost lookup

Don't use for: mainnet execution without testnet validation, or loans under $1K (gas dominates).

## Prerequisites

- Python 3.11+ with `requests` (stdlib otherwise)
- Alchemy/Infura RPC URL for live gas price (optional — defaults embedded)
- Existing `venom_arb/balancer_integration.py` for on-chain pool queries (not required for comparator)

## How to Run

```bash
# Basic: compare for $100K on Ethereum mainnet
python flash_loan_comparator.py 100000

# Sepolia testnet
python flash_loan_comparator.py 10000 --network sepolia

# JSON output (for AETHER controller integration)
python flash_loan_comparator.py 100000 --json

# Multi-chain with bridge fees
python flash_loan_comparator.py 100000 --multi-chain --network arb

# Custom token pair
python flash_loan_comparator.py 500000 --pair WBTC/WETH
```

## Quick Reference

| Flag | Default | Description |
|------|---------|-------------|
| `amount` | — | Loan amount in USD |
| `--pair` | `USDC/WETH` | Token pair |
| `--network` | `eth` | `eth` / `arb` / `base` / `sepolia` |
| `--multi-chain` | off | Add bridge fees |
| `--json` | off | Machine-readable output |

Exit codes: 0=Balancer cheaper, 1=Aave cheaper, 2=Tie.

## Procedure

1. **Pass loan amount** — `python flash_loan_comparator.py <USD>`
2. **Fetch gas price** — tries public RPC `eth_gasPrice`, falls back to default
3. **Compute Balancer cost** — flash fee $0 + swap fee (0.05%) + slippage (loan/2TVL) + gas
4. **Compute Aave V3 cost** — flash fee 0.05% + swap fee + slippage (loan/2×$10B) + gas
5. **Compare totals** — prints table, returns exit code
6. **Integrate** — pipe `--json` into AETHER controller or cron job

## Pitfalls

- **TVL hardcoded** — Balancer $30M / Aave $10B are Sept 2026 snapshots; live TVL drifts. Rerun with `--json` and feed actual values from Dune/Alchemy for production.
- **Slippage approximation** — `loan / (2 × liquidity)` is rough; real Balancer weighted pools vary by invariant. Cap at 10%.
- **Gas is static** — mainnet default $12 assumes ~200k gas @ 20 gwei. Actual varies with base fee.
- **Swap fee assumption** — uses 0.05% mid-range; stable pools can be 0.01%, volatile 1%.
- **No mainnet key** — comparator is analysis only; does not execute transactions.
- **Windows MSYS paths** — script uses forward-slash paths; PowerShell `cmd //c` for symlinks.

## Verification

Run the four test amounts and check crossover logic:

```bash
python flash_loan_comparator.py 10000    # → Balancer cheaper (exit 0)
python flash_loan_comparator.py 100000   # → Aave cheaper (exit 1)
python flash_loan_comparator.py 1000000  # → Aave cheaper (exit 1)
python flash_loan_comparator.py 10000000 # → Aave cheaper (exit 1)
```

Expected crossover: ~$25K–$50K where Aave's 0.05% fee beats Balancer's slippage on $30M TVL.
