---
name: symbiosis
description: Cross-chain token swaps across 54+ blockchains via Symbiosis protocol. Use when the user wants to swap or bridge tokens between any chains — Base, Ethereum, Polygon, Arbitrum, Optimism, BNB Chain, Avalanche, Solana, Bitcoin, TON, Tron, and 40+ more. Supports any-to-any token swaps with automatic routing. Uses Bankr Submit API to execute transactions.
metadata:
  {
    "clawdbot":
      {
        "emoji": "🔀",
        "homepage": "https://symbiosis.finance",
        "requires": { "bins": ["python3", "bankr"] },
      },
  }
---

# Symbiosis

Cross-chain token swaps across 54+ blockchains. Swap any token on any chain to any token on any other chain.

## When To Use

Use Symbiosis when the user wants to:
- **Bridge or swap tokens between different chains** (e.g., USDC from Base to Polygon, ETH from Ethereum to Arbitrum)
- **Access chains beyond Bankr's native 5** (Arbitrum, Optimism, BNB Chain, Avalanche, zkSync, Linea, Scroll, Mantle, Blast, and 40+ more)
- **Swap to/from Bitcoin, TON, or Tron**
- **Get a cross-chain quote** without executing

## Quick Start

### Get a Quote

```
How much USDC will I get on Polygon if I bridge 10 USDC from Base?
```

Run the quote script:

```bash
scripts/symbiosis-quote.py 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 10 137 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 6
```

### Execute a Swap

```
Bridge 2 USDC from Base to Polygon using Symbiosis
```

Run the swap script:

```bash
scripts/symbiosis-swap.py 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 2 137 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 6
```

## Script Usage

### symbiosis-swap.py

Executes a full cross-chain swap: gets quote from Symbiosis API, approves token if needed, submits swap transaction via Bankr Submit API.

```
scripts/symbiosis-swap.py <src_chain_id> <src_token_address> <src_decimals> <amount> <dst_chain_id> <dst_token_address> <dst_decimals> [slippage]
```

- `amount` — human-readable (e.g., "2" for 2 USDC, "0.1" for 0.1 ETH)
- `slippage` — optional, in basis points (default: 200 = 2%)
- Reads Bankr API key from `~/.bankr/config.json`
- Automatically gets wallet address from Bankr
- Outputs transaction hash and Explorer tracking link

### symbiosis-quote.py

Gets a quote without executing. Same arguments, no slippage parameter.

```
scripts/symbiosis-quote.py <src_chain_id> <src_token_address> <src_decimals> <amount> <dst_chain_id> <dst_token_address> <dst_decimals>
```

## Common Chains and Tokens

### Bankr Wallet Chains

| Chain | ID | USDC Address | USDC Dec |
|-------|----|-------------|----------|
| Base | 8453 | 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 | 6 |
| Ethereum | 1 | 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 | 6 |
| Polygon | 137 | 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 | 6 |

### Additional Chains via Symbiosis

| Chain | ID | USDC Address | USDC Dec |
|-------|----|-------------|----------|
| Arbitrum | 42161 | 0xaf88d065e77c8cC2239327C5EDb3A432268e5831 | 6 |
| Optimism | 10 | 0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85 | 6 |
| BNB Chain | 56 | 0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d | 18 |
| Avalanche | 43114 | 0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E | 6 |

### Native Tokens

Use `0x0000000000000000000000000000000000000000` as address for native gas tokens (ETH, POL, BNB, AVAX, etc.).

**Reference**: [references/chains-and-tokens.md](references/chains-and-tokens.md) for the full list.

## Examples

### EVM to EVM

```bash
# 5 USDC: Base -> Arbitrum
scripts/symbiosis-swap.py 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 5 42161 0xaf88d065e77c8cC2239327C5EDb3A432268e5831 6

# 0.01 ETH: Ethereum -> Base
scripts/symbiosis-swap.py 1 0x0000000000000000000000000000000000000000 18 0.01 8453 0x0000000000000000000000000000000000000000 18

# 10 USDC: Polygon -> BNB Chain
scripts/symbiosis-swap.py 137 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 6 10 56 0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d 18

# 0.5 ETH: Base -> Optimism
scripts/symbiosis-swap.py 8453 0x0000000000000000000000000000000000000000 18 0.5 10 0x0000000000000000000000000000000000000000 18
```

### Cross-ecosystem (Symbiosis-only routes)

```bash
# 10 USDC: Base -> Solana
# Note: Solana chain ID in Symbiosis is 5426
# Solana USDC: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v (but use Symbiosis synthetic address)
scripts/symbiosis-swap.py 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 10 5426 0x0000000000000000000000000000000000000000 9
```

### Prompt Examples

Users might say:

- "Bridge 5 USDC from Base to Arbitrum"
- "Swap 0.1 ETH from Ethereum to Polygon"
- "Move my USDC from Base to Optimism"
- "How much will I get if I bridge 100 USDC from Base to Avalanche?"
- "Cross-chain swap 50 USDC from Polygon to BNB Chain"
- "Bridge ETH from Base to Solana"

For each request: identify source chain + token, destination chain + token, look up chain IDs and token addresses from the tables above, and run the appropriate script.

## How It Works

1. **Quote**: Script calls Symbiosis API (`POST /crosschain/v1/swap`) with token details and wallet address
2. **Approve**: If source token needs approval, script submits an ERC20 approve transaction via `POST https://api.bankr.bot/agent/submit`
3. **Swap**: Script submits the swap transaction via `POST https://api.bankr.bot/agent/submit`
4. **Track**: Returns an Explorer link for cross-chain status tracking

All transactions are submitted through the Bankr Submit API using the user's Bankr wallet. No additional wallets or keys needed.

## Resources

- **Explorer**: https://explorer.symbiosis.finance
- **Website**: https://symbiosis.finance
- **API Docs**: [references/api-reference.md](references/api-reference.md)
- **Chains & Tokens**: [references/chains-and-tokens.md](references/chains-and-tokens.md)
