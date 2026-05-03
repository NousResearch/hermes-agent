# Portfolio Reference

Query token balances and portfolio across all supported chains.

## CLI Commands

```bash
bankr wallet portfolio                    # Full portfolio (hides tokens under $1)
bankr wallet portfolio --pnl              # Include profit/loss data
bankr wallet portfolio --nfts             # Include NFT holdings
bankr wallet portfolio --all              # PnL + NFTs
bankr wallet portfolio --chain base       # Filter by chain
bankr wallet portfolio --chain base,solana  # Multiple chains
bankr wallet portfolio --json             # Raw JSON output
```

## REST API

```bash
# Basic portfolio
curl -s "https://api.bankr.bot/wallet/portfolio" \
  -H "X-API-Key: $API_KEY"

# With PnL and NFTs (progressive loading)
curl -s "https://api.bankr.bot/wallet/portfolio?include=pnl,nfts" \
  -H "X-API-Key: $API_KEY"

# Filter by chain
curl -s "https://api.bankr.bot/wallet/portfolio?chains=base,solana" \
  -H "X-API-Key: $API_KEY"
```

> **Deprecation notice**: `GET /agent/balances` still works but is deprecated. Use `GET /wallet/portfolio` instead.

The `/wallet/portfolio` endpoint is a read endpoint — any valid API key with a wallet can access it (no feature flags required).

## Supported Chains

All chains: Base, Polygon, Ethereum, Unichain, Solana, World Chain, Arbitrum, BNB Chain

## Prompt Examples

**Full portfolio:**
- "Show my portfolio"
- "What's my total balance?"
- "How much crypto do I have?"
- "Portfolio value"
- "What's my net worth?"

**Chain-specific:**
- "Show my Base balance"
- "What tokens do I have on Polygon?"
- "Ethereum portfolio"
- "Solana holdings"

**Token-specific:**
- "How much ETH do I have?"
- "What's my USDC balance?"
- "Show my ETH across all chains"
- "BNKR balance"

## Features

- **USD Valuation**: All balances include current USD value
- **PnL Tracking**: Profit/loss data via `--pnl` or `?include=pnl`
- **NFT Holdings**: View NFTs via `--nfts` or `?include=nfts`
- **Progressive Loading**: Request only the data you need with `?include=` parameters
- **Multi-Chain Aggregation**: See the same token across all chains
- **Real-Time Prices**: Values reflect current market prices
- **Comprehensive View**: Shows all tokens with meaningful balances

## Common Tokens Tracked

- **Stablecoins**: USDC, USDT, DAI
- **Blue Chips**: ETH, WETH, WBTC
- **DeFi**: UNI, AAVE, LINK, COMP, CRV
- **Memecoins**: DOGE, SHIB, PEPE, BONK
- **Project tokens**: BNKR, ARB, OP, MATIC

## Use Cases

**Before trading:**
- "Do I have enough ETH to swap for 100 USDC?"
- "Check if I have MATIC for gas on Polygon"

**Portfolio review:**
- "What's my largest holding?"
- "Show portfolio breakdown by chain"
- "What percentage of my portfolio is stablecoins?"

**After transactions:**
- "Did my ETH arrive?"
- "Show my new BNKR balance"
- "Verify the swap completed"

## Output Format

Portfolio responses typically include:
- Token name and symbol
- Amount held
- Current USD value
- Chain location
- Price per token
- 24h price change

## Notes

- Portfolio queries are read-only (no transactions) — any valid API key works
- Shows balance of connected wallet address
- Tokens valued under $1 are hidden by default in CLI output
- Includes native tokens (ETH, MATIC, SOL) and ERC20/SPL tokens
- PnL and NFT data use progressive loading — only fetched when requested, keeping base queries fast
