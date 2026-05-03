# Hyperliquid Reference

Trade perpetual futures, spot tokens, and equities on Hyperliquid's on-chain order book.

## Overview

Hyperliquid is a high-performance L1 DEX with an on-chain order book. It supports perpetual futures (crypto, stocks, commodities), spot trading, and advanced order management.

**Chain**: Hyperliquid L1 (bridged via Arbitrum)
**Collateral**: USDC
**Protocol**: [Hyperliquid](https://hyperliquid.xyz)

### Account Structure

| Account | Purpose |
|---------|---------|
| **Spot** | Receives bridge deposits, holds spot tokens |
| **Perps** | USDC margin for perpetual trading |

Perps trading requires USDC in the perps account. Bankr auto-transfers from spot to perps when needed.

## Supported Assets

| Category | Examples | Max Leverage |
|----------|----------|-------------|
| Crypto | BTC, ETH, SOL, HYPE, 100+ more | Varies per asset (up to 50x) |
| Stocks | TSLA, AAPL, NVDA, GOOGL (via HIP-3) | Varies per asset |
| Spot | HYPE, PURR, and other HL-native tokens | N/A (no leverage) |

## Prompt Examples

**Open perps positions:**
- "Long $100 of BTC on hyperliquid"
- "Short $50 of ETH on hyperliquid with 10x leverage"
- "Long TSLA with 5x leverage"
- "Short SOL with 20x"

**Limit orders:**
- "Long $100 of BTC at $60000 on hyperliquid"
- "Short ETH with a limit price of $4000"

**Spot trading:**
- "Buy $50 of HYPE on hyperliquid"
- "Sell 100 PURR on hyperliquid"
- "Buy $200 of HYPE at $25"

**TP/SL on new positions:**
- "Long BTC with 10x and take profit at $70000 and stop loss at $55000"
- "Long ETH with 200% ROE take profit"
- "Short SOL with stop loss if price increases by $2000"

**TP/SL on existing positions:**
- "Set take profit at $70000 on my BTC position"
- "Set SL at $55000 on my ETH position"
- "Set TP at $4000 and SL at $3000 on my ETH position"

**View positions:**
- "Show my hyperliquid positions"
- "What positions do I have on HL?"

**Close positions:**
- "Close my BTC position on hyperliquid"
- "Close 50% of my ETH position"
- "Close all my hyperliquid positions"

**Manage leverage and margin:**
- "Set my BTC leverage to 20x"
- "Change ETH leverage to 5x isolated"
- "Add $500 margin to my BTC position"
- "Remove $200 margin from my ETH position"

**Order management:**
- "Show my open orders on hyperliquid"
- "Change my BTC limit order price to $62000"
- "Cancel my BTC order on hyperliquid"
- "Cancel all my hyperliquid orders"

**Balances and transfers:**
- "Show my hyperliquid balances"
- "Transfer $500 from spot to perps on hyperliquid"
- "Move $200 from perps to spot"

**Bridge funds:**
- "Deposit $500 USDC to hyperliquid"
- "Bridge $1000 to hyperliquid from arbitrum"
- "Withdraw $500 from hyperliquid"

**Market data:**
- "BTC price on hyperliquid"
- "What's the funding rate for SOL?"
- "What can I trade on hyperliquid?"
- "Search for TSLA on hyperliquid"

## Position Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Direction** | Long (price up) or Short (price down) | "long", "short" |
| **Collateral** | USDC amount for margin | "$100" |
| **Leverage** | 1x to max for asset (default 1x) | "10x leverage" |
| **Margin Mode** | Cross or isolated (default isolated) | "5x isolated" |
| **Order Type** | Market (default) or limit | "at $60000" |
| **Take Profit** | Price, ROE%, or delta (new positions) | "TP at $70000", "200% ROE TP" |
| **Stop Loss** | Price, ROE%, or delta (new positions) | "SL at $55000", "5% ROE SL" |

## TP/SL Formats

### On New Positions (all formats supported)

| Format | Example | Description |
|--------|---------|-------------|
| Absolute Price | "TP at $70000" | Trigger at exact price |
| ROE Percentage | "200% ROE take profit" | Based on return on equity |
| Price Delta | "SL if price drops by $2000" | Relative to entry price |

### On Existing Positions (absolute prices only)

| Format | Example | Description |
|--------|---------|-------------|
| Absolute Price | "Set TP at $70000 on my BTC position" | Trigger at exact price |

## Bridge Operations

| Operation | Min Amount | Fee | Time | Destination |
|-----------|-----------|-----|------|-------------|
| Deposit | 5 USDC | Gas on source chain | ~1 min | HL spot account |
| Withdraw | Any | 1 USDC | ~3-4 min | Arbitrum |

**Deposit source chains**: Arbitrum, Base, Polygon, Ethereum (auto-detected based on balances)

## Margin Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Isolated** (default) | Only specified collateral at risk | Most trades — limits downside |
| **Cross** | Entire perps account as collateral | Advanced — shares margin across positions |

## Funding Rates

- Charged every 8 hours
- Longs pay shorts (or vice versa) depending on rate
- Check with: "What's the funding rate for BTC on hyperliquid?"
- Can erode profits on long-held positions

## Common Issues

| Issue | Resolution |
|-------|------------|
| Insufficient USDC on HL | Bridge USDC from any EVM chain |
| USDC in spot, not perps | Transfer spot to perps (auto-handled for perps trades) |
| Asset not found | Check available assets, use correct symbol |
| Leverage exceeds max | Each asset has its own max leverage |
| Margin update rejected | Only works on isolated positions |
| Bridge deposit too small | Minimum 5 USDC |
| Withdrawal delayed | Normal: ~3-4 minutes to land on Arbitrum |

## HIP-3 Assets (Stocks, RWAs)

Hyperliquid supports equities and real-world assets via HIP-3 builder-deployed dexes:
- Trade stocks like TSLA, AAPL, NVDA as perpetual futures
- Same tools and workflow as crypto perps
- Dex abstraction is enabled automatically on first trade
- Search with: "Search for TSLA on hyperliquid" or "What stocks can I trade on hyperliquid?"

## Trading Flow

1. **Check balances** — "Show my hyperliquid balances"
2. **Bridge if needed** — "Deposit $500 USDC to hyperliquid"
3. **Open position** — "Long $100 of BTC on hyperliquid with 10x"
4. **Set risk management** — "Set TP at $70000 and SL at $55000 on my BTC position"
5. **Monitor** — "Show my hyperliquid positions"
6. **Close** — "Close my BTC position on hyperliquid"
7. **Withdraw** — "Withdraw $500 from hyperliquid"

## Risk Warnings

- Leverage amplifies both gains and losses
- Positions can be liquidated if margin is insufficient
- 50x leverage means 2% price move = 100% gain/loss
- Funding rates can erode profits on long-held positions
- Bridge deposits/withdrawals take a few minutes to process
