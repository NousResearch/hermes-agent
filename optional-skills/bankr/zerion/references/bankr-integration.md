# Zerion + Bankr Integration Guide

## Overview

Zerion provides the **research layer** (portfolio data, positions, PnL), while Bankr provides the **execution layer** (trading, stop-losses, DCA).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Agent                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐              ┌─────────────┐              │
│   │   Zerion    │   Research   │   Bankr     │   Execute   │
│   │   Skill     │ ──────────▶  │   Skill     │ ──────────▶ │
│   └─────────────┘              └─────────────┘              │
│         │                            │                       │
│         ▼                            ▼                       │
│   • Portfolio data            • Swaps                        │
│   • Positions                 • Stop-loss                    │
│   • PnL tracking              • Take-profit                  │
│   • Transaction history       • DCA orders                   │
│   • Webhooks                  • Limit orders                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

### 1. Portfolio Protection (PnL Guardian)

Monitor positions and auto-set protective orders:

```bash
#!/bin/bash
# Fetch positions from Zerion
POSITIONS=$(zerion-cli wallet positions $WALLET --positions simple --json)

# Find Base tokens over $500
RISKY=$(echo $POSITIONS | jq '[.data[]
  | select(.relationships.chain.data.id == "base")
  | select(.attributes.value > 500)
  | select(.attributes.fungible_info.symbol != "ETH")
  | select(.attributes.fungible_info.symbol != "USDC")
]')

# Set stop-losses via Bankr
for row in $(echo $RISKY | jq -c '.[]'); do
  SYMBOL=$(echo $row | jq -r '.attributes.fungible_info.symbol')
  VALUE=$(echo $row | jq -r '.attributes.value | floor')

  echo "Setting stop-loss for $SYMBOL ($VALUE)"
  bankr "set stop loss on $SYMBOL at -20%"
done
```

### 2. PnL-Based Rebalancing

Rebalance based on performance:

```bash
# Check PnL
PNL=$(zerion-cli wallet pnl $WALLET --json)
UNREALIZED=$(echo $PNL | jq '.data.attributes.unrealized_gain')

if (( $(echo "$UNREALIZED > 5000" | bc -l) )); then
  # Take some profit
  bankr "sell 20% of my ETH for USDC"
elif (( $(echo "$UNREALIZED < -5000" | bc -l) )); then
  # DCA more
  bankr "DCA $100 into ETH daily for 7 days"
fi
```

### 3. Whale Watching + Copy Trading

Mirror whale trades in real-time:

```javascript
// Webhook handler for Zerion events
app.post('/webhook/zerion', async (req, res) => {
  const { type, data } = req.body;

  if (type !== 'transaction') return res.sendStatus(200);
  if (data.operation_type !== 'trade') return res.sendStatus(200);

  const bought = data.transfers.find(t => t.direction === 'in');
  const sold = data.transfers.find(t => t.direction === 'out');

  // Only copy if whale bought > $10k worth
  if (bought && bought.value > 10000) {
    const token = bought.fungible_info.symbol;
    const myAmount = Math.min(bought.value * 0.01, 100); // 1% or max $100

    console.log(`Whale bought ${token}, mirroring with $${myAmount}`);
    await exec(`bankr "buy $${myAmount} of ${token}"`);
  }

  res.sendStatus(200);
});
```

### 4. Smart DCA Based on Portfolio Allocation

DCA into underweight positions:

```bash
# Get current allocation
PORTFOLIO=$(zerion-cli wallet portfolio $WALLET --json)
TOTAL=$(echo $PORTFOLIO | jq '.data.attributes.total.positions')

POSITIONS=$(zerion-cli wallet positions $WALLET --positions simple --json)

# Check ETH allocation
ETH_VALUE=$(echo $POSITIONS | jq '[.data[] | select(.attributes.fungible_info.symbol == "ETH") | .attributes.value] | add')
ETH_PERCENT=$(echo "scale=2; $ETH_VALUE / $TOTAL * 100" | bc)

# Target: 50% ETH
if (( $(echo "$ETH_PERCENT < 45" | bc -l) )); then
  DIFF=$(echo "scale=0; ($TOTAL * 0.50 - $ETH_VALUE) / 7" | bc)
  bankr "DCA $${DIFF} into ETH daily for 7 days"
fi
```

### 5. Transaction-Triggered Alerts

Set up automated responses to wallet activity:

```javascript
// When receiving tokens, auto-set stop-loss
app.post('/webhook/zerion', async (req, res) => {
  const { type, data } = req.body;

  if (type === 'transaction' && data.operation_type === 'receive') {
    for (const transfer of data.transfers) {
      if (transfer.direction === 'in' && transfer.value > 500) {
        const token = transfer.fungible_info.symbol;

        // Skip stables
        if (['USDC', 'USDT', 'DAI'].includes(token)) continue;

        console.log(`Received ${token} worth $${transfer.value}`);
        await exec(`bankr "set stop loss on ${token} at -15%"`);
      }
    }
  }

  res.sendStatus(200);
});
```

## CLI Cheatsheet

### Zerion CLI
```bash
zerion-cli wallet portfolio <addr>      # Total value
zerion-cli wallet positions <addr>      # All positions
zerion-cli wallet transactions <addr>   # History
zerion-cli wallet pnl <addr>            # Profit/loss
zerion-cli wallet analyze <addr>        # Full analysis
```

### Bankr CLI
```bash
bankr "price of ETH"                    # Get price
bankr "my balances"                     # Check balances
bankr "swap $100 USDC to ETH"           # Execute swap
bankr "stop loss on TOKEN at -20%"      # Set stop-loss
bankr "DCA $50 into ETH daily"          # Set up DCA
bankr "sell half my TOKEN"              # Partial sell
```

## Environment Setup

```bash
# Both API keys
export ZERION_API_KEY="zk_..."
export BANKR_API_KEY="bk_..."

# Or use x402 for Zerion (no key needed)
export ZERION_X402=true
```

## MCP Configuration

Use both skills in Claude/Cursor:

```json
{
  "mcpServers": {
    "zerion": {
      "command": "npx",
      "args": ["zerion-mcp-server"],
      "env": { "ZERION_API_KEY": "zk_..." }
    },
    "bankr": {
      "command": "npx",
      "args": ["bankr-mcp-server"],
      "env": { "BANKR_API_KEY": "bk_..." }
    }
  }
}
```

## Best Practices

1. **Research before execution**: Always fetch current positions/prices before trading
2. **Use webhooks for real-time**: Don't poll - use Zerion webhooks for instant updates
3. **Set reasonable thresholds**: Avoid over-trading on small moves
4. **Log everything**: Keep audit trail of research → decision → execution
5. **Handle errors gracefully**: Both APIs can fail - implement retries
