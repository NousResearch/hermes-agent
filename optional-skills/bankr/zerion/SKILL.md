---
name: zerion
description: Interpreted crypto wallet data for AI agents. Use when an agent needs portfolio values, token positions, DeFi positions, NFT holdings, transaction history, PnL data, token prices, charts, gas prices, swap quotes, or DApp information across 41+ chains. Zerion transforms raw blockchain data into agent-ready JSON with USD values, protocol labels, and enriched metadata. Supports x402 pay-per-request ($0.01 USDC on Base) and API key access. Triggers on mentions of portfolio, wallet analysis, positions, transactions, PnL, profit/loss, DeFi, token balances, NFTs, swap quotes, gas prices, or Zerion.
---

# Zerion: Wallet Intelligence for AI Agents

Zerion provides interpreted, enriched crypto wallet data across 41+ chains including Ethereum, Base, Arbitrum, Optimism, Polygon, Solana, and more.

Unlike raw RPC data, Zerion returns:
- **USD values** for all positions
- **Protocol labels** (Uniswap, Aave, Lido, etc.)
- **Human-readable transaction types** (swap, stake, bridge, mint, burn)
- **PnL calculations** (realized, unrealized, per-asset, FIFO method)
- **DeFi position breakdowns** (deposits, borrows, LP positions with `group_id`)
- **NFT portfolios** with floor prices and collection metadata
- **Spam filtering** built-in

Two ways to access:

- **x402 (no account needed)**: Pay $0.01 USDC per request on Base. No API key, no signup.
- **API key**: Get a free key instantly at [dashboard.zerion.io](https://dashboard.zerion.io) for higher rate limits.

## Research → Execute Pattern

Zerion is the **research layer**. Use it to analyze wallets, find opportunities, track PnL. Then hand off to Bankr for **execution** (swaps, stop-losses, DCA).

```
Zerion (Research)          Bankr (Execute)
─────────────────         ────────────────
Portfolio analysis   →    Rebalance trades
PnL tracking         →    Stop-loss orders
Position monitoring  →    Take-profit orders
Whale watching       →    Copy trades
Swap quotes          →    Execute best route
NFT floor tracking   →    Buy/sell NFTs
```

## CLI Quick Start

```bash
npm install -g zerion-cli

# Set API key
export ZERION_API_KEY="zk_..."

# Or use x402 (no key needed)
zerion-cli wallet portfolio 0x... --x402

# Commands
zerion-cli wallet portfolio <address>      # Total USD value
zerion-cli wallet positions <address>      # All token positions
zerion-cli wallet transactions <address>   # Transaction history
zerion-cli wallet pnl <address>            # Profit & loss
zerion-cli wallet analyze <address>        # Full analysis
zerion-cli chains list                     # Supported chains
```

---

## Wallet Endpoints

### GET /v1/wallets/{address}/portfolio

Returns aggregated portfolio value across all chains.

```bash
curl "https://api.zerion.io/v1/wallets/0x.../portfolio?currency=usd" \
  -H "Authorization: Basic $(echo -n $ZERION_API_KEY: | base64)"
```

Response:
```json
{
  "data": {
    "attributes": {
      "total": { "positions": 44469.60 },
      "positions_distribution_by_type": {
        "wallet": 40000,
        "deposited": 3000,
        "staked": 1469.60
      },
      "positions_distribution_by_chain": {
        "base": 27495.06,
        "ethereum": 6216.25,
        "arbitrum": 1234.56
      },
      "changes": {
        "absolute_1d": 305.86,
        "percent_1d": 0.69
      }
    }
  }
}
```

### GET /v1/wallets/{address}/positions

Returns all fungible token and DeFi positions.

Query params:
- `filter[positions]`: `only_simple` (tokens only), `only_defi` (protocol positions), `no_filter` (all)
- `filter[chain_ids]`: Comma-separated chain IDs (e.g., `base,ethereum,arbitrum`)
- `filter[trash]`: `only_non_trash` (exclude spam), `only_trash`, `no_filter`
- `sort`: `value` or `-value`

**Understanding LP Positions**: Liquidity pools return multiple positions (one per token) with shared `group_id`. Group by `group_id` to display LP holdings together.

Response includes:
- Token symbol, name, icon URL
- Quantity (int, float, decimals, numeric)
- USD value and price
- Position type: `wallet`, `deposited`, `borrowed`, `staked`, `locked`
- Protocol name and DApp relationship
- `group_id` for LP position grouping

### GET /v1/wallets/{address}/transactions

Returns interpreted transaction history.

Query params:
- `filter[chain_ids]`: Filter by chains
- `filter[asset_types]`: `fungible`, `nft`
- `filter[trash]`: `only_non_trash`, `no_filter`
- `page[size]`: Results per page (default 20)
- `page[after]`: Cursor for pagination

Each transaction includes:
- `operation_type`: `trade`, `send`, `receive`, `approve`, `stake`, `unstake`, `borrow`, `repay`, `bridge`, `mint`, `burn`, `bid`, `execute`
- `transfers` array with direction, token info, quantities, USD values
- `fee` with gas cost in native token and USD
- `application_metadata` with contract address and method info
- Related `dapp` and `chain` relationships

### GET /v1/wallets/{address}/pnl

Returns Profit and Loss using FIFO method.

Query params:
- `currency`: `usd` (default)
- `filter[chain_ids]`: Comma-separated chain IDs

Response:
```json
{
  "data": {
    "attributes": {
      "total_gain": -15076.15,
      "realized_gain": 45328.28,
      "unrealized_gain": -60404.44,
      "relative_total_gain_percentage": -5.65,
      "relative_realized_gain_percentage": 28.08,
      "relative_unrealized_gain_percentage": -57.36,
      "total_fee": 681.81,
      "total_invested": 266672.34,
      "realized_cost_basis": 161370.01,
      "net_invested": 105302.33,
      "received_external": 128217.01,
      "sent_external": 67415.77,
      "sent_for_nfts": 4333.36,
      "received_for_nfts": 423.01
    }
  }
}
```

### GET /v1/wallets/{address}/chart

Returns portfolio balance chart over time.

Query params:
- `currency`: `usd`
- `filter[chain_ids]`: Filter by chains
- `period`: Time period for chart

### GET /v1/wallets/{address}/nft-portfolio

Returns NFT portfolio overview with total estimated value.

### GET /v1/wallets/{address}/nft-positions

Returns list of NFT positions held by wallet.

Query params:
- `filter[chain_ids]`: Filter by chains
- `sort`: Sort order
- Pagination supported

### GET /v1/wallets/{address}/nft-collections

Returns NFT collections held by wallet with floor prices.

---

## Fungibles (Token) Endpoints

### GET /v1/fungibles

Returns paginated list of fungible assets. Supports search.

Query params:
- `filter[search_query]`: Search by name or symbol
- `filter[implementation_chain_id]`: Filter by chain
- `filter[implementation_address]`: Filter by contract address
- `sort`: Sort order

### GET /v1/fungibles/{fungible_id}

Returns single fungible asset by ID.

### GET /v1/fungibles/implementation/{chain}:{address}

Returns fungible by chain:address pair (e.g., `ethereum:0xa5a4...`).

### GET /v1/fungibles/{fungible_id}/chart

Returns price chart for fungible asset.

Query params:
- `filter[period]`: `hour`, `day`, `week`, `month`, `year`, `max`

---

## NFT Endpoints

### GET /v1/nfts

Returns list of NFTs with metadata.

Query params:
- Filter and pagination supported

### GET /v1/nfts/{nft_id}

Returns single NFT by ID with full metadata, traits, and collection info.

---

## DApp Endpoints

### GET /v1/dapps

Returns list of DApps (protocols) indexed by Zerion.

### GET /v1/dapps/{dapp_id}

Returns single DApp with metadata, supported chains, and categories.

---

## Chain Endpoints

### GET /v1/chains

Returns all 41+ supported chains with metadata.

### GET /v1/chains/{chain_id}

Returns single chain by ID.

---

## Gas Prices

### GET /v1/gas-prices

Returns real-time gas prices across all supported chains.

Useful for:
- Estimating transaction costs
- Choosing optimal chain for execution
- Timing transactions for lower fees

---

## Swap & Bridge Quotes

### GET /v1/swap/offers

Returns swap/bridge quotes from multiple providers (aggregator).

Query params:
- Input/output tokens
- Amount
- Slippage tolerance

Returns quotes from 0x, 1inch, Uniswap, and more. Zerion charges 0.5% on L2/alt-L1 trades (waived with Genesis NFT).

**Note**: Response time is 5-10 seconds due to multi-provider aggregation.

### GET /v1/swap/fungibles

Returns fungibles available for bridge exchange (cross-chain swaps).

---

## Webhooks (Subscriptions)

Real-time notifications for wallet activity.

### POST /v1/subscriptions/wallet-transactions

Create subscription for wallet transactions.

```json
{
  "data": {
    "type": "subscriptions",
    "attributes": {
      "wallet_addresses": ["0x...", "0x..."],
      "chain_ids": ["base", "ethereum"],
      "callback_url": "https://your-server/webhook"
    }
  }
}
```

### GET /v1/subscriptions

List all subscriptions.

### GET /v1/subscriptions/{id}

Get subscription by ID.

### DELETE /v1/subscriptions/{id}

Delete subscription.

### POST /v1/subscriptions/{id}/enable

Enable a disabled subscription.

### POST /v1/subscriptions/{id}/disable

Disable subscription (pause notifications).

### PATCH /v1/subscriptions/{id}/wallets

Add/remove wallets from subscription.

### PUT /v1/subscriptions/{id}/wallets

Replace all wallets in subscription.

### PUT /v1/subscriptions/{id}/callback-url

Update callback URL.

### PUT /v1/subscriptions/{id}/chain-ids

Update monitored chains.

### GET /v1/subscriptions/{id}/wallets

List wallets in subscription.

### GET /v1/subscriptions/{id}/wallets/count

Count wallets in subscription.

**Webhook Payload**:
- Signed with X-Signature header (RSA, verify with certificate)
- Includes X-Timestamp and X-Certificate-URL headers
- Transaction data with full interpretation
- Prices are `null` in webhooks (query transactions endpoint for prices)

**Limits**:
- Dev key: 1 subscription, 5 wallets max
- Production: Contact api@zerion.io for whitelist

---

## API Key Access

Get a free API key instantly — no credit card required:

1. Go to [dashboard.zerion.io](https://dashboard.zerion.io)
2. Sign up with email or connect wallet
3. Click "Create API Key" — key starts with `zk_...`
4. Copy and use immediately

```bash
export ZERION_API_KEY="zk_your_api_key"

curl "https://api.zerion.io/v1/wallets/0x.../portfolio" \
  -H "Authorization: Basic $(echo -n $ZERION_API_KEY: | base64)"
```

### Rate Limits

| Plan | Requests/Second | Requests/Day | Price |
|------|-----------------|--------------|-------|
| Free | 10 | 10,000 | $0 |
| Growth | 50 | 100,000 | $99/mo |
| Scale | 200 | 1,000,000 | $499/mo |
| Enterprise | Custom | Custom | Contact |

x402 has no rate limits — pay per request ($0.01 USDC each).

---

## x402 Access (Recommended for Agents)

x402 allows agents to pay per request without API keys. Payment is $0.01 USDC on Base.

```typescript
// Using x402 HTTP flow
const response = await fetch('https://api.zerion.io/v1/wallets/0x.../portfolio', {
  headers: {
    'X-402-Payment': signedPaymentHeader // ERC-3009 signature
  }
});
```

With zerion-cli:
```bash
zerion-cli wallet portfolio 0x... --x402
```

---

## Testnet Support

Add `X-Env: testnet` header to get testnet data:

```bash
curl "https://api.zerion.io/v1/wallets/0x.../portfolio" \
  -H "Authorization: Basic ..." \
  -H "X-Env: testnet"
```

---

## Integration with Bankr

### Example: PnL Guardian

Monitor positions and auto-set stop-losses:

```bash
#!/bin/bash
# Research with Zerion
positions=$(zerion-cli wallet positions $WALLET --json)

# Find volatile tokens on Base
risky=$(echo $positions | jq '[.data[] | select(.relationships.chain.data.id == "base") | select(.attributes.value > 500)]')

# Execute with Bankr
for token in $(echo $risky | jq -r '.[].attributes.fungible_info.symbol'); do
  bankr "set stop loss on $token at -20%"
done
```

### Example: Whale Copy Trading

Watch a whale wallet and mirror trades:

```typescript
// Webhook handler
app.post('/webhook/zerion', async (req, res) => {
  const { type, data } = req.body;

  if (type === 'transaction' && data.operation_type === 'trade') {
    const { transfers } = data;
    const bought = transfers.find(t => t.direction === 'in');

    if (bought && bought.value > 1000) {
      // Mirror the trade via Bankr
      await bankr(`buy $100 of ${bought.fungible_info.symbol}`);
    }
  }
});
```

### Example: Best Swap Route

Get quotes from Zerion, execute via Bankr:

```typescript
// Get swap quote from Zerion
const quote = await zerion.get('/v1/swap/offers', {
  params: { from: 'USDC', to: 'ETH', amount: '1000' }
});

const bestRate = quote.data[0];
console.log(`Best rate: ${bestRate.rate} from ${bestRate.provider}`);

// Execute via Bankr
await bankr(`swap $1000 USDC to ETH on base`);
```

---

## Supported Chains

All 41+ chains including:

| Chain | Chain ID |
|-------|----------|
| Ethereum | `ethereum` |
| Base | `base` |
| Arbitrum | `arbitrum` |
| Optimism | `optimism` |
| Polygon | `polygon` |
| Solana | `solana` |
| zkSync Era | `zksync-era` |
| Linea | `linea` |
| Scroll | `scroll` |
| Blast | `blast` |
| Zora | `zora` |
| Degen | `degen` |
| Berachain | `berachain` |
| Monad | `monad` |
| Abstract | `abstract` |
| ... | +26 more |

Full list: https://developers.zerion.io/reference/supported-blockchains

---

## MCP Server

Connect Claude, Cursor, or any MCP client:

```json
{
  "mcpServers": {
    "zerion": {
      "command": "npx",
      "args": ["zerion-mcp-server"],
      "env": {
        "ZERION_API_KEY": "zk_..."
      }
    }
  }
}
```

---

## Error Handling

| Code | Description |
|------|-------------|
| 200 | Success |
| 202 | Accepted - data being prepared, retry shortly |
| 400 | Bad Request - check query params |
| 401 | Unauthorized - invalid API key |
| 402 | Payment Required - x402 payment needed |
| 404 | Not Found - invalid address or resource |
| 429 | Rate Limited - back off with exponential backoff |
| 500 | Server Error - retry with backoff |

**Note**: 202 responses mean data is being indexed. Retry every few seconds until 200. Stop after 2 minutes if still 202.

---

## Resources

- **Get API Key**: https://dashboard.zerion.io (free, instant, no credit card)
- **API Documentation**: https://developers.zerion.io
- **Supported Chains**: https://developers.zerion.io/reference/supported-blockchains
- **CLI Repository**: https://github.com/zeriontech/zerion-cli
- **MCP Server**: https://github.com/zeriontech/zerion-mcp-server
- **x402 Protocol**: https://developers.zerion.io/reference/x402
- **Zerion for Agents**: https://zerion.io/agents
- **Spam Filtering**: https://developers.zerion.io/reference/token-spam-filtering
- **FAQs**: https://developers.zerion.io/reference/faqs
