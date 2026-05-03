# Zerion API Reference

Complete endpoint reference for the Zerion API.

## Authentication

### API Key (Basic Auth)
```
Authorization: Basic {base64(API_KEY:)}
```

Example:
```bash
curl "https://api.zerion.io/v1/wallets/0x.../portfolio" \
  -H "Authorization: Basic $(echo -n $ZERION_API_KEY: | base64)"
```

### x402 Payment
```
X-402-Payment: {signed_erc3009_authorization}
```

### Testnet Access
```
X-Env: testnet
```

---

## Wallet Endpoints

### GET /v1/wallets/{address}/portfolio

Returns aggregated portfolio value.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `currency` | string | `usd` (default), `eur`, `btc`, `eth` |
| `filter[positions]` | string | `only_simple`, `only_defi`, `no_filter` |

**Response:**
```json
{
  "data": {
    "type": "portfolio",
    "id": "0x...",
    "attributes": {
      "positions_distribution_by_type": {
        "wallet": 44469.60,
        "deposited": 1234.56,
        "borrowed": 0,
        "locked": 0,
        "staked": 5678.90
      },
      "positions_distribution_by_chain": {
        "base": 27495.06,
        "ethereum": 6216.25
      },
      "total": {
        "positions": 51383.06
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

Returns all fungible positions.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `filter[positions]` | string | `only_simple`, `only_defi`, `no_filter` |
| `filter[chain_ids]` | string | Comma-separated chain IDs |
| `filter[trash]` | string | `only_non_trash`, `only_trash`, `no_filter` |
| `sort` | string | `value`, `-value` |
| `page[size]` | integer | Results per page (max 100) |

**Response (single position):**
```json
{
  "type": "positions",
  "id": "...",
  "attributes": {
    "position_type": "wallet",
    "quantity": {
      "int": "6485257514999279000",
      "decimals": 18,
      "float": 6.485257514999279,
      "numeric": "6.485257514999279"
    },
    "value": 13968.45,
    "price": 2153.67,
    "group_id": "abc123...",
    "fungible_info": {
      "name": "Ethereum",
      "symbol": "ETH",
      "icon": { "url": "https://cdn.zerion.io/eth.png" },
      "flags": { "verified": true },
      "implementations": [
        { "chain_id": "ethereum", "address": "", "decimals": 18 },
        { "chain_id": "base", "address": "", "decimals": 18 }
      ]
    }
  },
  "relationships": {
    "chain": { "data": { "type": "chains", "id": "base" } },
    "dapp": { "data": { "type": "dapps", "id": "uniswap-v3" } }
  }
}
```

**Note on LP Positions:**
Liquidity pool positions return multiple entries with the same `group_id`. Group by this field to display all tokens in a pool together.

### GET /v1/wallets/{address}/transactions

Returns interpreted transaction history.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `filter[chain_ids]` | string | Comma-separated chain IDs |
| `filter[asset_types]` | string | `fungible`, `nft` |
| `filter[trash]` | string | `only_non_trash`, `no_filter` |
| `page[size]` | integer | Results per page |
| `page[after]` | string | Cursor for pagination |

**Response (single transaction):**
```json
{
  "type": "transactions",
  "id": "...",
  "attributes": {
    "operation_type": "trade",
    "hash": "0x...",
    "mined_at": "2024-03-21T15:22:35Z",
    "mined_at_block": 12345678,
    "status": "confirmed",
    "nonce": 42,
    "sent_from": "0x...",
    "sent_to": "0x...",
    "fee": {
      "fungible_info": { "symbol": "ETH" },
      "quantity": { "float": 0.001234 },
      "value": 2.65
    },
    "transfers": [
      {
        "direction": "out",
        "fungible_info": { "symbol": "USDC" },
        "quantity": { "float": 1000 },
        "value": 1000,
        "sender": "0x...",
        "recipient": "0x..."
      },
      {
        "direction": "in",
        "fungible_info": { "symbol": "ETH" },
        "quantity": { "float": 0.45 },
        "value": 970
      }
    ],
    "approvals": [],
    "flags": { "is_trash": false },
    "application_metadata": {
      "contract_address": "0x...",
      "method": { "id": "0x...", "name": "swap" }
    }
  },
  "relationships": {
    "chain": { "data": { "type": "chains", "id": "base" } },
    "dapp": { "data": { "type": "dapps", "id": "uniswap-v3" } }
  }
}
```

**Operation Types:**
- `trade` - Token swap
- `send` - Outgoing transfer
- `receive` - Incoming transfer
- `approve` - Token approval
- `stake` - Staking deposit
- `unstake` - Staking withdrawal
- `borrow` - Lending borrow
- `repay` - Lending repayment
- `bridge` - Cross-chain transfer
- `mint` - Token/NFT mint
- `burn` - Token burn
- `bid` - Auction bid
- `execute` - Contract execution

### GET /v1/wallets/{address}/pnl

Returns Profit & Loss (FIFO method).

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `currency` | string | `usd` (default) |
| `filter[chain_ids]` | string | Comma-separated chain IDs |

**Response:**
```json
{
  "data": {
    "type": "wallet_pnl",
    "id": "0x...",
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

Returns portfolio balance chart.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `currency` | string | `usd` |
| `filter[chain_ids]` | string | Filter by chains |
| `period` | string | Chart time period |

### GET /v1/wallets/{address}/nft-portfolio

Returns NFT portfolio overview.

### GET /v1/wallets/{address}/nft-positions

Returns NFT positions held.

### GET /v1/wallets/{address}/nft-collections

Returns NFT collections held.

---

## Fungibles Endpoints

### GET /v1/fungibles

Returns list of fungible assets.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `filter[search_query]` | string | Search by name/symbol |
| `filter[implementation_chain_id]` | string | Filter by chain |
| `filter[implementation_address]` | string | Filter by contract |
| `sort` | string | Sort order |
| `page[size]` | integer | Results per page |

### GET /v1/fungibles/{fungible_id}

Returns single fungible by ID.

### GET /v1/fungibles/implementation/{chain}:{address}

Returns fungible by implementation (e.g., `ethereum:0xa5a4...`).

### GET /v1/fungibles/{fungible_id}/chart

Returns price chart.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `filter[period]` | string | `hour`, `day`, `week`, `month`, `year`, `max` |

---

## NFT Endpoints

### GET /v1/nfts

Returns list of NFTs.

### GET /v1/nfts/{nft_id}

Returns single NFT with metadata.

---

## DApp Endpoints

### GET /v1/dapps

Returns list of DApps/protocols.

### GET /v1/dapps/{dapp_id}

Returns single DApp.

---

## Chain Endpoints

### GET /v1/chains

Returns all supported chains.

### GET /v1/chains/{chain_id}

Returns single chain.

---

## Gas Prices

### GET /v1/gas-prices

Returns real-time gas prices for all chains.

---

## Swap Endpoints

### GET /v1/swap/offers

Returns swap/bridge quotes from multiple providers.

**Note:** Response time 5-10 seconds due to multi-provider aggregation.

### GET /v1/swap/fungibles

Returns fungibles available for bridge.

---

## Subscription (Webhook) Endpoints

### POST /v1/subscriptions/wallet-transactions

Create transaction subscription.

**Request:**
```json
{
  "data": {
    "type": "subscriptions",
    "attributes": {
      "wallet_addresses": ["0x..."],
      "chain_ids": ["base", "ethereum"],
      "callback_url": "https://your-server/webhook"
    }
  }
}
```

### GET /v1/subscriptions

List subscriptions. Limited to 1000.

### GET /v1/subscriptions/{id}

Get subscription by ID.

### DELETE /v1/subscriptions/{id}

Delete subscription.

### POST /v1/subscriptions/{id}/enable

Enable subscription.

### POST /v1/subscriptions/{id}/disable

Disable subscription.

### PATCH /v1/subscriptions/{id}/wallets

Add/remove wallets.

### PUT /v1/subscriptions/{id}/wallets

Replace all wallets.

### PUT /v1/subscriptions/{id}/callback-url

Update callback URL.

### PUT /v1/subscriptions/{id}/chain-ids

Update chain filters.

### GET /v1/subscriptions/{id}/wallets

List wallets in subscription.

### GET /v1/subscriptions/{id}/wallets/count

Count wallets.

---

## Rate Limits

| Plan | Requests/Second | Requests/Day |
|------|-----------------|--------------|
| Free | 10 | 10,000 |
| Growth | 50 | 100,000 |
| Scale | 200 | 1,000,000 |
| Enterprise | Custom | Unlimited |
| x402 | Unlimited | Pay per request |

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 202 | Accepted - data being prepared, retry |
| 400 | Bad Request - invalid parameters |
| 401 | Unauthorized - invalid API key |
| 402 | Payment Required - x402 payment needed |
| 404 | Not Found - invalid address/resource |
| 429 | Rate Limited - back off |
| 500 | Server Error - retry |

**202 Handling:** Retry every 2-5 seconds until 200. Stop after 2 minutes.
