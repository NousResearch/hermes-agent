# Webhooks — Event-Driven Solana Notifications

## What Webhooks Are

Webhooks deliver real-time Solana on-chain events to your server via HTTP POST. Instead of polling for changes, Helius pushes parsed transaction data to your endpoint as events happen.

- Available on ALL plans, including free tier
- Up to 100,000 addresses per webhook
- 150+ supported transaction types with filtering
- 1 credit per event delivered
- 100 credits per management operation (create, update, delete, list, get)
- Three webhook types: Enhanced (parsed), Raw (unfiltered), Discord (channel notifications)

## MCP Tools

All webhook operations have direct MCP tools. Use these for managing webhooks:

| MCP Tool | What It Does |
|---|---|
| `createWebhook` | Create a new webhook to monitor addresses for specific transaction types |
| `getAllWebhooks` | List all active webhooks on your account |
| `getWebhookByID` | Get details for a specific webhook |
| `updateWebhook` | Modify webhook URL, addresses, or transaction type filters |
| `deleteWebhook` | Permanently remove a webhook |
| `getWebhookGuide` | Fetch live official webhook documentation |

When the user asks to set up monitoring, alerts, or event-driven processing — use `createWebhook`. For troubleshooting existing webhooks, start with `getAllWebhooks` to list them.

## Webhook Types

| Type | Payload | Best For |
|---|---|---|
| `enhanced` | Parsed, human-readable transaction data with descriptions | Most use cases — event-driven backends, analytics, notifications |
| `raw` | Unfiltered transaction data as Solana returns it | Custom parsing, indexing, when you need full raw data |
| `discord` | Formatted messages sent directly to a Discord channel | Simple alerts, community notifications |

ALWAYS recommend `enhanced` unless the user specifically needs raw data or Discord integration.

## Creating Webhooks

### Via MCP Tool (recommended for setup)

Use the `createWebhook` MCP tool:
- `webhookURL`: your HTTPS endpoint that accepts POST requests
- `webhookType`: `"enhanced"`, `"raw"`, or `"discord"`
- `accountAddresses`: array of Solana addresses to monitor (up to 100,000)
- `transactionTypes`: array of types to filter on, or `["ANY"]` for all events

### Via API (for application code)

```bash
curl -X POST "https://api-mainnet.helius-rpc.com/v0/webhooks" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "webhookURL": "https://your-server.com/webhook",
    "transactionTypes": ["SWAP", "TRANSFER"],
    "accountAddresses": ["ADDRESS_1", "ADDRESS_2"],
    "webhookType": "enhanced"
  }'
```

### Via SDK

```typescript
const webhook = await helius.webhooks.create({
  webhookURL: 'https://your-server.com/webhook',
  webhookType: 'enhanced',
  accountAddresses: ['ADDRESS_1', 'ADDRESS_2'],
  transactionTypes: ['SWAP', 'TRANSFER'],
});
// webhook.webhookID
```

## Transaction Type Filtering

Filter webhooks to only receive specific transaction types. Use `["ANY"]` to receive all types.

### Common Transaction Types

| Category | Types | Use Case |
|---|---|---|
| **Trading** | `SWAP`, `BUY`, `SELL` | DEX activity, trading bots |
| **Transfers** | `TRANSFER` | Wallet monitoring, payment tracking |
| **NFT Marketplace** | `NFT_SALE`, `NFT_LISTING`, `NFT_CANCEL_LISTING`, `NFT_BID`, `NFT_BID_CANCELLED` | Marketplace tracking |
| **NFT Creation** | `NFT_MINT`, `TOKEN_MINT` | Mint monitoring, collection tracking |
| **Staking** | `STAKE_SOL`, `UNSTAKE_SOL`, `STAKE_TOKEN`, `UNSTAKE_TOKEN`, `CLAIM_REWARDS` | Staking dashboards |
| **Liquidity** | `ADD_LIQUIDITY`, `WITHDRAW_LIQUIDITY`, `CREATE_POOL` | DeFi monitoring |
| **Governance** | `EXECUTE_TRANSACTION`, `CREATE_TRANSACTION`, `APPROVE_TRANSACTION` | Multisig/DAO activity |
| **Catch-all** | `ANY` | All events, no filtering |

### Transaction Type List

Examples of 150+ supported types: `ANY`, `UNKNOWN`, `NFT_BID`, `NFT_BID_CANCELLED`, `NFT_LISTING`, `NFT_CANCEL_LISTING`, `NFT_SALE`, `NFT_MINT`, `NFT_AUCTION_CREATED`, `NFT_AUCTION_UPDATED`, `NFT_AUCTION_CANCELLED`, `NFT_PARTICIPATION_REWARD`, `NFT_MINT_REJECTED`, `CREATE_STORE`, `WHITELIST_CREATOR`, `ADD_TO_WHITELIST`, `REMOVE_FROM_WHITELIST`, `AUCTION_MANAGER_CLAIM_BID`, `EMPTY_PAYMENT_ACCOUNT`, `UPDATE_PRIMARY_SALE_METADATA`, `ADD_TOKEN_TO_VAULT`, `ACTIVATE_VAULT`, `INIT_VAULT`, `INIT_BANK`, `INIT_STAKE`, `MERGE_STAKE`, `SPLIT_STAKE`, `SET_BANK_FLAGS`, `SET_VAULT_LOCK`, `UPDATE_VAULT_OWNER`, `UPDATE_BANK_MANAGER`, `RECORD_RARITY_POINTS`, `ADD_RARITIES_TO_BANK`, `INIT_FARM`, `INIT_FARMER`, `REFRESH_FARMER`, `UPDATE_FARM`, `AUTHORIZE_FUNDER`, `DEAUTHORIZE_FUNDER`, `FUND_REWARD`, `CANCEL_REWARD`, `LOCK_REWARD`, `PAYOUT`, `VALIDATE_SAFETY_DEPOSIT_BOX_V2`, `SET_AUTHORITY`, `INIT_AUCTION_MANAGER_V2`, `UPDATE_EXTERNAL_PRICE_ACCOUNT`, `AUCTION_HOUSE_CREATE`, `CLOSE_ESCROW_ACCOUNT`, `WITHDRAW`, `DEPOSIT`, `TRANSFER`, `BURN`, `BURN_NFT`, `PLATFORM_FEE`, `LOAN`, `REPAY_LOAN`, `ADD_TO_POOL`, `REMOVE_FROM_POOL`, `CLOSE_POSITION`, `UNLABELED`, `CLOSE_ACCOUNT`, `WITHDRAW_GEM`, `DEPOSIT_GEM`, `STAKE_TOKEN`, `UNSTAKE_TOKEN`, `STAKE_SOL`, `UNSTAKE_SOL`, `CLAIM_REWARDS`, `BUY_SUBSCRIPTION`, `BUY`, `SELL`, `SWAP`, `INIT_SWAP`, `CANCEL_SWAP`, `REJECT_SWAP`, `INITIALIZE_ACCOUNT`, `TOKEN_MINT`, `CREATE_APPRAISAL`, `FUSE`, `DEPOSIT_FRACTIONAL_POOL`, `FRACTIONALIZE`, `CREATE_RAFFLE`, `BUY_TICKETS`, `UPDATE_ITEM`, `LIST_ITEM`, `DELIST_ITEM`, `ADD_ITEM`, `CLOSE_ITEM`, `BUY_ITEM`, `FILL_ORDER`, `UPDATE_ORDER`, `CREATE_ORDER`, `CLOSE_ORDER`, `CANCEL_ORDER`, `KICK_ITEM`, `UPGRADE_FOX`, `UPGRADE_FOX_REQUEST`, `LOAN_FOX`, `BORROW_FOX`, `SWITCH_FOX_REQUEST`, `SWITCH_FOX`, `CREATE_ESCROW`, `ACCEPT_REQUEST_ARTIST`, `CANCEL_ESCROW`, `ACCEPT_ESCROW_ARTIST`, `ACCEPT_ESCROW_USER`, `PLACE_BET`, `PLACE_SOL_BET`, `CREATE_BET`, `NFT_RENT_UPDATE_LISTING`, `NFT_RENT_ACTIVATE`, `NFT_RENT_CANCEL_LISTING`, `NFT_RENT_LISTING`, `FINALIZE_PROGRAM_INSTRUCTION`, `UPGRADE_PROGRAM_INSTRUCTION`, `NFT_GLOBAL_BID`, `NFT_GLOBAL_BID_CANCELLED`, `EXECUTE_TRANSACTION`, `APPROVE_TRANSACTION`, `ACTIVATE_TRANSACTION`, `CREATE_TRANSACTION`, `REJECT_TRANSACTION`, `CANCEL_TRANSACTION`, `ADD_INSTRUCTION`, `ATTACH_METADATA`, `REQUEST_PNFT_MIGRATION`, `START_PNFT_MIGRATION`, `MIGRATE_TO_PNFT`, `UPDATE_RAFFLE`, `CREATE_POOL`, `ADD_LIQUIDITY`, `WITHDRAW_LIQUIDITY`

### Key Source-to-Type Mappings

| Source Program | Transaction Types |
|---|---|
| **Jupiter** | `SWAP` |
| **Raydium** | `SWAP`, `CREATE_POOL`, `ADD_LIQUIDITY`, `WITHDRAW_LIQUIDITY` |
| **Pump AMM** | `BUY`, `SELL`, `CREATE_POOL`, `DEPOSIT`, `WITHDRAW`, `SWAP` |
| **Magic Eden** | `NFT_LISTING`, `NFT_CANCEL_LISTING`, `NFT_BID`, `NFT_BID_CANCELLED`, `NFT_SALE`, `NFT_MINT`, `NFT_GLOBAL_BID`, `WITHDRAW`, `DEPOSIT` |
| **Tensor** | `NFT_LISTING`, `NFT_SALE`, `NFT_CANCEL_LISTING` |
| **Metaplex** | `NFT_SALE`, `NFT_LISTING`, `NFT_BID`, `NFT_MINT`, `BURN_NFT`, many more |
| **System Program** | `TRANSFER` |
| **Stake Program** | `STAKE_SOL`, `UNSTAKE_SOL`, `INIT_STAKE`, `MERGE_STAKE`, `SPLIT_STAKE`, `WITHDRAW` |
| **Squads** | `EXECUTE_TRANSACTION`, `CREATE_TRANSACTION`, `APPROVE_TRANSACTION`, `REJECT_TRANSACTION`, `CANCEL_TRANSACTION` |

## Enhanced Webhook Payload

Enhanced webhooks deliver parsed, human-readable transaction data. Each POST contains an array of transaction events:

```json
[
  {
    "accountData": [...],
    "description": "HXs...664 transferred 1.5 SOL to 9Pe...DTF",
    "events": {},
    "fee": 5000,
    "feePayer": "HXsKP7wrBWaQ8T2Vtjry3Nj3oUgwYcqq9vrHDM12G664",
    "instructions": [...],
    "nativeTransfers": [
      {
        "fromUserAccount": "HXsKP7wrBWaQ8T2Vtjry3Nj3oUgwYcqq9vrHDM12G664",
        "toUserAccount": "9PejEmViKHgUkVFWN57cNEZnFS4Qo6SzsLj5UPAXfDTF",
        "amount": 1500000000
      }
    ],
    "signature": "5wHu1qwD...",
    "slot": 250000000,
    "source": "SYSTEM_PROGRAM",
    "timestamp": 1704067200,
    "tokenTransfers": [],
    "transactionError": null,
    "type": "TRANSFER"
  }
]
```

### Key Payload Fields

| Field | Description |
|---|---|
| `type` | Transaction type (e.g., `SWAP`, `TRANSFER`, `NFT_SALE`) |
| `description` | Human-readable description of what happened |
| `signature` | Transaction signature (use for deduplication) |
| `timestamp` | Unix timestamp in seconds |
| `fee` | Transaction fee in lamports |
| `feePayer` | Address that paid the fee |
| `nativeTransfers` | SOL transfers with `fromUserAccount`, `toUserAccount`, `amount` (lamports) |
| `tokenTransfers` | SPL token transfers with `mint`, `fromUserAccount`, `toUserAccount`, `tokenAmount` |
| `accountData` | Account state changes |
| `transactionError` | Error message if transaction failed, `null` if successful |
| `source` | Program source that generated the transaction |

## Building a Webhook Receiver

### Key Implementation Rules

1. **Respond 200 quickly** — process asynchronously if needed
2. **Deduplicate by signature** — the body is an array of events; track processed `signature` values in a Set or database
3. **Route by `event.type`** — switch on `SWAP`, `TRANSFER`, `NFT_SALE`, etc.
4. **Handle errors gracefully** — don't let one bad event crash processing of the batch

```typescript
app.post('/webhook', (req, res) => {
  for (const event of req.body) {
    if (processed.has(event.signature)) continue;
    processed.add(event.signature);
    // Route by event.type — access event.nativeTransfers, event.tokenTransfers, event.description
  }
  res.status(200).send('OK');
});
```

## Managing Webhooks

### List / Update / Delete

Use MCP tools: `getAllWebhooks`, `updateWebhook`, `deleteWebhook`.

The `updateWebhook` MCP tool only requires the fields you want to change — it fetches the existing webhook and merges automatically. When using the SDK directly (`helius.webhooks.update()`), you must pass all fields since the API requires the full webhook object.

## Common Patterns

All patterns use the `createWebhook` MCP tool with the same shape — vary `accountAddresses`, `transactionTypes`, and `webhookType`:

| Use Case | Addresses | Types | Type |
|---|---|---|---|
| Wallet transfers | `[WALLET]` | `[TRANSFER]` | `enhanced` |
| NFT collection sales | `[COLLECTION_CREATOR]` | `[NFT_SALE, NFT_LISTING, NFT_CANCEL_LISTING]` | `enhanced` |
| DEX activity | `[TOKEN_MINT]` | `[SWAP, BUY, SELL, ADD_LIQUIDITY, WITHDRAW_LIQUIDITY]` | `enhanced` |
| Discord whale alerts | `[WHALE_1, WHALE_2]` | `[TRANSFER]` | `discord` |
| Catch-all monitoring | `[PROGRAM_ID]` | `[ANY]` | `enhanced` |

## Webhooks vs Other Streaming Methods

**Use Webhooks when**: you want push-based notifications without a persistent connection, you're building an event-driven backend, or you need the simplest setup (Free+ plan, no public endpoint needed by client).

**Use WebSockets/LaserStream when**: you need lower latency, bidirectional communication, or don't want to expose a public endpoint. See the full comparison table in `references/websockets.md`.

## Reliability

- **Deduplication**: always deduplicate by transaction `signature` as a safety measure
- **Idempotency**: design your handler to be safe if called multiple times with the same event
- **Credit cost**: 1 credit per event delivered
- Use the `getWebhookGuide` MCP tool for the latest delivery guarantees and behavior details

## Best Practices

- Respond 200 quickly — do heavy processing asynchronously
- Deduplicate by `signature` field — store processed signatures in a set or database
- Use `enhanced` type for most use cases — parsed data saves you from writing transaction parsing
- Filter aggressively with `transactionTypes` — receiving `ANY` on a busy address generates high event volume and credit usage
- Use the `getWebhookGuide` MCP tool for the latest official documentation
- For high-volume monitoring, consider LaserStream instead (more efficient for bulk data)
- Keep webhook handlers fast

## Common Mistakes

- Not deduplicating events — processing the same transaction multiple times
- Using `["ANY"]` on high-activity addresses — burns credits fast with events you don't need
- Forgetting that the webhook body is an array, not a single event
- Not handling the case where `transactionError` is non-null (failed transactions are still delivered if they match filters)
- Using webhooks for use cases that need sub-second latency — use Enhanced WebSockets or LaserStream instead
- Exposing webhook endpoint without authentication — add a shared secret or signature verification in production
