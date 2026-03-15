# Enhanced Transactions — Human-Readable Transaction Data

## What This Covers

The Enhanced Transactions API transforms raw Solana transactions into structured, human-readable data. Instead of decoding instruction bytes and account lists manually, you get transaction types, descriptions, transfers, events, and metadata parsed automatically. Credit cost: 100 credits per call.

Two endpoints:
- **Parse transactions**: `POST /v0/transactions/?api-key=KEY` — parse known signatures
- **Transaction history**: `GET /v0/addresses/{address}/transactions?api-key=KEY` — fetch + parse history for an address

## MCP Tools

ALWAYS use these MCP tools for transaction analysis. Only generate raw API code when the user is building an application.

| MCP Tool | What It Does | Credits |
|---|---|---|
| `parseTransactions` | Parse signatures into human-readable format. Returns type, source program, transfers, fees, description. Use `showRaw: true` for instruction-level data. | 100/call |
| `getTransactionHistory` | Get transaction history for a wallet. Three modes: `parsed` (default, human-readable), `signatures` (lightweight list), `raw` (full data with advanced filters). | ~110 (parsed), ~10 (signatures/raw) |

Related tool (Wallet API, covered in `helius-wallet-api.md`):

| MCP Tool | What It Does | Credits |
|---|---|---|
| `getWalletHistory` | Transaction history with balance changes per tx. Simpler pagination, different response format. | 100/call |

### When to Use Which

| You want to... | Use this |
|---|---|
| Parse specific transaction signatures | `parseTransactions` |
| Get a wallet's recent activity (human-readable) | `getTransactionHistory` (mode: `parsed`) |
| Get a lightweight list of signatures for a wallet | `getTransactionHistory` (mode: `signatures`) |
| Filter by time range, slot range, or status | `getTransactionHistory` (mode: `raw`) |
| See balance changes per transaction | `getWalletHistory` (Wallet API) |
| Debug raw instruction data | `parseTransactions` with `showRaw: true` |

## parseTransactions

Parses one or more transaction signatures into structured data.

**Parameters:**
- `signatures`: array of base58-encoded transaction signatures
- `showRaw` (default: `false`): include raw instruction data (program IDs, accounts, inner instructions, decoded ComputeBudget instructions)

**What you get back:**
- `description`: plain-English summary ("Transfer 0.1 SOL to FXv...")
- `type`: transaction category (`TRANSFER`, `SWAP`, `NFT_SALE`, etc.)
- `source`: program that executed it (`SYSTEM_PROGRAM`, `JUPITER`, `RAYDIUM`, `MAGIC_EDEN`, etc.)
- `fee` / `feePayer`: transaction fees in SOL and lamports
- `timestamp`: when the transaction was processed
- `nativeTransfers`: SOL movements between accounts
- `tokenTransfers`: SPL token movements with token names, symbols, and proper decimal formatting
- `events`: high-level event summaries (swap details, sale details, etc.)
- `accountData`: account balance changes
- Raw instruction data (when `showRaw: true`)

### Response Structure

```json
{
  "description": "Transfer 0.1 SOL to FXvStt8aeQHMGKDgqaQ2HXWfJsXnqiKSoBEpHJahkuD",
  "type": "TRANSFER",
  "source": "SYSTEM_PROGRAM",
  "fee": 5000,
  "feePayer": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
  "signature": "5rfFLBUp5YPr6rC2g...",
  "slot": 171341028,
  "timestamp": 1674080473,
  "nativeTransfers": [
    {
      "fromUserAccount": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
      "toUserAccount": "FXvStt8aeQHMGKDgqaQ2HXWfJsXnqiKSoBEpHJahkuD",
      "amount": 100000000
    }
  ],
  "tokenTransfers": [],
  "events": {}
}
```

## getTransactionHistory

Three modes with different tradeoffs:

### Mode: `parsed` (default)

Human-readable decoded history. Two-step process internally: fetches signatures, then enriches via the Enhanced API.

**Key parameters:**
- `address`: wallet address
- `limit` (1-100, default: 10)
- `sortOrder`: `"desc"` (newest first, default) or `"asc"` (oldest first — good for finding funding sources)
- `status`: `"succeeded"` (default), `"failed"`, or `"any"`
- `paginationToken`: cursor from previous response for next page

### Mode: `signatures`

Lightweight signature list with slot, time, and status. Much cheaper (~10 credits).

**Key parameters:**
- Same as parsed, plus `limit` up to 1000
- `before`: cursor — start searching backwards from this signature (desc only)
- `until`: cursor — search until this signature (desc only)

### Mode: `raw`

Full transaction data with advanced Helius filters. Cheapest for bulk data (~10 credits).

**Key parameters:**
- All parsed parameters, plus:
- `transactionDetails`: `"signatures"` (basic info, up to 1000) or `"full"` (complete data, up to 100)
- `tokenAccounts`: `"none"` | `"balanceChanged"` | `"all"` (see Token Account Filtering below)
- `blockTimeGte` / `blockTimeLte`: Unix timestamp range filters
- `slotGte` / `slotLte`: slot range filters

## Token Account Filtering

Controls whether the history includes transactions that only touched associated token accounts (ATAs):

| Value | Behavior | Use for |
|---|---|---|
| `none` (default for raw) | Only direct wallet interactions | Simple activity view |
| `balanceChanged` | Include transactions that changed token balances | Clean token transfer history (filters spam) |
| `all` | All token account activity | Complete audit trail (includes spam) |

`balanceChanged` is recommended for most use cases — it captures meaningful token activity while filtering noise.

**Limitation**: The `token-accounts` filter relies on the `owner` field in token balance metadata, which was not available before slot 111,491,819 (~December 2022). Older token account transactions may be missing from `balanceChanged` and `all` results.

## Transaction Types

The Enhanced Transactions API categorizes transactions by type. Common types:

| Type | Description |
|---|---|
| `TRANSFER` | SOL or token transfer |
| `SWAP` | Token swap (Jupiter, Raydium, etc.) |
| `NFT_SALE` | NFT sold |
| `NFT_LISTING` | NFT listed for sale |
| `NFT_BID` | Bid placed on NFT |
| `NFT_MINT` | NFT minted |
| `NFT_CANCEL_LISTING` | NFT listing cancelled |
| `TOKEN_MINT` | Token minted |
| `BURN` | Token burned |
| `STAKE_SOL` / `UNSTAKE_SOL` | SOL staking/unstaking |
| `ADD_LIQUIDITY` / `WITHDRAW_LIQUIDITY` | LP operations |
| `COMPRESSED_NFT_MINT` / `COMPRESSED_NFT_TRANSFER` | cNFT operations |

A longer list of 138+ types is shared with the Webhooks system — see Helius docs at `docs.helius.dev` for the complete transaction type reference and source-to-type mappings.

## Source Programs

The `source` field identifies which program executed the transaction:

Common sources: `SYSTEM_PROGRAM`, `JUPITER`, `RAYDIUM`, `ORCA`, `MAGIC_EDEN`, `TENSOR`, `DFLOW`, `JITO`, `METAPLEX`, `PUMP_FUN`, and many more.

## Time and Slot Filtering

Available in `raw` mode via `getTransactionHistory`:

```typescript
// Last 24 hours
const oneDayAgo = Math.floor(Date.now() / 1000) - 86400;
// Use: blockTimeGte = oneDayAgo

// Specific date range
const start = Math.floor(new Date('2024-01-01').getTime() / 1000);
const end = Math.floor(new Date('2024-01-31').getTime() / 1000);
// Use: blockTimeGte = start, blockTimeLte = end

// Slot range
// Use: slotGte = 148000000, slotLte = 148100000
```

Time and slot filters cannot be combined in the same request.

## Pagination

### Parsed and Raw Modes

Use `paginationToken` from the previous response:

```typescript
// First page
const page1 = await getTransactionHistory({ address, mode: 'parsed', limit: 25 });
// Next page
const page2 = await getTransactionHistory({ address, mode: 'parsed', limit: 25, paginationToken: page1.paginationToken });
```

### Signatures Mode (desc only)

Use `before` with the last signature from the previous page:

```typescript
const page1 = await getTransactionHistory({ address, mode: 'signatures', limit: 100 });
const lastSig = page1.signatures[page1.signatures.length - 1].signature;
const page2 = await getTransactionHistory({ address, mode: 'signatures', limit: 100, before: lastSig });
```

## Runtime Type Filtering

When using the `type` parameter on the REST API directly, filtering happens at runtime — the API searches sequentially until it finds matches. If no matches exist in the current search window, the API returns an error with a continuation signature:

```json
{
  "error": "Failed to find events within the search period. To continue search, query the API again with the `before-signature` parameter set to <signature>."
}
```

Handle this by extracting the continuation signature and retrying. Use `before-signature` for descending order, `after-signature` for ascending. Implement a max retry limit to prevent infinite loops.

The MCP `getTransactionHistory` tool handles this automatically in parsed mode.

## Common Patterns

- **Parse a specific tx**: use `parseTransactions` MCP tool, or `POST /v0/transactions/?api-key=KEY` with `{ transactions: [sig] }`
- **Recent wallet history**: use `getTransactionHistory` MCP tool (mode: `parsed`), or `GET /v0/addresses/{addr}/transactions?api-key=KEY`
- **Paginate full history**: loop with `before-signature` param set to `batch[batch.length - 1].signature`, break when response is empty
- **Filter by type**: append `&type=SWAP&token-accounts=balanceChanged` to the history URL
- **Oldest transactions first**: use `sort-order=asc` — no need to paginate to the end

## SDK Methods & Parameter Names

The SDK exposes **two different methods** for transaction history with **different parameter names**. Mixing them up causes silent bugs.

### Method 1: `helius.enhanced.getTransactionsByAddress()` — Enhanced API

Direct wrapper around the Enhanced Transactions REST API. Returns parsed `EnhancedTransaction[]`.

```typescript
import type { GetEnhancedTransactionsByAddressRequest } from 'helius-sdk';

const history = await helius.enhanced.getTransactionsByAddress({
  address: 'WalletAddress',
  limit: 100,
  sortOrder: 'desc',
  beforeSignature: 'lastSigFromPreviousPage',  // NOT "before"
  // afterSignature: 'sig',  // for ascending pagination
  // type: TransactionType.SWAP,
  // source: TransactionSource.JUPITER,
  gteTime: Math.floor(new Date('2025-01-01').getTime() / 1000),
  lteTime: Math.floor(new Date('2025-01-31').getTime() / 1000),
  // gteSlot: 250000000,
  // lteSlot: 251000000,
});
```

**Pagination**: use `beforeSignature` (desc) or `afterSignature` (asc) with the last signature from the previous page.

### Method 2: `helius.getTransactionsForAddress()` — RPC-based

Uses `getSignaturesForAddress` + enrichment. Supports filters object with nested comparison operators.

```typescript
const history = await helius.getTransactionsForAddress(
  'WalletAddress',
  {
    limit: 25,
    sortOrder: 'desc',
    transactionDetails: 'full',
    paginationToken: 'tokenFromPreviousResponse',
    filters: {
      status: 'succeeded',
      tokenAccounts: 'balanceChanged',
      blockTime: { gte: startTimestamp, lte: endTimestamp },
      slot: { gte: 250000000 },
    },
  }
);
```

**Pagination**: use `paginationToken` from the previous response.

### Parameter Name Mapping

| Concept | REST API (kebab-case) | Enhanced SDK method | RPC SDK method | MCP tool param |
|---|---|---|---|---|
| Pagination cursor (backward) | `before-signature` | `beforeSignature` | `paginationToken` | `paginationToken` or `before` |
| Pagination cursor (forward) | `after-signature` | `afterSignature` | — | — |
| Time range (start) | `gte-time` | `gteTime` | `filters.blockTime.gte` | `blockTimeGte` |
| Time range (end) | `lte-time` | `lteTime` | `filters.blockTime.lte` | `blockTimeLte` |
| Slot range (start) | `gte-slot` | `gteSlot` | `filters.slot.gte` | `slotGte` |
| Slot range (end) | `lte-slot` | `lteSlot` | `filters.slot.lte` | `slotLte` |
| Sort order | `sort-order` | `sortOrder` | `sortOrder` | `sortOrder` |
| Token account filter | `token-accounts` | — | `filters.tokenAccounts` | `tokenAccounts` |

### SDK Pagination Examples

**Enhanced API — paginate all transactions in a date range:**

```typescript
import type { EnhancedTransaction } from 'helius-sdk';

async function getAllTransactions(
  address: string,
  startTime: number,
  endTime: number,
): Promise<EnhancedTransaction[]> {
  const all: EnhancedTransaction[] = [];
  let beforeSignature: string | undefined;

  while (true) {
    const batch = await helius.enhanced.getTransactionsByAddress({
      address,
      limit: 100,
      sortOrder: 'desc',
      gteTime: startTime,
      lteTime: endTime,
      ...(beforeSignature && { beforeSignature }),
    });

    if (batch.length === 0) break;
    all.push(...batch);
    beforeSignature = batch[batch.length - 1].signature;
  }

  return all;
}
```

**RPC method — paginate with paginationToken:**

```typescript
let paginationToken: string | undefined;
const all = [];

while (true) {
  const result = await helius.getTransactionsForAddress('address', {
    limit: 100,
    transactionDetails: 'full',
    filters: { tokenAccounts: 'balanceChanged' },
    ...(paginationToken && { paginationToken }),
  });

  if (result.transactions.length === 0) break;
  all.push(...result.transactions);
  paginationToken = result.paginationToken;
  if (!paginationToken) break;
}
```

### Parse Transactions

```typescript
const parsed = await helius.enhanced.getTransactions({ transactions: ['sig1', 'sig2'] });
```

## Common Mistakes

- **Using `before` instead of `beforeSignature`** — The Enhanced SDK method uses `beforeSignature` (camelCase). Using `before` silently does nothing because JavaScript destructuring ignores unknown keys. This causes infinite pagination loops returning page 1 repeatedly. Always import and use the `GetEnhancedTransactionsByAddressRequest` type to catch this at compile time.
- **Using `any` for SDK params** — Casting params as `any` disables TypeScript's ability to catch name mismatches. Always use the proper request types: `GetEnhancedTransactionsByAddressRequest`, `GetEnhancedTransactionsRequest`, or `GetTransactionsForAddressConfigFull`.
- **Mixing up the two SDK methods** — `helius.enhanced.getTransactionsByAddress()` uses `beforeSignature`/`afterSignature` for pagination. `helius.getTransactionsForAddress()` uses `paginationToken`. They are NOT interchangeable.
- Using raw RPC `getTransaction` when you could use `parseTransactions` for human-readable data — Enhanced Transactions saves significant parsing work
- Not handling the runtime type filtering continuation pattern — the API may return an error with a continuation signature instead of results
- Using `tokenAccounts: "all"` when `"balanceChanged"` would filter spam
- Confusing `getTransactionHistory` (Enhanced Transactions API, 100 credits, parsed data) with `getWalletHistory` (Wallet API, 100 credits, balance changes per tx) — they return different response formats
- Expecting type filtering to work pre-December 2022 with `tokenAccounts` — the `owner` metadata wasn't available before slot 111,491,819
- Not paginating — high-volume wallets can have thousands of transactions
