---
id: references/solana-rpc.md
name: 'Solana JSON-RPC'
description: 'Standard Solana JSON-RPC endpoints for account, program, and transaction data.'
tags:
  - alchemy
  - solana
  - json-rpc
related:
  - solana-das-api.md
  - solana-wallets.md
updated: 2026-02-23
---
# Solana JSON-RPC

Standard Solana JSON-RPC methods via Alchemy's node infrastructure. All requests are `POST` with JSON-RPC 2.0 bodies.

**Base URL**: `https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY`

**Testnet**: `https://solana-devnet.g.alchemy.com/v2/$ALCHEMY_API_KEY`

---

## `getBalance`

Returns the SOL balance for an account.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pubkey` | string | Yes | Base58-encoded account public key |
| `config.commitment` | string | No | `"finalized"` (default), `"confirmed"`, or `"processed"` |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getBalance",
    "params": ["83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri"]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "context": { "slot": 250000000, "apiVersion": "1.18.0" },
    "value": 1500000000
  }
}
```

Result `value` is in lamports (1 SOL = 10^9 lamports).

---

## `getAccountInfo`

Returns account data including owner program and lamport balance.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pubkey` | string | Yes | — | Base58 public key |
| `config.encoding` | string | No | `"base64"` | `"base64"`, `"base58"`, `"jsonParsed"` |
| `config.commitment` | string | No | `"finalized"` | Commitment level |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAccountInfo",
    "params": [
      "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri",
      { "encoding": "jsonParsed" }
    ]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "context": { "slot": 250000000 },
    "value": {
      "data": { "parsed": { "type": "account", "info": {} }, "program": "system", "space": 0 },
      "executable": false,
      "lamports": 1500000000,
      "owner": "11111111111111111111111111111111",
      "rentEpoch": 361,
      "space": 0
    }
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `value.data` | object or string | Account data (format depends on `encoding`) |
| `value.executable` | boolean | Whether account is executable (program) |
| `value.lamports` | integer | Balance in lamports |
| `value.owner` | string | Owner program (base58) |
| `value.rentEpoch` | integer | Epoch at which rent was last collected |
| `value.space` | integer | Data size in bytes |

---

## `getSignaturesForAddress`

Returns confirmed signatures (transaction hashes) for an address.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pubkey` | string | Yes | — | Base58 public key |
| `config.limit` | integer | No | `1000` | Max signatures to return (max 1000) |
| `config.before` | string | No | — | Return signatures before this signature |
| `config.until` | string | No | — | Return signatures until this signature |
| `config.commitment` | string | No | `"finalized"` | Commitment level |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getSignaturesForAddress",
    "params": [
      "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri",
      { "limit": 5 }
    ]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": [
    {
      "signature": "5UfDuX7WXYNqMnT3fXmJBk8ZH...",
      "slot": 250000000,
      "blockTime": 1717200000,
      "confirmationStatus": "finalized",
      "err": null,
      "memo": null
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `signature` | string | Transaction signature (base58) |
| `slot` | integer | Slot number |
| `blockTime` | integer | Unix timestamp (seconds) |
| `confirmationStatus` | string | `"finalized"`, `"confirmed"`, `"processed"` |
| `err` | object | Error if transaction failed (null on success) |
| `memo` | string | Memo data (null if none) |

---

## `getTransaction`

Returns details for a confirmed transaction.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `signature` | string | Yes | — | Transaction signature (base58) |
| `config.encoding` | string | No | `"json"` | `"json"`, `"jsonParsed"`, `"base64"` |
| `config.commitment` | string | No | `"finalized"` | Commitment level |
| `config.maxSupportedTransactionVersion` | integer | No | — | Set to `0` for versioned transactions |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getTransaction",
    "params": [
      "5UfDuX7WXYNqMnT3fXmJBk8ZH...",
      { "encoding": "jsonParsed", "maxSupportedTransactionVersion": 0 }
    ]
  }'
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `slot` | integer | Slot number |
| `blockTime` | integer | Unix timestamp |
| `transaction.message.accountKeys` | array | Accounts involved |
| `transaction.message.instructions` | array | Program instructions |
| `transaction.signatures` | string[] | Signatures |
| `meta.fee` | integer | Transaction fee (lamports) |
| `meta.preBalances` | integer[] | Pre-execution balances |
| `meta.postBalances` | integer[] | Post-execution balances |
| `meta.preTokenBalances` | array | Pre-execution token balances |
| `meta.postTokenBalances` | array | Post-execution token balances |
| `meta.logMessages` | string[] | Program log messages |
| `meta.err` | object | Error (null on success) |

---

## Other Common Methods

| Method | Description |
|--------|-------------|
| `getSlot` | Current slot number |
| `getBlockHeight` | Current block height |
| `getRecentBlockhash` | Recent blockhash for transaction signing |
| `getLatestBlockhash` | Latest blockhash with expiry info |
| `sendTransaction` | Submit a signed transaction |
| `simulateTransaction` | Simulate a transaction |
| `getProgramAccounts` | Accounts owned by a program |
| `getTokenAccountsByOwner` | SPL token accounts for a wallet |

---

## Notes

- Solana uses base58-encoded addresses (not hex).
- Account data is base64-encoded by default. Use `"jsonParsed"` encoding for human-readable output.
- Always set `commitment` explicitly for consistency (`"finalized"` is safest).
- For versioned transactions, set `maxSupportedTransactionVersion: 0`.

## Official Docs
- [Solana API Quickstart](https://www.alchemy.com/docs/reference/solana-api-quickstart)
- [Solana API Endpoints](https://www.alchemy.com/docs/chains/solana/solana-api-endpoints)
