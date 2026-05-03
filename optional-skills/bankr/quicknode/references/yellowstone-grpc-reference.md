# Yellowstone gRPC Reference

Yellowstone gRPC is a high-performance Solana Geyser plugin that enables real-time blockchain data streaming through gRPC interfaces. Available as a Marketplace add-on on Quicknode.

## Overview

| Property | Value |
|----------|-------|
| **Protocol** | gRPC (HTTP/2) |
| **Port** | 10000 |
| **Package** | `@triton-one/yellowstone-grpc` (TypeScript) |
| **Compression** | zstd supported |
| **Commitment Levels** | Processed, Confirmed, Finalized |
| **Languages** | TypeScript, Rust, Go, Python |
| **Prerequisite** | Enable [Yellowstone Geyser gRPC add-on](https://marketplace.quicknode.com/add-on/yellowstone-grpc-geyser-plugin) on your Quicknode endpoint |

## Endpoint & Authentication

### Endpoint Format

```
https://<endpoint-name>.solana-mainnet.quiknode.pro:10000
```

### Deriving Credentials

From your HTTP Provider URL:
```
https://example-guide-demo.solana-mainnet.quiknode.pro/123456789/
```

- **Endpoint**: `https://example-guide-demo.solana-mainnet.quiknode.pro:10000`
- **Token**: `123456789` (the path segment after the endpoint name)

## Installation

### TypeScript

```bash
npm install @triton-one/yellowstone-grpc
```

### Rust

```toml
[dependencies]
yellowstone-grpc-client = "11.0.0"
yellowstone-grpc-proto = "10.1.1"
tokio = { version = "1.28" }
futures = "0.3"
```

### Go

```bash
go get google.golang.org/grpc
go get google.golang.org/protobuf
```

Download proto files (`geyser.proto`, `solana-storage.proto`) from the [Yellowstone gRPC GitHub repo](https://github.com/rpcpool/yellowstone-grpc) and compile with `protoc`.

### Python

```
grpcio==1.63.0
grpcio-tools==1.63.0
protobuf==5.26.1
base58==2.1.1
```

Generate stubs:
```bash
python -m grpc_tools.protoc \
  -I./proto/ \
  --python_out=./generated \
  --pyi_out=./generated \
  --grpc_python_out=./generated \
  ./proto/*
```

## Connection Setup

### TypeScript

```typescript
import Client, { CommitmentLevel } from "@triton-one/yellowstone-grpc";

const ENDPOINT = "https://example-guide-demo.solana-mainnet.quiknode.pro:10000";
const TOKEN = "123456789";

const client = new Client(ENDPOINT, TOKEN, {});
```

### Go

```go
opts := []grpc.DialOption{
    grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})),
    grpc.WithKeepaliveParams(keepalive.ClientParameters{
        Time:                10 * time.Second,
        Timeout:             time.Second,
        PermitWithoutStream: true,
    }),
    grpc.WithDefaultCallOptions(
        grpc.MaxCallRecvMsgSize(1024 * 1024 * 1024),
        grpc.UseCompressor(gzip.Name),
    ),
    grpc.WithPerRPCCredentials(tokenAuth{token: token}),
}

conn, err := grpc.Dial(endpoint, opts...)
client := pb.NewGeyserClient(conn)
```

### Python

```python
import grpc

def create_grpc_channel(endpoint: str, token: str) -> grpc.Channel:
    endpoint = endpoint.replace('http://', '').replace('https://', '')
    auth_creds = grpc.metadata_call_credentials(
        lambda context, callback: callback((("x-token", token),), None)
    )
    ssl_creds = grpc.ssl_channel_credentials()
    combined_creds = grpc.composite_channel_credentials(ssl_creds, auth_creds)
    return grpc.secure_channel(endpoint, credentials=combined_creds)

channel = create_grpc_channel(
    "example-guide-demo.solana-mainnet.quiknode.pro:10000",
    "123456789"
)
stub = geyser_pb2_grpc.GeyserStub(channel)
```

### Rust

```rust
use yellowstone_grpc_client::GeyserGrpcClient;
use tonic::transport::ClientTlsConfig;

let client = GeyserGrpcClient::build_from_shared(endpoint.to_string())?
    .x_token(Some(token.to_string()))?
    .tls_config(ClientTlsConfig::new().with_native_roots())?
    .connect()
    .await?;
```

## Subscribe Filter Types

The `subscribe` method accepts a `SubscribeRequest` with the following filter maps:

| Filter | Key | Description |
|--------|-----|-------------|
| **accounts** | `SubscribeRequestFilterAccounts` | Account data changes by pubkey, owner, or data filters |
| **transactions** | `SubscribeRequestFilterTransactions` | Transaction events with account/vote/failure filters |
| **transactionsStatus** | `SubscribeRequestFilterTransactions` | Lightweight transaction status updates (same filter shape) |
| **slots** | `SubscribeRequestFilterSlots` | Slot progression and status changes |
| **blocks** | `SubscribeRequestFilterBlocks` | Full block data with optional transaction/account inclusion |
| **blocksMeta** | `SubscribeRequestFilterBlocksMeta` | Block metadata without full contents |
| **entry** | `SubscribeRequestFilterEntry` | PoH entry updates |

Global options on the request:

| Field | Type | Description |
|-------|------|-------------|
| `commitment` | CommitmentLevel | PROCESSED (0), CONFIRMED (1), FINALIZED (2) |
| `accountsDataSlice` | Array | Slice account data: `{ offset, length }` |
| `ping` | Object | Keepalive ping: `{ id }` |
| `from_slot` | uint64 | Replay from a specific slot |

## Transaction Filter Options

| Field | Type | Description |
|-------|------|-------------|
| `vote` | bool (optional) | Include/exclude vote transactions |
| `failed` | bool (optional) | Include/exclude failed transactions |
| `signature` | string (optional) | Filter by specific transaction signature |
| `accountInclude` | string[] | Include transactions involving these accounts |
| `accountExclude` | string[] | Exclude transactions involving these accounts |
| `accountRequired` | string[] | Require all listed accounts in the transaction |

## Account Filter Options

| Field | Type | Description |
|-------|------|-------------|
| `account` | string[] | Filter by specific account pubkeys |
| `owner` | string[] | Filter by owner program pubkeys |
| `filters` | Array | Data filters: `memcmp`, `datasize`, `token_account_state`, `lamports` |
| `nonempty_txn_signature` | bool (optional) | Only accounts with non-empty transaction signatures |

### Account Data Filters

- **memcmp**: Match bytes at a specific offset (`{ offset, bytes | base58 | base64 }`)
- **datasize**: Match accounts with exact data size
- **token_account_state**: Match valid SPL token account state
- **lamports**: Compare lamport balance (`eq`, `ne`, `lt`, `gt`)

## Available Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `subscribe` | Bidirectional stream for real-time data | SubscribeRequest (via stream) |
| `subscribeReplayInfo` | Earliest available slot for replay | None |
| `getBlockHeight` | Current block height | Optional CommitmentLevel |
| `getLatestBlockhash` | Most recent blockhash | Optional CommitmentLevel |
| `getSlot` | Current slot number | Optional CommitmentLevel |
| `getVersion` | Geyser plugin version info | None |
| `isBlockhashValid` | Check blockhash validity | blockhash (string), optional CommitmentLevel |
| `ping` | Connection health check | count (integer) |

## Subscription Examples

### Account Updates

```typescript
import Client, { CommitmentLevel } from "@triton-one/yellowstone-grpc";

const client = new Client(ENDPOINT, TOKEN, {});
const stream = await client.subscribe();

stream.on("data", (data) => {
  if (data.account) {
    const account = data.account;
    console.log("Account updated:", {
      pubkey: Buffer.from(account.account.pubkey).toString("hex"),
      lamports: account.account.lamports,
      slot: account.slot,
      owner: Buffer.from(account.account.owner).toString("hex"),
    });
  }
});

stream.on("error", (error) => {
  console.error("Stream error:", error);
});

await new Promise<void>((resolve, reject) => {
  stream.write(
    {
      accounts: {
        account_filter: {
          account: ["ACCOUNT_PUBKEY"],
          owner: [],
          filters: [],
        },
      },
      slots: {},
      transactions: {},
      transactionsStatus: {},
      entry: {},
      blocks: {},
      blocksMeta: {},
      accountsDataSlice: [],
      ping: undefined,
      commitment: CommitmentLevel.CONFIRMED,
    },
    (err) => {
      if (err) reject(err);
      else resolve();
    }
  );
});
```

### Transaction Streaming

```typescript
import Client, { CommitmentLevel } from "@triton-one/yellowstone-grpc";

const client = new Client(ENDPOINT, TOKEN, {});
const stream = await client.subscribe();

stream.on("data", (data) => {
  if (data.transaction) {
    const txn = data.transaction;
    console.log("Transaction:", {
      signature: Buffer.from(txn.transaction.signature).toString("base64"),
      slot: txn.slot,
      isVote: txn.transaction.isVote,
    });
  }
});

await new Promise<void>((resolve, reject) => {
  stream.write(
    {
      accounts: {},
      slots: {},
      transactions: {
        txn_filter: {
          vote: false,
          failed: false,
          accountInclude: ["PROGRAM_OR_ACCOUNT_PUBKEY"],
          accountExclude: [],
          accountRequired: [],
        },
      },
      transactionsStatus: {},
      entry: {},
      blocks: {},
      blocksMeta: {},
      accountsDataSlice: [],
      ping: undefined,
      commitment: CommitmentLevel.CONFIRMED,
    },
    (err) => {
      if (err) reject(err);
      else resolve();
    }
  );
});
```

### Slot Updates

```typescript
import Client, { CommitmentLevel } from "@triton-one/yellowstone-grpc";

const client = new Client(ENDPOINT, TOKEN, {});
const stream = await client.subscribe();

stream.on("data", (data) => {
  if (data.slot) {
    console.log("Slot:", {
      slot: data.slot.slot,
      parent: data.slot.parent,
      status: data.slot.status,
    });
  }
});

await new Promise<void>((resolve, reject) => {
  stream.write(
    {
      accounts: {},
      slots: {
        slot_filter: {
          filterByCommitment: true,
        },
      },
      transactions: {},
      transactionsStatus: {},
      entry: {},
      blocks: {},
      blocksMeta: {},
      accountsDataSlice: [],
      ping: undefined,
      commitment: CommitmentLevel.CONFIRMED,
    },
    (err) => {
      if (err) reject(err);
      else resolve();
    }
  );
});
```

### Unary RPC Methods

```typescript
import Client, { CommitmentLevel } from "@triton-one/yellowstone-grpc";

const client = new Client(ENDPOINT, TOKEN, {});

// Block height
const blockHeight = await client.getBlockHeight();
console.log("Block height:", blockHeight);

// Latest blockhash
const blockhash = await client.getLatestBlockhash(CommitmentLevel.CONFIRMED);
console.log("Blockhash:", blockhash);

// Current slot
const slot = await client.getSlot();
console.log("Slot:", slot);

// Version info
const version = await client.getVersion();
console.log("Version:", version);

// Validate blockhash
const valid = await client.isBlockhashValid(blockhash.blockhash);
console.log("Valid:", valid);

// Ping
const pong = await client.ping(1);
console.log("Pong:", pong);

// Replay info
const replayInfo = await client.subscribeReplayInfo({});
console.log("First available slot:", replayInfo.firstAvailable);
```

## Stream Handling

### Async Iteration Pattern

```typescript
const stream = await client.subscribe();

// Write subscription request
stream.write(subscribeRequest);

// Process updates
stream.on("data", (update) => {
  if (update.account) handleAccount(update);
  if (update.transaction) handleTransaction(update);
  if (update.slot) handleSlot(update);
  if (update.block) handleBlock(update);
  if (update.blockMeta) handleBlockMeta(update);
  if (update.entry) handleEntry(update);
  if (update.pong) handlePong(update);
});

stream.on("error", (error) => {
  console.error("Stream error:", error);
  // Implement reconnection logic
});

stream.on("end", () => {
  console.log("Stream ended");
  // Implement reconnection logic
});
```

### Keepalive Pings

```typescript
// Send periodic pings to keep the connection alive
const pingInterval = setInterval(() => {
  stream.write({
    ping: { id: Date.now() },
  });
}, 10000); // every 10 seconds

// Clean up on stream end
stream.on("end", () => clearInterval(pingInterval));
```

### Reconnection with Backoff

```typescript
async function connectWithRetry(maxRetries = 5) {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      const client = new Client(ENDPOINT, TOKEN, {});
      const stream = await client.subscribe();
      stream.write(subscribeRequest);
      return stream;
    } catch (error) {
      attempt++;
      const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
      console.error(`Connection failed (attempt ${attempt}), retrying in ${delay}ms`);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw new Error("Max retries exceeded");
}
```

## Best Practices

1. **Use narrow filters** — Subscribe only to accounts, programs, or transaction patterns you need. Broad filters increase bandwidth and processing overhead.
2. **Set appropriate commitment levels** — Use `CONFIRMED` for most use cases. Use `FINALIZED` when you need irreversibility guarantees. Avoid `PROCESSED` unless you need the lowest latency and can handle rollbacks.
3. **Implement reconnection logic** — gRPC streams can drop due to network issues or server maintenance. Always implement exponential backoff reconnection.
4. **Enable zstd compression** — Reduces bandwidth significantly for high-throughput subscriptions.
5. **Test on devnet first** — Validate your filter logic and stream handling on devnet before deploying to mainnet.
6. **Use `accountsDataSlice`** — When you only need part of an account's data, slice it to reduce payload size.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused on port 10000 | Yellowstone add-on not enabled | Enable the Yellowstone Geyser gRPC add-on in the Quicknode dashboard |
| Authentication failed | Invalid or missing token | Extract the token from your HTTP Provider URL (path segment after endpoint name) |
| No data received | Filters too restrictive or wrong commitment level | Start with broad filters and narrow down; check commitment level |
| Stream drops frequently | No keepalive pings | Send periodic pings (every 10s) and implement reconnection logic |
| Large payloads / high bandwidth | Subscribing to too much data | Narrow filters, use `accountsDataSlice`, enable zstd compression |
| Stale data | Using `PROCESSED` commitment | Switch to `CONFIRMED` or `FINALIZED` |

## Documentation

- **Yellowstone gRPC Overview**: https://www.quicknode.com/docs/solana/yellowstone-grpc/overview
- **Subscribe Method**: https://www.quicknode.com/docs/solana/yellowstone-grpc/subscribe
- **TypeScript Setup**: https://www.quicknode.com/docs/solana/yellowstone-grpc/overview/typescript
- **Go Setup**: https://www.quicknode.com/docs/solana/yellowstone-grpc/overview/go
- **Rust Setup**: https://www.quicknode.com/docs/solana/yellowstone-grpc/overview/rust
- **Python Setup**: https://www.quicknode.com/docs/solana/yellowstone-grpc/overview/python
- **Marketplace Add-on**: https://marketplace.quicknode.com/add-on/yellowstone-grpc-geyser-plugin
- **Guides**: https://www.quicknode.com/guides/tags/geyser
