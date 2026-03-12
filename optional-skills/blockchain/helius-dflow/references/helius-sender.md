# Helius Sender — Transaction Submission

## When To Use

ALWAYS use Helius Sender for transaction submission instead of the standard `sendTransaction` to a regular RPC endpoint. Sender dual-routes transactions to both Solana validators and Jito simultaneously, maximizing block inclusion probability with ultra-low latency.

- Available on ALL plans, including free tier
- Consumes ZERO API credits
- Default 50 TPS (Professional plan users can request higher limits)
- For simpler use cases where you do not need manual control, the Helius TypeScript SDK provides `sendSmartTransaction` which handles priority fees, compute units, and retries automatically — but it does NOT use Sender endpoints. For maximum performance, use Sender via the SDK's `sendTransactionWithSender` method, or directly as described below.

## Mandatory Requirements

Every Sender transaction MUST include all three of these or it will be rejected:

### 1. Skip Preflight

```typescript
{ skipPreflight: true, maxRetries: 0 }
```

`skipPreflight` MUST be `true`. Set `maxRetries: 0` and implement your own retry logic.

### 2. Jito Tip

A SOL transfer instruction to one of the designated tip accounts. Pick one randomly per transaction to distribute load.

**Minimum tip amounts:**
- Default dual routing: **0.0002 SOL** (200,000 lamports)
- SWQOS-only mode: **0.000005 SOL** (5,000 lamports)

**Mainnet tip accounts:**
```
4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE
D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ
9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta
5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn
2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD
2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ
wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF
3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT
4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey
4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or
```

For dynamic tip sizing, fetch the 75th percentile from the Jito API and use `Math.max(tip75th, 0.0002)`:

```typescript
async function getDynamicTipAmount(): Promise<number> {
  try {
    const response = await fetch('https://bundles.jito.wtf/api/v1/bundles/tip_floor');
    const data = await response.json();
    if (data?.[0]?.landed_tips_75th_percentile) {
      return Math.max(data[0].landed_tips_75th_percentile, 0.0002);
    }
    return 0.0002;
  } catch {
    return 0.0002;
  }
}
```

### 3. Priority Fee

A `ComputeBudgetProgram.setComputeUnitPrice` instruction. Use the `getPriorityFeeEstimate` MCP tool to get the right fee — never hardcode.

Also include `ComputeBudgetProgram.setComputeUnitLimit` set to the actual compute units needed (simulate first, then add a 10% margin). Do NOT use the default 200,000 CU — a tighter limit means lower total cost and better priority.

## Endpoints

### Frontend (HTTPS — use for browser apps)

```
https://sender.helius-rpc.com/fast
```

Auto-routes to the nearest location. Avoids CORS preflight failures that occur with regional HTTP endpoints.

### Backend (Regional HTTP — use for servers)

Choose the endpoint closest to your infrastructure:

```
http://slc-sender.helius-rpc.com/fast      # Salt Lake City
http://ewr-sender.helius-rpc.com/fast      # Newark
http://lon-sender.helius-rpc.com/fast      # London
http://fra-sender.helius-rpc.com/fast      # Frankfurt
http://ams-sender.helius-rpc.com/fast      # Amsterdam
http://sg-sender.helius-rpc.com/fast       # Singapore
http://tyo-sender.helius-rpc.com/fast      # Tokyo
```

### SWQOS-Only Mode

Append `?swqos_only=true` to any endpoint URL for cost-optimized routing. Routes exclusively through SWQOS infrastructure with a lower 0.000005 SOL minimum tip. Use this when cost matters more than maximum inclusion speed.

```
https://sender.helius-rpc.com/fast?swqos_only=true
```

### Custom TPS (Professional plan)

If approved for higher TPS, append your Sender-specific API key:

```
https://sender.helius-rpc.com/fast?api-key=YOUR_SENDER_API_KEY
```

### Request Format

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "sendTransaction",
  "params": [
    "BASE64_ENCODED_TRANSACTION",
    {
      "encoding": "base64",
      "skipPreflight": true,
      "maxRetries": 0
    }
  ]
}
```

## Implementation Pattern — Basic Send (@solana/web3.js)

When building a basic Sender transaction with `@solana/web3.js`, follow this pattern:

```typescript
import {
  Connection,
  TransactionMessage,
  VersionedTransaction,
  SystemProgram,
  PublicKey,
  Keypair,
  LAMPORTS_PER_SOL,
  ComputeBudgetProgram,
  TransactionInstruction
} from '@solana/web3.js';

const TIP_ACCOUNTS = [
  "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
  "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
  "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
  "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
  "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
  "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
  "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
  "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
  "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
  "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or"
];

async function sendViaSender(
  keypair: Keypair,
  instructions: TransactionInstruction[],
  connection: Connection
): Promise<string> {
  // 1. Get blockhash
  const { value: { blockhash, lastValidBlockHeight } } =
    await connection.getLatestBlockhashAndContext('confirmed');

  // 2. Get dynamic tip
  const tipAmountSOL = await getDynamicTipAmount();
  const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];

  // 3. Build all instructions: compute budget + user instructions + tip
  const allInstructions = [
    ComputeBudgetProgram.setComputeUnitLimit({ units: 200_000 }), // placeholder, refine via simulation
    ComputeBudgetProgram.setComputeUnitPrice({ microLamports: 200_000 }), // use getPriorityFeeEstimate for production
    ...instructions,
    SystemProgram.transfer({
      fromPubkey: keypair.publicKey,
      toPubkey: new PublicKey(tipAccount),
      lamports: Math.floor(tipAmountSOL * LAMPORTS_PER_SOL),
    }),
  ];

  // 4. Build and sign
  const transaction = new VersionedTransaction(
    new TransactionMessage({
      instructions: allInstructions,
      payerKey: keypair.publicKey,
      recentBlockhash: blockhash,
    }).compileToV0Message()
  );
  transaction.sign([keypair]);

  // 5. Submit to Sender
  const response = await fetch('https://sender.helius-rpc.com/fast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: Date.now().toString(),
      method: 'sendTransaction',
      params: [
        Buffer.from(transaction.serialize()).toString('base64'),
        { encoding: 'base64', skipPreflight: true, maxRetries: 0 }
      ]
    })
  });

  const json = await response.json();
  if (json.error) throw new Error(json.error.message);
  return json.result;
}
```

## Implementation Pattern — Basic Send (@solana/kit)

When building with the newer, and recommended, `@solana/kit`:

```typescript
import { pipe } from "@solana/kit";
import {
  createSolanaRpc,
  createTransactionMessage,
  setTransactionMessageFeePayerSigner,
  setTransactionMessageLifetimeUsingBlockhash,
  appendTransactionMessageInstruction,
  signTransactionMessageWithSigners,
  lamports,
  getBase64EncodedWireTransaction,
  address,
} from "@solana/kit";
import { getTransferSolInstruction } from "@solana-program/system";
import {
  getSetComputeUnitLimitInstruction,
  getSetComputeUnitPriceInstruction,
} from "@solana-program/compute-budget";

async function sendViaSender(
  signer: KeyPairSigner,
  instructions: IInstruction[],
  rpc: Rpc
): Promise<string> {
  const { value: blockhash } = await rpc.getLatestBlockhash().send();

  const tipAmountSOL = await getDynamicTipAmount();
  const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];

  // Build transaction: compute budget, user instructions, tip
  let tx = pipe(
    createTransactionMessage({ version: 0 }),
    (m) => setTransactionMessageFeePayerSigner(signer, m),
    (m) => setTransactionMessageLifetimeUsingBlockhash(blockhash, m),
    (m) => appendTransactionMessageInstruction(getSetComputeUnitLimitInstruction({ units: 200_000 }), m),
    (m) => appendTransactionMessageInstruction(getSetComputeUnitPriceInstruction({ microLamports: 200_000 }), m),
  );

  // Append user instructions
  for (const ix of instructions) {
    tx = appendTransactionMessageInstruction(ix, tx);
  }

  // Append tip
  tx = appendTransactionMessageInstruction(
    getTransferSolInstruction({
      source: signer,
      destination: address(tipAccount),
      amount: lamports(BigInt(Math.floor(tipAmountSOL * 1_000_000_000))),
    }),
    tx
  );

  const signedTx = await signTransactionMessageWithSigners(tx);
  const base64Tx = getBase64EncodedWireTransaction(signedTx);

  const res = await fetch("https://sender.helius-rpc.com/fast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: Date.now().toString(),
      method: "sendTransaction",
      params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
    }),
  });

  const { result, error } = await res.json();
  if (error) throw new Error(error.message);
  return result;
}
```

## Production Pattern — Dynamic Optimization

For production use, add these optimizations on top of the basic pattern:

### 1. Simulate to get actual compute units

```typescript
// Build a test transaction with max CU limit for simulation
const testTx = buildTransaction([
  ComputeBudgetProgram.setComputeUnitLimit({ units: 1_400_000 }),
  ...userInstructions,
  tipInstruction,
]);
testTx.sign([keypair]);

const simulation = await connection.simulateTransaction(testTx, {
  replaceRecentBlockhash: true,
  sigVerify: false,
});

// Set CU limit to actual usage + 10% margin (minimum 1000)
const units = simulation.value.unitsConsumed;
const computeUnits = units < 1000 ? 1000 : Math.ceil(units * 1.1);
```

### 2. Get dynamic priority fee

Use the `getPriorityFeeEstimate` MCP tool, or call the API directly:

```typescript
const response = await fetch(heliusRpcUrl, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    jsonrpc: "2.0",
    id: "1",
    method: "getPriorityFeeEstimate",
    params: [{
      transaction: bs58.encode(tempTx.serialize()),
      options: { recommended: true },
    }],
  }),
});

const data = await response.json();
// Add 20% buffer on top of recommended fee
const priorityFee = Math.ceil(data.result.priorityFeeEstimate * 1.2);
```

### 3. Retry with blockhash expiry check

```typescript
async function sendWithRetry(
  transaction: VersionedTransaction,
  connection: Connection,
  lastValidBlockHeight: number,
  maxRetries = 3
): Promise<string> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const currentHeight = await connection.getBlockHeight('confirmed');
    if (currentHeight > lastValidBlockHeight) {
      throw new Error('Blockhash expired — rebuild transaction with fresh blockhash');
    }

    try {
      const response = await fetch('https://sender.helius-rpc.com/fast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: Date.now().toString(),
          method: 'sendTransaction',
          params: [
            Buffer.from(transaction.serialize()).toString('base64'),
            { encoding: 'base64', skipPreflight: true, maxRetries: 0 }
          ]
        })
      });

      const result = await response.json();
      if (result.error) throw new Error(result.error.message);

      // Poll for confirmation
      return await confirmTransaction(result.result, connection);
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
  throw new Error('All retry attempts failed');
}

async function confirmTransaction(signature: string, connection: Connection): Promise<string> {
  for (let i = 0; i < 30; i++) {
    const status = await connection.getSignatureStatuses([signature]);
    if (status?.value[0]?.confirmationStatus === "confirmed") {
      return signature;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error(`Confirmation timeout: ${signature}`);
}
```

## Connection Warming

If your application has gaps longer than 1 minute between transactions, periodically ping the Sender endpoint to keep connections warm:

```typescript
// Ping every 30 seconds during idle periods
const endpoint = 'https://sender.helius-rpc.com'; // or regional HTTP endpoint

setInterval(async () => {
  try {
    await fetch(`${endpoint}/ping`);
  } catch {
    // Ignore ping failures
  }
}, 30_000);
```

Ping endpoints:
- HTTPS: `https://sender.helius-rpc.com/ping`
- Regional: `http://{region}-sender.helius-rpc.com/ping` (slc, ewr, lon, fra, ams, sg, tyo)

## Choosing a Routing Mode

| | Default Dual Routing | SWQOS-Only |
|---|---|---|
| Routes to | Validators AND Jito | SWQOS infrastructure only |
| Minimum tip | 0.0002 SOL | 0.000005 SOL |
| Best for | Maximum inclusion probability | Cost-sensitive operations |
| Endpoint | `/fast` | `/fast?swqos_only=true` |

Use default dual routing for anything time-sensitive (trading, swaps, minting). Use SWQOS-only when you want to save on tips and only want to leverage staked connections.

## Instruction Ordering

When building the transaction, instructions MUST be ordered:

1. `ComputeBudgetProgram.setComputeUnitLimit` (first)
2. `ComputeBudgetProgram.setComputeUnitPrice` (second)
3. Your application instructions (middle)
4. Jito tip transfer (last)

## Common Mistakes

- Forgetting `skipPreflight: true` — transaction will be rejected
- Forgetting the Jito tip — transaction will not be forwarded to Jito
- Hardcoding priority fees instead of using `getPriorityFeeEstimate`
- Using the default 200,000 CU limit instead of simulating actual usage
- Not implementing retry logic (relying on `maxRetries` param instead)
- Using regional HTTP endpoints in browser apps (causes CORS failures — use HTTPS)
- Including compute budget instructions in user instructions AND in the wrapper (duplicates)
