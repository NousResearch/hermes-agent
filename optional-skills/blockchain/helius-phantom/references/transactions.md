# Transaction Patterns Reference

Detailed transaction patterns for Solana with Phantom Connect SDKs and Helius infrastructure.

## The Sign → Sender Flow

For extension wallets (`"injected"` provider), use this pattern for optimal landing rates:

```
1. Build transaction with @solana/kit (pipe → compileTransaction)
2. Phantom signs (signTransaction)
3. Submit to Helius Sender (https://sender.helius-rpc.com/fast)
4. Poll for confirmation
```

**Embedded wallet limitation**: `signTransaction` is NOT supported for embedded wallets (`"google"`, `"apple"` providers). Embedded wallets must use `signAndSendTransaction`, which signs and submits atomically through Phantom's infrastructure. The `signTransaction` + Sender pattern in this file applies to extension wallets only.

**`window.phantom.solana` compatibility**: The legacy injected extension provider (`window.phantom.solana`) requires `@solana/web3.js` v1 types (`VersionedTransaction`, `PublicKey`, etc.) and does NOT work with `@solana/kit`. Always use the Phantom Connect SDK (`@phantom/react-sdk` or `@phantom/browser-sdk`), which accepts `@solana/kit` types natively.

## Dependencies

```bash
npm install @solana/kit @solana-program/system @solana-program/compute-budget @solana-program/token @solana-program/associated-token
```

## SOL Transfer

```ts
import {
  pipe,
  createTransactionMessage,
  setTransactionMessageFeePayer,
  setTransactionMessageLifetimeUsingBlockhash,
  appendTransactionMessageInstruction,
  compileTransaction,
  address,
  lamports,
} from "@solana/kit";
import { getTransferSolInstruction } from "@solana-program/system";
import {
  getSetComputeUnitLimitInstruction,
  getSetComputeUnitPriceInstruction,
} from "@solana-program/compute-budget";

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
  "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
];

async function transferSol(solana: any, recipient: string, amountSOL: number) {
  // 1. Get blockhash via backend proxy (API key stays server-side)
  // See references/frontend-security.md for proxy setup
  const bhRes = await fetch("/api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: "1",
      method: "getLatestBlockhash",
      params: [{ commitment: "confirmed" }],
    }),
  });
  const { result: bhResult } = await bhRes.json();
  const blockhash = bhResult.value;

  // 2. Get priority fee via backend proxy
  const fromAddress = await solana.getPublicKey();
  const feeRes = await fetch("/api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: "1",
      method: "getPriorityFeeEstimate",
      params: [{ accountKeys: [fromAddress], options: { priorityLevel: "High" } }],
    }),
  });
  const { result: feeResult } = await feeRes.json();
  const priorityFee = Math.ceil((feeResult?.priorityFeeEstimate || 200_000) * 1.2);

  // 3. Build transaction with proper instruction ordering
  const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];
  const payer = address(fromAddress);

  const txMessage = pipe(
    createTransactionMessage({ version: 0 }),
    (m) => setTransactionMessageFeePayer(payer, m),
    (m) => setTransactionMessageLifetimeUsingBlockhash(blockhash, m),
    // CU limit FIRST
    (m) => appendTransactionMessageInstruction(getSetComputeUnitLimitInstruction({ units: 50_000 }), m),
    // CU price SECOND
    (m) => appendTransactionMessageInstruction(getSetComputeUnitPriceInstruction({ microLamports: priorityFee }), m),
    // Your instructions in the MIDDLE
    (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
      source: payer,
      destination: address(recipient),
      amount: lamports(BigInt(Math.floor(amountSOL * 1_000_000_000))),
    }), m),
    // Jito tip LAST
    (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
      source: payer,
      destination: address(tipAccount),
      amount: lamports(200_000n), // 0.0002 SOL minimum Jito tip
    }), m),
  );

  const transaction = compileTransaction(txMessage);

  // 4. Phantom signs (does NOT send)
  const signedTx = await solana.signTransaction(transaction);

  // 5. Submit to Helius Sender — see references/helius-sender.md
  const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

  const response = await fetch("https://sender.helius-rpc.com/fast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      method: "sendTransaction",
      params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
    }),
  });

  const result = await response.json();
  if (result.error) throw new Error(result.error.message);

  // 6. Poll for confirmation
  const signature = result.result;
  await pollConfirmation(signature);

  return signature;
}
```

## SPL Token Transfer

```ts
import {
  pipe,
  createTransactionMessage,
  setTransactionMessageFeePayer,
  setTransactionMessageLifetimeUsingBlockhash,
  appendTransactionMessageInstruction,
  compileTransaction,
  address,
  lamports,
} from "@solana/kit";
import { getTransferSolInstruction } from "@solana-program/system";
import {
  getSetComputeUnitLimitInstruction,
  getSetComputeUnitPriceInstruction,
} from "@solana-program/compute-budget";
import { getTransferInstruction } from "@solana-program/token";
import {
  findAssociatedTokenPda,
  getCreateAssociatedTokenIdempotentInstruction,
} from "@solana-program/associated-token";

async function transferToken(
  solana: any,
  mint: string,
  recipient: string,
  amount: number,
  decimals: number
) {
  const fromAddress = await solana.getPublicKey();
  const payer = address(fromAddress);
  const mintAddress = address(mint);
  const recipientAddress = address(recipient);

  const [fromAta] = await findAssociatedTokenPda({ mint: mintAddress, owner: payer });
  const [toAta] = await findAssociatedTokenPda({ mint: mintAddress, owner: recipientAddress });

  const transferAmount = BigInt(Math.floor(amount * Math.pow(10, decimals)));

  // Get blockhash + priority fee via proxy (same as SOL transfer above)
  const bhRes = await fetch("/api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: "1",
      method: "getLatestBlockhash",
      params: [{ commitment: "confirmed" }],
    }),
  });
  const { result: bhResult } = await bhRes.json();

  const feeRes = await fetch("/api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: "1",
      method: "getPriorityFeeEstimate",
      params: [{ accountKeys: [fromAddress, mint], options: { priorityLevel: "High" } }],
    }),
  });
  const { result: feeResult } = await feeRes.json();
  const priorityFee = Math.ceil((feeResult?.priorityFeeEstimate || 200_000) * 1.2);

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
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
  ];
  const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];

  const txMessage = pipe(
    createTransactionMessage({ version: 0 }),
    (m) => setTransactionMessageFeePayer(payer, m),
    (m) => setTransactionMessageLifetimeUsingBlockhash(bhResult.value, m),
    (m) => appendTransactionMessageInstruction(getSetComputeUnitLimitInstruction({ units: 100_000 }), m),
    (m) => appendTransactionMessageInstruction(getSetComputeUnitPriceInstruction({ microLamports: priorityFee }), m),
    // Ensure recipient ATA exists — creates if missing, skips if it exists
    (m) => appendTransactionMessageInstruction(getCreateAssociatedTokenIdempotentInstruction({
      payer,
      owner: recipientAddress,
      mint: mintAddress,
      ata: toAta,
    }), m),
    (m) => appendTransactionMessageInstruction(getTransferInstruction({
      source: fromAta,
      destination: toAta,
      authority: payer,
      amount: transferAmount,
    }), m),
    (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
      source: payer,
      destination: address(tipAccount),
      amount: lamports(200_000n),
    }), m),
  );

  const transaction = compileTransaction(txMessage);

  // Sign with Phantom, submit to Sender
  const signedTx = await solana.signTransaction(transaction);
  const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

  const response = await fetch("https://sender.helius-rpc.com/fast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: "1",
      method: "sendTransaction",
      params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
    }),
  });

  const result = await response.json();
  if (result.error) throw new Error(result.error.message);
  return result.result;
}
```

## Signing a Pre-Built Transaction (from Swap APIs)

When an API (Jupiter, DFlow, etc.) returns a serialized transaction, you only need to sign and submit:

```ts
async function signAndSubmitApiTransaction(solana: any, serializedTx: string) {
  // Decode the base64 transaction from the API
  const txBytes = Uint8Array.from(atob(serializedTx), (c) => c.charCodeAt(0));

  // Sign with Phantom (accepts raw transaction bytes)
  const signedTx = await solana.signTransaction(txBytes);

  // Submit to Helius Sender
  const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

  const response = await fetch("https://sender.helius-rpc.com/fast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      method: "sendTransaction",
      params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
    }),
  });

  const result = await response.json();
  if (result.error) throw new Error(result.error.message);
  return result.result;
}
```

## Message Signing

Use for authentication, proof of ownership, or off-chain verification:

```ts
// Sign a message
const message = "Hello World";
const { signature } = await solana.signMessage(message);
console.log("Signature:", signature);
```

## Confirmation Polling

Always poll for confirmation after submitting via Sender:

```ts
async function pollConfirmation(signature: string): Promise<void> {
  for (let i = 0; i < 30; i++) {
    // Poll via backend proxy (API key stays server-side)
    const response = await fetch("/api/rpc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: "1",
        method: "getSignatureStatuses",
        params: [[signature]],
      }),
    });
    const { result } = await response.json();
    const status = result?.value?.[0];
    if (status?.confirmationStatus === "confirmed" || status?.confirmationStatus === "finalized") {
      if (status.err) throw new Error("Transaction failed on-chain");
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("Confirmation timeout — check explorer");
}
```

## Instruction Ordering (Required for Sender)

When building transactions for Helius Sender with Jito tips, instructions **must** be in this order:

1. `getSetComputeUnitLimitInstruction(...)` — first
2. `getSetComputeUnitPriceInstruction(...)` — second
3. Your instructions — middle
4. Jito tip transfer — last

See `references/helius-sender.md` and `references/helius-priority-fees.md` for details.

## Error Handling

```ts
try {
  const signedTx = await solana.signTransaction(transaction);
  // ... submit to Sender
} catch (error: any) {
  if (error.message?.includes("User rejected")) {
    console.log("User cancelled the transaction");
    // Not an error — don't retry
  } else if (error.message?.includes("insufficient funds")) {
    console.log("Not enough balance");
  } else {
    console.error("Transaction failed:", error);
  }
}
```

## Common Mistakes

- **Using `signAndSendTransaction` when `signTransaction` + Sender is available** — for extension wallets, `signAndSendTransaction` submits through standard RPC. Use `signTransaction` then POST to Helius Sender for better landing rates. Note: embedded wallets (`"google"`, `"apple"`) only support `signAndSendTransaction`.
- **Using `window.phantom.solana` instead of the Connect SDK** — the legacy injected provider requires `@solana/web3.js` v1 types and does not work with `@solana/kit`. Use `@phantom/react-sdk` or `@phantom/browser-sdk`.
- **Missing priority fees** — transactions without priority fees are deprioritized. Use `getPriorityFeeEstimate` via your backend proxy.
- **Missing Jito tip** — Helius Sender uses Jito for dual routing. Include a minimum 0.0002 SOL tip to benefit from Jito block building.
- **Wrong instruction ordering** — CU limit must be first, CU price second, Jito tip last. Incorrect ordering causes Sender to reject the transaction.
- **Using legacy `Transaction` class** — always use `@solana/kit`'s `createTransactionMessage({ version: 0 })` for v0 transaction support and forward compatibility.
- **Hardcoding priority fees** — network conditions change. Always query `getPriorityFeeEstimate` for current fee levels.
- **Using public RPC for blockhash** — use your backend proxy to get the blockhash via Helius RPC (faster, more reliable). See `references/frontend-security.md`.
- **Not polling for confirmation** — Sender returns a signature immediately, but the transaction may not be confirmed yet. Always poll `getSignatureStatuses`.
