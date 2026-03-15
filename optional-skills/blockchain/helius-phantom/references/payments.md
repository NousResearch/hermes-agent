# Crypto Payments

Accept cryptocurrency payments using Phantom Connect for signing and Helius infrastructure for submission and verification.

## Architecture

```
1. User connects wallet (Phantom Connect SDK)
2. Backend creates payment transaction (Helius RPC, API key server-side)
3. Frontend receives serialized transaction
4. Phantom signs (signTransaction)
5. Submit to Helius Sender
6. Backend verifies on-chain (Helius Enhanced Transactions API)
7. Fulfill order
```

## Simple SOL Payment

```tsx
import { useSolana, useAccounts } from "@phantom/react-sdk";
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
import { useState } from "react";

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

function PayButton({ recipient, amountSol }: { recipient: string; amountSol: number }) {
  const { solana } = useSolana();
  const { isConnected } = useAccounts();
  const [status, setStatus] = useState<"idle" | "paying" | "success" | "error">("idle");

  async function handlePay() {
    setStatus("paying");
    try {
      const wallet = await solana.getPublicKey();
      const payer = address(wallet);

      // Get blockhash + priority fee via backend proxy
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

      const feeRes = await fetch("/api/rpc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0", id: "1",
          method: "getPriorityFeeEstimate",
          params: [{ accountKeys: [wallet], options: { priorityLevel: "High" } }],
        }),
      });
      const { result: feeResult } = await feeRes.json();
      const priorityFee = Math.ceil((feeResult?.priorityFeeEstimate || 200_000) * 1.2);

      const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];

      const txMessage = pipe(
        createTransactionMessage({ version: 0 }),
        (m) => setTransactionMessageFeePayer(payer, m),
        (m) => setTransactionMessageLifetimeUsingBlockhash(bhResult.value, m),
        (m) => appendTransactionMessageInstruction(getSetComputeUnitLimitInstruction({ units: 50_000 }), m),
        (m) => appendTransactionMessageInstruction(getSetComputeUnitPriceInstruction({ microLamports: priorityFee }), m),
        (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
          source: payer,
          destination: address(recipient),
          amount: lamports(BigInt(Math.floor(amountSol * 1_000_000_000))),
        }), m),
        (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
          source: payer,
          destination: address(tipAccount),
          amount: lamports(200_000n), // Jito tip
        }), m),
      );

      const tx = compileTransaction(txMessage);

      // Sign with Phantom, submit to Helius Sender
      const signedTx = await solana.signTransaction(tx);
      const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

      const senderRes = await fetch("https://sender.helius-rpc.com/fast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: "1",
          method: "sendTransaction",
          params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
        }),
      });
      const senderResult = await senderRes.json();
      if (senderResult.error) throw new Error(senderResult.error.message);

      setStatus("success");
    } catch {
      setStatus("error");
    }
  }

  return (
    <button onClick={handlePay} disabled={!isConnected || status === "paying"}>
      {status === "paying" ? "Processing..." : `Pay ${amountSol} SOL`}
    </button>
  );
}
```

## SPL Token Payment (USDC)

```tsx
import {
  pipe,
  createTransactionMessage,
  setTransactionMessageFeePayer,
  setTransactionMessageLifetimeUsingBlockhash,
  appendTransactionMessageInstruction,
  compileTransaction,
  address,
} from "@solana/kit";
import { getTransferInstruction } from "@solana-program/token";
import {
  findAssociatedTokenPda,
  getCreateAssociatedTokenIdempotentInstruction,
} from "@solana-program/associated-token";

const USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

async function payWithUSDC(solana: any, recipient: string, amount: number) {
  const wallet = await solana.getPublicKey();
  const payer = address(wallet);
  const mintAddress = address(USDC_MINT);
  const recipientAddress = address(recipient);

  const [fromAta] = await findAssociatedTokenPda({ mint: mintAddress, owner: payer });
  const [toAta] = await findAssociatedTokenPda({ mint: mintAddress, owner: recipientAddress });

  // Get blockhash via proxy
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

  const txMessage = pipe(
    createTransactionMessage({ version: 0 }),
    (m) => setTransactionMessageFeePayer(payer, m),
    (m) => setTransactionMessageLifetimeUsingBlockhash(bhResult.value, m),
    // Ensure recipient ATA exists — creates if missing, skips if it exists
    (m) => appendTransactionMessageInstruction(getCreateAssociatedTokenIdempotentInstruction({
      payer,
      owner: recipientAddress,
      mint: mintAddress,
      ata: toAta,
    }), m),
    // Transfer (USDC has 6 decimals)
    (m) => appendTransactionMessageInstruction(getTransferInstruction({
      source: fromAta,
      destination: toAta,
      authority: payer,
      amount: BigInt(Math.floor(amount * 1e6)),
    }), m),
  );

  // Sign with Phantom, submit to Sender
  const tx = compileTransaction(txMessage);
  const signedTx = await solana.signTransaction(tx);
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

## Checkout with Backend Verification

### Client

```tsx
async function checkout(orderId: string, solana: any) {
  // 1. Create payment on backend
  const { paymentId, transaction } = await fetch("/api/payments/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ orderId }),
  }).then(r => r.json());

  // 2. Decode, sign with Phantom, submit to Sender
  const txBytes = Uint8Array.from(atob(transaction), (c) => c.charCodeAt(0));
  const signedTx = await solana.signTransaction(txBytes);

  const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));
  const senderRes = await fetch("https://sender.helius-rpc.com/fast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      method: "sendTransaction",
      params: [base64Tx, { encoding: "base64", skipPreflight: true, maxRetries: 0 }],
    }),
  });
  const senderResult = await senderRes.json();
  if (senderResult.error) throw new Error(senderResult.error.message);
  const txHash = senderResult.result;

  // 3. Confirm with backend (backend verifies on-chain via Helius)
  const { success } = await fetch("/api/payments/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paymentId, txHash }),
  }).then(r => r.json());

  return success;
}
```

### Server

```ts
// app/api/payments/create/route.ts
const HELIUS_API_KEY = process.env.HELIUS_API_KEY!;

export async function POST(req: Request) {
  const { orderId } = await req.json();

  // Get order, calculate amount
  const order = await db.orders.findUnique({ where: { id: orderId } });
  const solAmount = order.total / await getSolPrice();

  // Create payment record
  const payment = await db.payments.create({
    data: { orderId, solAmount, status: "pending" }
  });

  // Build transaction using Helius RPC (API key server-side)
  // ... build and serialize transaction ...

  return Response.json({ paymentId: payment.id, transaction: "..." });
}

// app/api/payments/confirm/route.ts
export async function POST(req: Request) {
  const { paymentId, txHash } = await req.json();

  // Verify transaction on-chain using Helius Enhanced Transactions API
  // See references/helius-enhanced-transactions.md
  const txRes = await fetch(`https://api.helius.xyz/v0/transactions?api-key=${HELIUS_API_KEY}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transactions: [txHash] }),
  });
  const [parsed] = await txRes.json();

  if (!parsed || parsed.transactionError) {
    return Response.json({ success: false });
  }

  // Verify amount and recipient match expected values
  // Update payment status
  // Fulfill order

  return Response.json({ success: true });
}
```

## Price Display with Live Rates

```tsx
import { useState, useEffect } from "react";

function PriceDisplay({ usdAmount }: { usdAmount: number }) {
  const [solPrice, setSolPrice] = useState(0);

  useEffect(() => {
    async function fetchPrice() {
      const res = await fetch("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd");
      const data = await res.json();
      setSolPrice(data.solana.usd);
    }
    fetchPrice();
    const interval = setInterval(fetchPrice, 60000);
    return () => clearInterval(interval);
  }, []);

  const solAmount = solPrice ? (usdAmount / solPrice).toFixed(4) : "...";

  return (
    <div>
      <p>${usdAmount} USD</p>
      <p>{solAmount} SOL</p>
    </div>
  );
}
```

## Best Practices

1. **Always verify on-chain** — don't trust client-side confirmation alone. Use Helius Enhanced Transactions API to verify payment details on the server.
2. **Use unique payment IDs** — track each payment to prevent double-fulfillment.
3. **Handle price volatility** — lock prices or use stablecoins (USDC) for predictable amounts.
4. **Set expiration** — payment requests should expire (blockhash expiry is ~60 seconds; create fresh transactions for each attempt).
5. **Wait for confirmations** — use `confirmed` commitment level before fulfilling orders.
6. **Link to explorer** — show users `https://orbmarkets.io/tx/{signature}` for transparency.

## Common Mistakes

- **Using `signAndSendTransaction` when `signTransaction` + Sender is available** — for extension wallets, use `signTransaction` + Helius Sender for better landing rates. Note: embedded wallets (`"google"`, `"apple"`) only support `signAndSendTransaction`. See `references/transactions.md`.
- **Not verifying on the server** — the client can lie about transaction success. Always verify on-chain using Helius Enhanced Transactions API.
- **Exposing Helius API key in payment flow** — build payment transactions on the server, verify on the server. Only signing happens client-side.
- **Not handling blockhash expiry** — if the user takes too long to sign, the transaction will fail. Build a fresh transaction on each attempt.
- **Trusting client-reported amounts** — always compute the expected payment amount on the server and verify it matches the on-chain transaction.
