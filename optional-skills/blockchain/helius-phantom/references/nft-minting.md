# NFT Minting

Build NFT mint pages and drop experiences with Phantom Connect and Helius infrastructure.

## Architecture

```
1. User connects wallet (Phantom Connect SDK)
2. User clicks "Mint" → request sent to backend
3. Backend builds mint transaction (using Helius RPC, API key server-side)
4. Frontend receives serialized transaction
5. Phantom signs (signTransaction)
6. Submit to Helius Sender
7. Verify via Helius DAS (optional — confirm NFT was minted)
```

## Mint Page Pattern

```tsx
import { PhantomProvider, useModal, useAccounts, useSolana, darkTheme } from "@phantom/react-sdk";
import { AddressType } from "@phantom/browser-sdk";
import { useState } from "react";

function MintPage() {
  const { isConnected, addresses } = useAccounts();
  const { open } = useModal();
  const { solana } = useSolana();
  const [quantity, setQuantity] = useState(1);
  const [status, setStatus] = useState<"idle" | "minting" | "success" | "error">("idle");
  const [txSignature, setTxSignature] = useState<string | null>(null);

  const PRICE = 0.5; // SOL
  const MAX_PER_WALLET = 5;

  async function handleMint() {
    if (!isConnected) { open(); return; }

    setStatus("minting");
    try {
      const wallet = addresses?.find(a => a.addressType === "solana")?.address;

      // 1. Get mint transaction from backend
      const res = await fetch("/api/mint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wallet, quantity }),
      });
      const { transaction } = await res.json();

      // 2. Decode and sign with Phantom (accepts raw transaction bytes)
      const txBytes = Uint8Array.from(atob(transaction), (c) => c.charCodeAt(0));
      const signedTx = await solana.signTransaction(txBytes);

      // 3. Submit to Helius Sender — see references/helius-sender.md
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

      setTxSignature(senderResult.result);
      setStatus("success");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div>
      <h1>Mint NFT</h1>
      <p>Price: {PRICE} SOL each</p>

      <div>
        <button onClick={() => setQuantity(q => Math.max(1, q - 1))}>-</button>
        <span>{quantity}</span>
        <button onClick={() => setQuantity(q => Math.min(MAX_PER_WALLET, q + 1))}>+</button>
      </div>

      <p>Total: {(PRICE * quantity).toFixed(2)} SOL</p>

      <button onClick={handleMint} disabled={status === "minting"}>
        {!isConnected ? "Connect Wallet" : status === "minting" ? "Minting..." : "Mint"}
      </button>

      {status === "success" && txSignature && (
        <p>
          Minted! <a href={`https://orbmarkets.io/tx/${txSignature}`} target="_blank" rel="noopener">
            View on Orb
          </a>
        </p>
      )}
    </div>
  );
}
```

## Backend: Build Mint Transaction

The backend uses Helius RPC (API key stays server-side) to build the mint transaction:

```ts
// app/api/mint/route.ts
import { createUmi } from "@metaplex-foundation/umi-bundle-defaults";
import { createNft, mplCore } from "@metaplex-foundation/mpl-core";
import { generateSigner, publicKey, keypairIdentity } from "@metaplex-foundation/umi";

const HELIUS_API_KEY = process.env.HELIUS_API_KEY!;

export async function POST(req: Request) {
  const { wallet, quantity } = await req.json();

  // Use Helius RPC (server-side, API key safe here)
  const umi = createUmi(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`)
    .use(mplCore())
    .use(keypairIdentity(authorityKeypair));

  const asset = generateSigner(umi);

  const tx = createNft(umi, {
    asset,
    name: "NFT #1",
    uri: "https://arweave.net/metadata.json",
    owner: publicKey(wallet),
  });

  const built = await tx.build(umi);
  const serialized = Buffer.from(umi.transactions.serialize(built)).toString("base64");

  return Response.json({ transaction: serialized });
}
```

## Allowlist Mint

```tsx
async function allowlistMint(solana: any, wallet: string, qty: number) {
  const { proof, transaction } = await fetch("/api/allowlist-mint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ wallet, quantity: qty }),
  }).then(r => r.json());

  if (!proof) throw new Error("Not on allowlist");

  // Sign with Phantom (accepts raw transaction bytes), submit to Sender
  const txBytes = Uint8Array.from(atob(transaction), (c) => c.charCodeAt(0));
  const signedTx = await solana.signTransaction(txBytes);

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

## Metadata Format (Metaplex)

```json
{
  "name": "Collection #1",
  "symbol": "COLL",
  "description": "Description",
  "image": "https://arweave.net/image.png",
  "attributes": [
    { "trait_type": "Background", "value": "Blue" }
  ],
  "properties": {
    "files": [{ "uri": "https://arweave.net/image.png", "type": "image/png" }]
  }
}
```

## Compressed NFTs (cNFTs)

For large collections, use compressed NFTs to reduce costs. Backend builds the transaction, frontend signs and submits via Sender. After minting, verify with Helius DAS — `getAsset` works with both regular and compressed NFTs.

```ts
// Backend: create tree + mint cNFT
import { createTree, mintV1 } from "@metaplex-foundation/mpl-bubblegum";

// Create merkle tree (one-time setup)
const tree = generateSigner(umi);
await createTree(umi, {
  merkleTree: tree,
  maxDepth: 14,     // Up to 16,384 NFTs
  maxBufferSize: 64,
}).sendAndConfirm(umi);

// Mint compressed NFT
await mintV1(umi, {
  leafOwner: publicKey(wallet),
  merkleTree: tree.publicKey,
  metadata: {
    name: "cNFT #1",
    uri: "https://arweave.net/metadata.json",
    sellerFeeBasisPoints: 500, // 5%
    collection: { key: collectionMint, verified: false },
    creators: [{ address: umi.identity.publicKey, verified: true, share: 100 }],
  },
}).sendAndConfirm(umi);
```

After minting, verify the cNFT with Helius DAS — see `references/helius-das.md`:
```ts
// Verify cNFT was minted via Helius DAS (server-side)
const dasRes = await fetch(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    jsonrpc: "2.0", id: "1",
    method: "getAsset",
    params: { id: mintAddress },
  }),
});
```

## Best Practices

1. **Generate transactions server-side** — don't expose mint authority keys in frontend code
2. **Validate wallet limits** — check mints per wallet server-side, not just in the UI
3. **Show clear pricing** — display total including estimated fees
4. **Handle all states** — loading, success, error, sold out
5. **Link to explorer** — let users verify with `https://orbmarkets.io/tx/{signature}`
6. **Use Helius RPC on the backend** — faster and more reliable than public RPC

## Common Mistakes

- **Exposing mint authority private key in frontend** — always build mint transactions on the server. The client only signs (as the payer).
- **Using `signAndSendTransaction`** — use `signTransaction` + Helius Sender for better landing rates.
- **Not validating server-side** — client-side quantity limits are trivially bypassed. Always validate on the server.
- **Using public RPC for mint transactions** — use Helius RPC (server-side) for reliability and speed during high-traffic mints.
