# Token-Gated Access

Implement token-gated features that require users to hold specific tokens, using Phantom Connect for wallet connection and Helius DAS for on-chain verification.

## Architecture

```
1. User connects wallet (Phantom Connect SDK)
2. App gets wallet address
3. Query Helius DAS for token/NFT ownership (via backend proxy)
4. If balance meets criteria → grant access
5. Optional: Sign message to prove ownership (recommended for security)
```

## Client-Side Gating (Simple)

Best for low-stakes content and UI personalization. Uses Helius DAS via backend proxy to check token ownership.

```tsx
import { useAccounts } from "@phantom/react-sdk";
import { useState, useEffect } from "react";

const TOKEN_MINT = "YOUR_TOKEN_MINT_ADDRESS";
const REQUIRED_AMOUNT = 1;

function TokenGatedContent() {
  const { addresses, isConnected } = useAccounts();
  const [hasAccess, setHasAccess] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isConnected) { setLoading(false); return; }
    checkBalance();
  }, [isConnected, addresses]);

  async function checkBalance() {
    const wallet = addresses?.find(a => a.addressType === "solana")?.address;
    if (!wallet) return;

    try {
      // Use Helius DAS via backend proxy — see references/frontend-security.md
      const res = await fetch("/api/rpc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: "1",
          method: "searchAssets",
          params: {
            ownerAddress: wallet,
            tokenType: "fungible",
            page: 1,
            limit: 1000,
          },
        }),
      });
      const data = await res.json();
      const items = data.result?.items || [];

      // Check if user holds the required token
      const tokenAsset = items.find((item: any) => item.id === TOKEN_MINT);
      const balance = tokenAsset?.token_info?.balance || 0;
      const decimals = tokenAsset?.token_info?.decimals || 0;
      const amount = balance / Math.pow(10, decimals);

      setHasAccess(amount >= REQUIRED_AMOUNT);
    } catch {
      setHasAccess(false);
    } finally {
      setLoading(false);
    }
  }

  if (!isConnected) return <ConnectPrompt />;
  if (loading) return <Loading />;
  if (!hasAccess) return <AccessDenied />;
  return <ProtectedContent />;
}
```

## Server-Side Verification (Secure)

Best for valuable content and actual access control. Combines Phantom message signing with Helius DAS verification on the server.

### Client: Sign Message

```tsx
import { useSolana } from "@phantom/react-sdk";

async function verifyAccess(solana: any) {
  const address = await solana.getPublicKey();
  const timestamp = Date.now();
  const message = `Verify ownership\nAddress: ${address}\nTimestamp: ${timestamp}`;

  const { signature } = await solana.signMessage(message);

  const res = await fetch("/api/verify-access", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ address, signature, message, timestamp }),
  });

  return await res.json();
}
```

### Server: Verify Signature + Check with Helius DAS

```ts
// app/api/verify-access/route.ts
import nacl from "tweetnacl";
import bs58 from "bs58";
import jwt from "jsonwebtoken";

const HELIUS_API_KEY = process.env.HELIUS_API_KEY!;
const JWT_SECRET = process.env.JWT_SECRET!;
const TOKEN_MINT = "YOUR_TOKEN_MINT_ADDRESS";
const REQUIRED_BALANCE = 1;

export async function POST(req: Request) {
  const { address, signature, message, timestamp } = await req.json();

  // 1. Check timestamp (5 min window)
  if (Date.now() - timestamp > 5 * 60 * 1000) {
    return Response.json({ error: "Expired" }, { status: 400 });
  }

  // 2. Verify signature
  const isValid = nacl.sign.detached.verify(
    new TextEncoder().encode(message),
    bs58.decode(signature),
    bs58.decode(address)
  );
  if (!isValid) {
    return Response.json({ error: "Invalid signature" }, { status: 401 });
  }

  // 3. Check token balance using Helius DAS (API key server-side)
  try {
    const dasRes = await fetch(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: "1",
        method: "searchAssets",
        params: {
          ownerAddress: address,
          tokenType: "fungible",
          page: 1,
          limit: 1000,
        },
      }),
    });
    const dasData = await dasRes.json();
    const items = dasData.result?.items || [];

    const tokenAsset = items.find((item: any) => item.id === TOKEN_MINT);
    const balance = tokenAsset?.token_info?.balance || 0;
    const decimals = tokenAsset?.token_info?.decimals || 0;
    const amount = balance / Math.pow(10, decimals);

    if (amount < REQUIRED_BALANCE) {
      return Response.json({ hasAccess: false, balance: amount });
    }

    const accessToken = jwt.sign({ address, balance: amount }, JWT_SECRET, { expiresIn: "24h" });
    return Response.json({ hasAccess: true, accessToken });
  } catch {
    return Response.json({ hasAccess: false, balance: 0 });
  }
}
```

## NFT Collection Gating

Use Helius DAS `searchAssets` to check if a wallet owns an NFT from a specific collection — no Metaplex SDK needed:

```ts
// Server-side: check NFT ownership via Helius DAS
// See references/helius-das.md for full API details
async function checkNFTOwnership(wallet: string, collectionAddress: string): Promise<boolean> {
  const response = await fetch(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      method: "getAssetsByOwner",
      params: {
        ownerAddress: wallet,
        page: 1,
        limit: 1000,
        displayOptions: { showCollectionMetadata: true },
      },
    }),
  });

  const data = await response.json();
  const items = data.result?.items || [];

  return items.some((item: any) =>
    item.grouping?.some(
      (g: any) => g.group_key === "collection" && g.group_value === collectionAddress
    )
  );
}
```

## SOL Balance Gating

Use Helius RPC via backend proxy:

```ts
async function checkSolBalance(wallet: string, requiredSol: number): Promise<boolean> {
  const response = await fetch(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      method: "getBalance",
      params: [wallet],
    }),
  });
  const data = await response.json();
  const solBalance = (data.result?.value || 0) / 1e9;
  return solBalance >= requiredSol;
}
```

## Security Best Practices

1. **Always verify server-side** for valuable content — client-side checks are trivially bypassed
2. **Use message signing** to prove wallet ownership — prevents address spoofing
3. **Include timestamps** to prevent replay attacks — reject signatures older than 5 minutes
4. **Cache verification** with short TTLs — re-verify periodically, not on every request
5. **Re-verify on sensitive actions** — don't rely on cached access for high-value operations

## Common Mistakes

- **Client-side only gating for valuable content** — anyone can bypass frontend checks. Always verify on the server for anything worth protecting.
- **Not verifying message signature** — without signature verification, anyone can claim to own any wallet address.
- **Using Metaplex SDK for NFT checks** — Helius DAS is simpler and more efficient. One `getAssetsByOwner` call replaces multiple Metaplex SDK calls.
- **Exposing Helius API key in token check** — client-side DAS calls expose your key. Use a backend proxy for the token balance check.
- **Not including a timestamp in the signed message** — without timestamps, signed messages can be replayed indefinitely.
