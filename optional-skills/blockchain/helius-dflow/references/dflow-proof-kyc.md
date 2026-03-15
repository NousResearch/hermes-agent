# DFlow Proof KYC — Identity Verification

## What This Covers

Proof KYC links verified real-world identities to Solana wallets. Required for prediction market outcome token trading. Also useful for any gated feature needing verified wallet ownership.

Full docs: `https://pond.dflow.net/learn/proof`

## When KYC Is Required

**Required for:**
- Buying outcome tokens (prediction market trades)
- Selling outcome tokens (prediction market trades)

**NOT required for:**
- Browsing markets, fetching events/orderbooks/metadata
- Viewing market details
- Spot crypto swaps (see `references/dflow-spot-trading.md`)
- Any non-prediction-market operation

Gate verification only at trade time — not for browsing or API access.

## Key Facts

* **KYC provider**: Stripe Identity under the hood.
* **Cost**: Free to use.
* **Geoblocking still required**: KYC verifies identity but does not replace jurisdictional restrictions. Builders must enforce geoblocking independently.

## Verify API

Check if a wallet is verified:

```bash
curl "https://proof.dflow.net/verify/{address}"
# Response: { "verified": true } or { "verified": false }
```

For prediction markets: call before allowing buys/sells of outcome tokens. For other use cases: call whenever you need to gate a feature by verification status.

## Deep Link (Send Unverified Users to Proof)

When a user is not verified, redirect them to the Proof verification flow.

### Required Parameters

| Param | Required | Description |
|---|---|---|
| `wallet` | Yes | Solana wallet address |
| `signature` | Yes | Base58-encoded signature of the message |
| `timestamp` | Yes | Unix timestamp in milliseconds |
| `redirect_uri` | Yes | URL to return to after verification |
| `projectId` | No | Project identifier for tracking |

### Message Format

The user signs this message with their wallet:

```
Proof KYC verification: {timestamp}
```

### Verification Flow

1. User connects wallet.
2. User signs `Proof KYC verification: {Date.now()}` with their wallet.
3. Build the deep link URL:

```
https://dflow.net/proof?wallet={wallet}&signature={signature}&timestamp={timestamp}&redirect_uri={redirect_uri}
```

4. Open in new tab or redirect.
5. User completes KYC via Stripe Identity.
6. User is redirected to `redirect_uri`.
7. Call the verify API on return to confirm status. If the user cancelled, `verified` will still be false.

### Implementation Pattern

```typescript
async function initiateKYC(wallet: PublicKey, signMessage: (msg: Uint8Array) => Promise<Uint8Array>) {
  const timestamp = Date.now();
  const message = `Proof KYC verification: ${timestamp}`;
  const messageBytes = new TextEncoder().encode(message);

  // User signs the message
  const signature = await signMessage(messageBytes);
  const signatureBase58 = bs58.encode(signature);

  // Build deep link
  const params = new URLSearchParams({
    wallet: wallet.toBase58(),
    signature: signatureBase58,
    timestamp: timestamp.toString(),
    redirect_uri: window.location.href, // or your desired return URL
  });

  // Open Proof KYC page
  window.open(`https://dflow.net/proof?${params.toString()}`, '_blank');
}

async function checkKYCStatus(walletAddress: string): Promise<boolean> {
  const response = await fetch(`https://proof.dflow.net/verify/${walletAddress}`);
  const { verified } = await response.json();
  return verified;
}
```

### Handling the Redirect Return

When the user returns to your `redirect_uri` after completing (or cancelling) KYC, you must check their status — there is no callback or webhook. The redirect itself does not indicate success.

```typescript
// On page load (or when redirect_uri is hit), check verification
async function handleKYCReturn(walletAddress: string) {
  const verified = await checkKYCStatus(walletAddress);

  if (verified) {
    // User is verified — allow prediction market trading
    enableTrading();
  } else {
    // User cancelled or verification failed — offer to retry
    showRetryPrompt();
  }
}
```

For apps that open Proof in a new tab (rather than redirect), poll the verify API after the user signals they've completed the flow:

```typescript
async function pollKYCStatus(walletAddress: string, maxAttempts = 30): Promise<boolean> {
  for (let i = 0; i < maxAttempts; i++) {
    const verified = await checkKYCStatus(walletAddress);
    if (verified) return true;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  return false;
}
```

## Common Mistakes

- Requiring KYC for spot swaps (only needed for prediction markets)
- Gating market browsing behind KYC (only gate at trade time)
- Not checking KYC status on return from the Proof page (user may have cancelled)
- Assuming KYC replaces geoblocking (it doesn't — builders must enforce jurisdictional restrictions)
- Not handling the case where a user returns from Proof but is still unverified

## Resources

* Proof Docs: `pond.dflow.net/learn/proof`
* Proof API: `pond.dflow.net/build/proof-api/introduction`
* Proof Partner Integration: `pond.dflow.net/build/proof/partner-integration`
