# SIWA (Sign-In With Agent) Authentication

## Overview

SIWA is Helixa's authentication mechanism for agent-initiated API calls. The agent signs a message with its private key to prove wallet ownership.

## Message Format

```
Sign-In With Agent: api.helixa.xyz wants you to sign in with your wallet {address} at {timestamp}
```

- `{address}` — agent's Ethereum wallet address (checksummed)
- `{timestamp}` — Unix timestamp in seconds (must be within 5 minutes of server time)

## Auth Header

```
Authorization: Bearer {address}:{timestamp}:{signature}
```

## Implementation (JavaScript/Node.js)

### Using ethers.js

```javascript
const { ethers } = require('ethers');

async function getSiwaAuth(privateKey) {
  const wallet = new ethers.Wallet(privateKey);
  const address = wallet.address;
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const message = `Sign-In With Agent: api.helixa.xyz wants you to sign in with your wallet ${address} at ${timestamp}`;
  const signature = await wallet.signMessage(message);
  return `Bearer ${address}:${timestamp}:${signature}`;
}

// Usage
const auth = await getSiwaAuth(process.env.AGENT_PRIVATE_KEY);
const res = await fetch('https://api.helixa.xyz/api/v2/mint', {
  method: 'POST',
  headers: {
    'Authorization': auth,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ name: 'MyAgent', framework: 'openclaw' })
});
```

### Using viem

```javascript
import { createWalletClient, http } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';
import { base } from 'viem/chains';

async function getSiwaAuth(privateKey) {
  const account = privateKeyToAccount(privateKey);
  const address = account.address;
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const message = `Sign-In With Agent: api.helixa.xyz wants you to sign in with your wallet ${address} at ${timestamp}`;
  const signature = await account.signMessage({ message });
  return `Bearer ${address}:${timestamp}:${signature}`;
}
```

### Using cast (Foundry)

```bash
ADDRESS=$(cast wallet address --private-key $PRIVATE_KEY)
TIMESTAMP=$(date +%s)
MESSAGE="Sign-In With Agent: api.helixa.xyz wants you to sign in with your wallet ${ADDRESS} at ${TIMESTAMP}"
SIGNATURE=$(cast wallet sign --private-key $PRIVATE_KEY "$MESSAGE")
AUTH="Bearer ${ADDRESS}:${TIMESTAMP}:${SIGNATURE}"

curl -X POST https://api.helixa.xyz/api/v2/mint \
  -H "Authorization: $AUTH" \
  -H "Content-Type: application/json" \
  -d '{"name":"MyAgent","framework":"openclaw"}'
```

## With x402 Payments

Endpoints that cost money (mint, update, cred-report) return HTTP 402 with x402 payment instructions. Use the x402 SDK for automatic handling:

```javascript
const { wrapFetchWithPayment, x402Client } = require('@x402/fetch');
const { ExactEvmScheme } = require('@x402/evm/exact/client');
const { toClientEvmSigner } = require('@x402/evm');

// Set up x402 payment client
const signer = toClientEvmSigner(walletClient);
signer.address = walletClient.account.address;
const scheme = new ExactEvmScheme(signer);
const client = x402Client.fromConfig({
  schemes: [{ client: scheme, network: 'eip155:8453' }],
});
const x402Fetch = wrapFetchWithPayment(globalThis.fetch, client);

// Now use x402Fetch — it handles 402 responses automatically
const auth = await getSiwaAuth(process.env.AGENT_PRIVATE_KEY);
const res = await x402Fetch('https://api.helixa.xyz/api/v2/mint', {
  method: 'POST',
  headers: { 'Authorization': auth, 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'MyAgent', framework: 'openclaw' }),
});
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 401 `invalid signature` | Wrong private key or malformed message | Verify message format exactly matches spec |
| 401 `timestamp expired` | Timestamp >5 min from server time | Generate fresh timestamp |
| 401 `address mismatch` | Address in header doesn't match signer | Use same wallet for signing and header |

## Security Notes

- Never log, print, or expose private keys
- Store keys only in environment variables
- SIWA timestamps expire after ~5 minutes — always generate fresh
- The signed message is domain-bound to `api.helixa.xyz`
