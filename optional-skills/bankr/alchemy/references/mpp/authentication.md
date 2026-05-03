# Authentication

Every request to the gateway must include a SIWE auth token. The token proves wallet ownership without transmitting the private key.

## Token Format

```
Authorization: SIWE <base64(siwe_message)>.<signature>
```

### Using `x-token` (recommended for SDK usage)

Since MPP uses the `Authorization` header for both auth and payment, the mppx SDK will overwrite `Authorization` with the payment credential on retry. To avoid losing the SIWE token, send it via the `x-token` header instead:

```
x-token: SIWE <base64(siwe_message)>.<signature>
```

The gateway checks `x-token` first, then falls back to `Authorization` for auth extraction. See [reference](reference.md) for full details on the header conflict resolution.

## CLI: Generate an Auth Token

For ad-hoc requests and curl workflows, use a Node.js script with `viem` to generate a SIWE token. **Important:** The domain must be `mpp.alchemy.com` to target the MPP gateway:

```bash
node -e "
  const { createWalletClient, http } = require('viem');
  const { privateKeyToAccount } = require('viem/accounts');
  const { base } = require('viem/chains');
  const { createSiweMessage, generateSiweNonce } = require('viem/siwe');
  const fs = require('fs');
  const pk = fs.readFileSync('./wallet-key.txt', 'utf8').trim();
  const account = privateKeyToAccount(pk);
  const message = createSiweMessage({
    address: account.address, chainId: base.id,
    domain: 'mpp.alchemy.com', nonce: generateSiweNonce(),
    uri: 'https://mpp.alchemy.com', version: '1',
    statement: 'Sign in to Alchemy Gateway',
    expirationTime: new Date(Date.now() + 3600000),
  });
  const client = createWalletClient({ account, chain: base, transport: http() });
  client.signMessage({ message }).then(sig => {
    process.stdout.write(Buffer.from(message).toString('base64') + '.' + sig);
  });
" > siwe-token.txt
```

> **Security:** Always read the private key from a file rather than passing it as a CLI argument to avoid exposing it in shell history and process listings.

The script prints the encoded token to stdout. Pipe it to a file to avoid terminal exposure:

```bash
TOKEN=$(cat siwe-token.txt)

# The chain URL is independent of wallet type — you can query any chain
curl -s -X POST "https://mpp.alchemy.com/eth-mainnet/v2" \
  -H "Authorization: SIWE $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'
```

## Library: Generate a Token in Code

For applications, use `viem` to generate a SIWE token. Read the private key from an environment variable — never hardcode it. **Important:** Use `domain: "mpp.alchemy.com"` to target the MPP gateway:

```bash
npm install viem
```

```typescript
import { createWalletClient, http } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { createSiweMessage, generateSiweNonce } from "viem/siwe";

const privateKey = process.env.PRIVATE_KEY as `0x${string}`;
const account = privateKeyToAccount(privateKey);

const message = createSiweMessage({
  address: account.address,
  chainId: base.id,
  domain: "mpp.alchemy.com",
  nonce: generateSiweNonce(),
  uri: "https://mpp.alchemy.com",
  version: "1",
  statement: "Sign in to Alchemy Gateway",
  expirationTime: new Date(Date.now() + 60 * 60 * 1000), // 1 hour
});

const client = createWalletClient({
  account,
  chain: base,
  transport: http(),
});

const signature = await client.signMessage({ message });
const siweToken = `${btoa(message)}.${signature}`;

// Use in requests
const headers = { Authorization: `SIWE ${siweToken}` };
```

## SIWE Message Fields

The generated token contains a SIWE message with these fields:

| Field | Value | Notes |
|-------|-------|-------|
| `domain` | `mpp.alchemy.com` | Must match the gateway's configured domain |
| `address` | Your wallet address | Checksummed Ethereum address |
| `statement` | `Sign in to Alchemy Gateway` | Human-readable intent |
| `uri` | `https://mpp.alchemy.com` | Must use `https://` + domain |
| `version` | `1` | SIWE spec version |
| `chainId` | `8453` | Base Mainnet chain ID |
| `nonce` | Random alphanumeric string | At least 8 characters |
| `expirationTime` | ISO 8601 timestamp | Default: 1 hour from now |

## Auth Error Codes

When authentication fails, the gateway returns HTTP 401 with a JSON body:

```json
{
  "error": "Unauthorized",
  "message": "<description>",
  "code": "<error_code>"
}
```

| Code | Cause | How to fix |
|------|-------|------------|
| `MISSING_AUTH` | No `Authorization` header provided | Add `Authorization: SIWE <token>` header |
| `INVALID_AUTH_FORMAT` | Token is not in `base64.signature` format or Base64 decoding failed | Ensure the token is `base64(message).signature` with exactly one `.` separator |
| `INVALID_SIWE` | SIWE message could not be parsed | Regenerate the token using the viem-based signing script |
| `INVALID_SIGNATURE` | Signature does not match the message signer | Ensure the correct private key is being used |
| `INVALID_DOMAIN` | Message domain does not match `mpp.alchemy.com` | Ensure `domain: "mpp.alchemy.com"` is passed in the `createSiweMessage` call |
| `MESSAGE_EXPIRED` | Message `expirationTime` has passed or `notBefore` is in the future | Generate a new token — the current one has expired |

## Handling Auth Errors

If you receive a 401 response, parse the `code` field and take the appropriate action:

- **Regenerable errors** (`MESSAGE_EXPIRED`): Generate a fresh token and retry the request.
- **Configuration errors** (`INVALID_DOMAIN`, `MISSING_AUTH`, `INVALID_AUTH_FORMAT`): Fix the token generation code — these will not resolve by retrying.
- **Signing errors** (`INVALID_SIGNATURE`, `INVALID_SIWE`): The token is malformed. Regenerate using the viem-based signing code.

```typescript
import { createWalletClient, http } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { createSiweMessage, generateSiweNonce } from "viem/siwe";

async function generateSiweToken(privateKey: `0x${string}`): Promise<string> {
  const account = privateKeyToAccount(privateKey);
  const message = createSiweMessage({
    address: account.address,
    chainId: base.id,
    domain: "mpp.alchemy.com",
    nonce: generateSiweNonce(),
    uri: "https://mpp.alchemy.com",
    version: "1",
    statement: "Sign in to Alchemy Gateway",
    expirationTime: new Date(Date.now() + 60 * 60 * 1000),
  });
  const client = createWalletClient({ account, chain: base, transport: http() });
  const signature = await client.signMessage({ message });
  return `${btoa(message)}.${signature}`;
}

if (response.status === 401) {
  const body = await response.json();

  if (body.code === "MESSAGE_EXPIRED") {
    // Token expired — regenerate and retry
    const newToken = await generateSiweToken(privateKey);
    // retry request with new token...
  } else {
    // Configuration or signing error — do not retry, fix the code
    throw new Error(`Auth failed (${body.code}): ${body.message}`);
  }
}
```
