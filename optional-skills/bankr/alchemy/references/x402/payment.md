# Manual x402 Payment

When the gateway returns **402**, you must create an x402 payment and retry the request with a `Payment-Signature` header. The payment method depends on your wallet type (EVM or Solana), not the chain being queried.

## CLI: Create a Payment

For ad-hoc requests and curl workflows, use the `@alchemy/x402` CLI. Pass the path to a file containing the private key:

### EVM Path

```bash
PAYMENT_REQUIRED=$(grep -i 'payment-required:' headers.txt | sed 's/^[^:]*: //' | tr -d '\r')
npx @alchemy/x402 pay --private-key ./wallet-key.txt --payment-required "$PAYMENT_REQUIRED"
```

### Solana Path

```bash
PAYMENT_REQUIRED=$(grep -i 'payment-required:' headers.txt | sed 's/^[^:]*: //' | tr -d '\r')
npx @alchemy/x402 pay --architecture svm --private-key ./wallet-key.txt --payment-required "$PAYMENT_REQUIRED"
```

The `--payment-required` flag expects the **base64-encoded** header value as-is from the HTTP response. Do not re-encode it — the header is already base64. The command creates a signed payment and prints the encoded `Payment-Signature` value to stdout.

### Full 402 Handling with curl (EVM Wallet)

```bash
TOKEN=$(cat siwe-token.txt)
CHAIN="eth-mainnet"  # Replace with any supported chain slug

# Save response headers and capture HTTP status code
HTTP_CODE=$(curl -s -o response.json -D headers.txt -w "%{http_code}" -X POST "https://x402.alchemy.com/$CHAIN/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}')

if [ "$HTTP_CODE" = "402" ]; then
  # Extract the PAYMENT-REQUIRED header value (already base64-encoded)
  PAYMENT_REQUIRED=$(grep -i 'payment-required:' headers.txt | sed 's/^[^:]*: //' | tr -d '\r')

  # Generate payment signature using the CLI
  PAYMENT_SIG=$(npx @alchemy/x402 pay --private-key ./wallet-key.txt --payment-required "$PAYMENT_REQUIRED")

  # Retry with payment
  curl -s -X POST "https://x402.alchemy.com/$CHAIN/v2" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Authorization: SIWE $TOKEN" \
    -H "Payment-Signature: $PAYMENT_SIG" \
    -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'
else
  cat response.json
fi
```

### Full 402 Handling with curl (Solana Wallet)

```bash
TOKEN=$(cat siws-token.txt)
CHAIN="solana-mainnet"  # Replace with any supported chain slug

# Save response headers and capture HTTP status code
HTTP_CODE=$(curl -s -o response.json -D headers.txt -w "%{http_code}" -X POST "https://x402.alchemy.com/$CHAIN/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWS $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"getSlot"}')

if [ "$HTTP_CODE" = "402" ]; then
  # Extract the PAYMENT-REQUIRED header value (already base64-encoded)
  PAYMENT_REQUIRED=$(grep -i 'payment-required:' headers.txt | sed 's/^[^:]*: //' | tr -d '\r')

  # Generate payment signature using the CLI (note --architecture svm)
  PAYMENT_SIG=$(npx @alchemy/x402 pay --architecture svm --private-key ./wallet-key.txt --payment-required "$PAYMENT_REQUIRED")

  # Retry with payment
  curl -s -X POST "https://x402.alchemy.com/$CHAIN/v2" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Authorization: SIWS $TOKEN" \
    -H "Payment-Signature: $PAYMENT_SIG" \
    -d '{"id":1,"jsonrpc":"2.0","method":"getSlot"}'
else
  cat response.json
fi
```

## Library: Create a Payment in Code

For applications, use `createPayment` (EVM) or `createSolanaPayment` (Solana) from `@alchemy/x402`. Read the private key from an environment variable — never hardcode it:

```bash
npm install @alchemy/x402
```

### EVM Path

```typescript
import { createPayment } from "@alchemy/x402";

const privateKey = process.env.PRIVATE_KEY as `0x${string}`;
const paymentSignature = await createPayment({
  privateKey,
  paymentRequiredHeader: response.headers.get("PAYMENT-REQUIRED")!,
});
```

### Solana Path

```typescript
import { createSolanaPayment } from "@alchemy/x402";

const privateKey = process.env.PRIVATE_KEY as string; // base58-encoded
const paymentSignature = await createSolanaPayment({
  privateKey,
  paymentRequiredHeader: response.headers.get("PAYMENT-REQUIRED")!,
});
```

Then retry the request with the `Payment-Signature` header:

```typescript
const chainUrl = "https://x402.alchemy.com/{chainNetwork}/v2"; // any supported chain
const retryResponse = await fetch(chainUrl, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
    Authorization: `SIWE ${token}`, // or `SIWS ${token}` for Solana wallet
    "Payment-Signature": paymentSignature,
  },
  body: JSON.stringify({
    id: 1,
    jsonrpc: "2.0",
    method: "eth_blockNumber",
  }),
});
```

## Payment Details

- **Asset**: USDC (6 decimals)
- **Scheme**: `exact`
  - **EVM**: Uses EIP-3009 gasless `transferWithAuthorization`
  - **Solana**: Uses SVM x402 payment scheme via `@x402/svm`
- **Amount**: Specified in the 402 response `accepts` array (USDC atomic units, 6 decimals)
- **Signing**: The `@alchemy/x402` package signs the payment authorization. No on-chain gas is needed from the payer.
- **Settlement**: The gateway's facilitator submits the signed authorization on-chain.

## Selecting a Payment Option

The 402 `accepts` array may contain multiple options (different networks). The CLI and `createPayment()` / `createSolanaPayment()` automatically select a compatible option. When using `buildX402Client` or `buildSolanaX402Client` directly, you can pass a custom selector:

### EVM Path

```typescript
import { buildX402Client } from "@alchemy/x402";

const privateKey = process.env.PRIVATE_KEY as `0x${string}`;
// buildX402Client returns an x402Client configured for Base Mainnet and Base Sepolia
const client = buildX402Client(privateKey);
```

### Solana Path

```typescript
import { buildSolanaX402Client } from "@alchemy/x402";

const privateKey = process.env.PRIVATE_KEY as string; // base58-encoded
// buildSolanaX402Client returns an x402Client configured for Solana payments
const client = buildSolanaX402Client(privateKey);
```

## Payment Error Responses

If a payment fails verification or settlement, the gateway returns 402 with additional error info:

```json
{
  "x402Version": 2,
  "error": "Payment verification failed: insufficient_balance",
  "accepts": [ ... ],
  "extensions": {
    "paymentError": {
      "info": {
        "type": "verify_failed",
        "reason": "insufficient_balance",
        "payer": "0x..."
      }
    }
  }
}
```

Error types: `verify_failed`, `settle_failed`, `settle_timeout`, `payment_error`.
