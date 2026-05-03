# Curl Workflow

A lightweight way to call Alchemy endpoints using curl.

## If `ALCHEMY_API_KEY` Is Set

No wallet or auth token needed. Just use the API key in the URL:

```bash
# Node JSON-RPC
curl -s -X POST "https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'

# NFT API
curl -s -G "https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTsForOwner" \
  --data-urlencode "owner=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045" \
  --data-urlencode "withMetadata=true" \
  --data-urlencode "pageSize=10"

# Prices API
curl -s -G "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/by-symbol" \
  --data-urlencode "symbols=ETH"

# Portfolio API
curl -s -X POST "https://api.g.alchemy.com/data/v1/$ALCHEMY_API_KEY/assets/tokens/by-address" \
  -H "Content-Type: application/json" \
  -d '{"addresses":["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],"withMetadata":true}'
```

Skip all steps below — no wallet, auth token, or payment handling needed.

---

## If `ALCHEMY_API_KEY` Is NOT Set (x402 Flow)

Use the `@alchemy/x402` CLI with wallet-based authentication and x402 USDC payments.

> **Auth vs chain:** Your wallet type determines the auth scheme (`SIWE` for EVM wallets, `SIWS` for Solana wallets) and payment commands. The chain URL in each curl request is independent — use whichever chain you want to query.

### When to Use

- Answering quick blockchain questions (latest block, ETH balance, token balance, NFT ownership, token prices, portfolio data)
- Making a few API calls from the command line or a bash script
- No existing npm project and you don't want to set one up

For SDK-based workflows with automatic payment handling, see [making-requests](making-requests.md) instead.

### Step 0: Ensure Wallet Exists

Follow [wallet-bootstrap](wallet-bootstrap.md) before proceeding. Do NOT generate or import a wallet from this file — the wallet-bootstrap rule contains a mandatory user prompt that must be followed.

### Step 1: Generate an Auth Token

### EVM Path

```bash
npx @alchemy/x402 sign --private-key ./wallet-key.txt > siwe-token.txt
TOKEN=$(cat siwe-token.txt)
```

### Solana Path

```bash
npx @alchemy/x402 sign --architecture svm --private-key ./wallet-key.txt > siws-token.txt
TOKEN=$(cat siws-token.txt)
```

> **Important:** Auth tokens expire after 1 hour by default. Use `--expires-after` to customize (e.g. `--expires-after 2h`). If you get a 401 `MESSAGE_EXPIRED` error, regenerate the token (see Step 4). Always add token files to `.gitignore`.

### Step 2: Make API Calls with curl

All gateway endpoints share the same base URL (`https://x402.alchemy.com`) and auth pattern. Use `$AUTH_SCHEME` and `$TOKEN` from Step 1 — the auth header depends on your wallet type, while the chain URL depends on what you're querying. See [reference](reference.md) for the full list of supported endpoints, chain network slugs, and API methods.

In the examples below, replace `$AUTH_SCHEME` with `SIWE` (EVM wallet) or `SIWS` (Solana wallet), and read `$TOKEN` from the appropriate token file.

---

### Node JSON-RPC (`/:chainNetwork/v2`)

#### Get the Latest Block Number (EVM chain)

```bash
# Works with either SIWE or SIWS token
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt for Solana wallet

curl -s -X POST "https://x402.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'
```

#### Get the Latest Slot (Solana chain)

```bash
# Works with either SIWE or SIWS token
TOKEN=$(cat siws-token.txt)  # or siwe-token.txt for EVM wallet

curl -s -X POST "https://x402.alchemy.com/solana-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWS $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"getSlot"}'
```

#### Get ETH Balance for an Address

```bash
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt

curl -s -X POST "https://x402.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_getBalance","params":["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045","latest"]}'
```

#### Get SOL Balance for an Address

```bash
TOKEN=$(cat siws-token.txt)  # or siwe-token.txt

curl -s -X POST "https://x402.alchemy.com/solana-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWS $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"getBalance","params":["83astBRguLMdt2h5U1Tbd2hpAXRC8gZDjX6BY1BV9Nc7"]}'
```

#### Read a Contract (e.g. USDC `balanceOf`)

The `eth_call` method lets you call read-only contract functions. For ERC-20 `balanceOf`, the data is the function selector `0x70a08231` followed by the address padded to 32 bytes:

```bash
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt

# USDC balanceOf(0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045) on Ethereum Mainnet
# USDC contract: 0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
curl -s -X POST "https://x402.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_call","params":[{"to":"0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48","data":"0x70a08231000000000000000000000000d8dA6BF26964aF9D7eEd9e03E53415D37aA96045"},"latest"]}'
```

---

### NFT API (`/:chainNetwork/nft/v3/*`)

#### Get NFTs Owned by an Address

```bash
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt

curl -s -G "https://x402.alchemy.com/eth-mainnet/nft/v3/getNFTsForOwner" \
  --data-urlencode "owner=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045" \
  --data-urlencode "withMetadata=true" \
  --data-urlencode "pageSize=10" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN"
```

---

### Prices API (`/prices/v1/tokens/*`)

#### Get Token Prices by Symbol

```bash
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt

curl -s -G "https://x402.alchemy.com/prices/v1/tokens/by-symbol" \
  --data-urlencode "symbols=ETH" \
  --data-urlencode "symbols=BTC" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN"
```

> **Note:** Prices and Portfolio APIs are not chain-specific. Either a SIWE or SIWS token can be used for authentication — as can all other gateway endpoints.

---

### Portfolio API (`/data/v1/assets/*`)

#### Get Token Balances Across Chains

```bash
TOKEN=$(cat siwe-token.txt)  # or siws-token.txt

curl -s -X POST "https://x402.alchemy.com/data/v1/assets/tokens/by-address" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"addresses":["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],"withMetadata":true}'
```

### Step 3: Handle 402 Payment Required

If curl returns HTTP 402, the gateway requires a one-time USDC payment for this auth token. Extract the `PAYMENT-REQUIRED` header and use the CLI to create a payment:

The payment command (`npx @alchemy/x402 pay`) depends on your wallet type, not the chain being queried. Use `--architecture svm` for Solana wallets.

### EVM Wallet Path
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

### Solana Wallet Path

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

  # Note: --architecture svm for Solana wallet payments
  PAYMENT_SIG=$(npx @alchemy/x402 pay --architecture svm --private-key ./wallet-key.txt --payment-required "$PAYMENT_REQUIRED")

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

For more details on the payment flow, see [payment](payment.md).

**Note:** After a successful payment, subsequent requests using the same auth token will return 200 without requiring payment again.

### Step 4: Handle 401 MESSAGE_EXPIRED

If curl returns HTTP 401 with `"code":"MESSAGE_EXPIRED"`, the auth token has expired. Regenerate it:

### EVM Path

```bash
npx @alchemy/x402 sign --private-key ./wallet-key.txt > siwe-token.txt
# Retry the request with the new token
```

### Solana Path

```bash
npx @alchemy/x402 sign --architecture svm --private-key ./wallet-key.txt > siws-token.txt
# Retry the request with the new token
```

For other 401 error codes, see [authentication](authentication.md) for the full list of auth error codes.
