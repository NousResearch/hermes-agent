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

## If `ALCHEMY_API_KEY` Is NOT Set (MPP Flow)

Use `viem` for auth token generation and `mppx` for payment handling.

### When to Use

- Answering quick blockchain questions (latest block, ETH balance, token balance, NFT ownership, token prices, portfolio data)
- Making a few API calls from the command line or a bash script
- No existing npm project and you don't want to set one up

For SDK-based workflows with automatic payment handling, see [making-requests](making-requests.md) instead.

### Step 0: Ensure Wallet Exists

Follow [wallet-bootstrap](wallet-bootstrap.md) before proceeding. Do NOT generate or import a wallet from this file — the wallet-bootstrap rule contains a mandatory user prompt that must be followed.

### Step 1: Generate an Auth Token

**Important:** Use `domain: 'mpp.alchemy.com'` to target the MPP gateway.

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
TOKEN=$(cat siwe-token.txt)
```

> **Important:** Auth tokens expire after 1 hour by default. Adjust the `expirationTime` in the script to customize. If you get a 401 `MESSAGE_EXPIRED` error, regenerate the token (see Step 4). Always add token files to `.gitignore`.

### Step 2: Make API Calls with curl

All gateway endpoints share the same base URL (`https://mpp.alchemy.com`) and auth pattern. See [reference](reference.md) for the full list of supported endpoints, chain network slugs, and API methods.

---

### Node JSON-RPC (`/:chainNetwork/v2`)

#### Get the Latest Block Number (EVM chain)

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -X POST "https://mpp.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'
```

#### Get the Latest Slot (Solana chain)

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -X POST "https://mpp.alchemy.com/solana-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"getSlot"}'
```

#### Get ETH Balance for an Address

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -X POST "https://mpp.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_getBalance","params":["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045","latest"]}'
```

#### Get SOL Balance for an Address

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -X POST "https://mpp.alchemy.com/solana-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"getBalance","params":["83astBRguLMdt2h5U1Tbd2hpAXRC8gZDjX6BY1BV9Nc7"]}'
```

#### Read a Contract (e.g. USDC `balanceOf`)

The `eth_call` method lets you call read-only contract functions. For ERC-20 `balanceOf`, the data is the function selector `0x70a08231` followed by the address padded to 32 bytes:

```bash
TOKEN=$(cat siwe-token.txt)

# USDC balanceOf(0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045) on Ethereum Mainnet
# USDC contract: 0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
curl -s -X POST "https://mpp.alchemy.com/eth-mainnet/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_call","params":[{"to":"0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48","data":"0x70a08231000000000000000000000000d8dA6BF26964aF9D7eEd9e03E53415D37aA96045"},"latest"]}'
```

---

### NFT API (`/:chainNetwork/nft/v3/*`)

#### Get NFTs Owned by an Address

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -G "https://mpp.alchemy.com/eth-mainnet/nft/v3/getNFTsForOwner" \
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
TOKEN=$(cat siwe-token.txt)

curl -s -G "https://mpp.alchemy.com/prices/v1/tokens/by-symbol" \
  --data-urlencode "symbols=ETH" \
  --data-urlencode "symbols=BTC" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN"
```

> **Note:** Prices and Portfolio APIs are not chain-specific. A SIWE token can be used for authentication — as can all other gateway endpoints.

---

### Portfolio API (`/data/v1/assets/*`)

#### Get Token Balances Across Chains

```bash
TOKEN=$(cat siwe-token.txt)

curl -s -X POST "https://mpp.alchemy.com/data/v1/assets/tokens/by-address" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"addresses":["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],"withMetadata":true}'
```

### Step 3: Handle 402 Payment Required

If curl returns HTTP 402, the gateway requires payment. The handling depends on which payment method the user chose during setup. Extract the `WWW-Authenticate` header and create a credential for the correct method.

For full details on both payment methods, see [payment](payment.md).

**Note:** After a successful payment, subsequent requests using the same auth token will return 200 without requiring payment again.

#### Tempo (on-chain USDC)

```bash
TOKEN=$(cat siwe-token.txt)
CHAIN="eth-mainnet"  # Replace with any supported chain slug

HTTP_CODE=$(curl -s -o response.json -D headers.txt -w "%{http_code}" -X POST "https://mpp.alchemy.com/$CHAIN/v2" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}')

if [ "$HTTP_CODE" = "402" ]; then
  WWW_AUTH=$(grep -i 'www-authenticate:' headers.txt | sed 's/^[^:]*: //' | tr -d '\r')

  # Select the Tempo challenge and create credential
  CREDENTIAL=$(node -e "
    const { Challenge, Credential } = require('mppx');
    const fs = require('fs');
    const challenges = Challenge.fromHeaders(new Headers({ 'WWW-Authenticate': process.argv[1] }));
    const tempo = challenges.find(c => c.method === 'tempo');
    const privateKey = fs.readFileSync('./wallet-key.txt', 'utf8').trim();
    Credential.from(tempo, { privateKey }).then(c => {
      process.stdout.write(Credential.serialize(c));
    });
  " "$WWW_AUTH")

  curl -s -X POST "https://mpp.alchemy.com/$CHAIN/v2" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Authorization: SIWE $TOKEN, Payment $CREDENTIAL" \
    -d '{"id":1,"jsonrpc":"2.0","method":"eth_blockNumber"}'
else
  cat response.json
fi
```

#### Stripe (credit card)

Stripe payments are best handled through `mppx/client` in a Node.js/browser environment, since the `createToken` callback needs to proxy through a server endpoint. Curl-only workflows are not practical for Stripe — see [payment](payment.md) for the full Stripe setup with `mppx/client`.

### Step 4: Handle 401 MESSAGE_EXPIRED

If curl returns HTTP 401 with `"code":"MESSAGE_EXPIRED"`, the auth token has expired. Regenerate it:

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
# Retry the request with the new token
```

For other 401 error codes, see [authentication](authentication.md) for the full list of auth error codes.
