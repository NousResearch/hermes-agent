# Wallet & Payment Setup

> **Skip this entirely if `ALCHEMY_API_KEY` is set.** Wallet setup is only needed for the MPP gateway flow.

Use this rule when the MPP gateway flow needs a wallet. If a wallet file (e.g. `wallet-key.txt`) already exists on disk, use it and proceed directly to authentication.

**This is the mandatory entry point for MPP gateway requests when no API key is available.** No data can be fetched until setup is complete. If the user asks a blockchain question and no wallet is configured (and no API key is set), redirect them here first. Do NOT mention the API key or suggest obtaining one — go straight to setup.

## Step 1: Choose a Payment Method (Hard Requirement)

You MUST ask the user which payment method they want to use. Present this prompt exactly:

> How would you like to pay for API requests?
>
> 1. **Tempo** — on-chain USDC payment (gasless, EVM wallet required). Requires a wallet funded with USDC.
> 2. **Stripe** — credit card payment. No wallet funding needed — just a card.

**Do NOT skip this prompt. Do NOT pick a payment method on behalf of the user.** Wait for their explicit choice before proceeding.

**Record the user's choice:** Set `PAYMENT_METHOD = tempo` or `PAYMENT_METHOD = stripe`.

---

## Step 2: Wallet Setup

A wallet is required for **both** payment methods — it provides the SIWE auth token needed to authenticate with the gateway. The difference is that **Tempo** requires the wallet to be funded with USDC, while **Stripe** does not.

Both Tempo and Stripe use EVM wallets (SIWE auth).

### If wallet files already exist on disk (e.g. `wallet-key.txt`)

Use the existing wallet and proceed directly to:
- [Step 3: Fund the Wallet](#step-3-fund-the-wallet-tempo-only) (if Tempo)
- [Step 4: Generate Auth Token](#step-4-generate-auth-token) (if Stripe)

### If no wallet is configured

Ask the user:

> 1. **EVM — create a new wallet**
> 2. **EVM — import an existing private key**

Set `ARCHITECTURE = evm`.

Do not generate a wallet, import a key, or proceed to any other step until the user answers.

### Path A: Use an Existing Connected Wallet

If the user already has a wallet available (e.g. a private key in an environment variable or config file), proceed directly to Step 3 (Tempo) or Step 4 (Stripe).

### Path B: Import an Existing Wallet

Ask the user where their private key file is located. Extract the key into `wallet-key.txt` using a shell pipe:

```bash
# Example: extract from a .env file
grep PRIVATE_KEY /path/to/.env | cut -d '=' -f2 > wallet-key.txt

# Example: key is already the sole content of a file
cp /path/to/keyfile wallet-key.txt
```

> **Important:** Never use agent tools (Read, Write, Edit) on ANY file that may contain a private key. Always use shell pipes. Never echo or print key contents to stdout.

Verify the imported key:

```bash
node -e "const { privateKeyToAccount } = require('viem/accounts'); const fs = require('fs'); const pk = fs.readFileSync('./wallet-key.txt', 'utf8').trim(); console.log(JSON.stringify({ address: privateKeyToAccount(pk).address }));"
```

Add the key file to `.gitignore`:

```bash
echo "wallet-key.txt" >> .gitignore
```

### Path C: Create a New Wallet

```bash
node -e "const { generatePrivateKey } = require('viem/accounts'); process.stdout.write(generatePrivateKey());" > wallet-key.txt
echo "wallet-key.txt" >> .gitignore
node -e "const { privateKeyToAccount } = require('viem/accounts'); const fs = require('fs'); const pk = fs.readFileSync('./wallet-key.txt', 'utf8').trim(); console.log(JSON.stringify({ address: privateKeyToAccount(pk).address }));"
```

> **Important:** Never run `wallet generate` without piping to a file — it prints the private key to stdout.

---

## Step 3: Fund the Wallet (Tempo only)

**Skip this step if the user chose Stripe.** Stripe payments use a credit card — no USDC funding is needed.

Tempo requires USDC on an EVM network (e.g. Base Mainnet). Transfer USDC to the wallet address displayed during wallet setup.

---

## Step 4: Generate Auth Token

Generate a SIWE auth token for the MPP gateway. This is required for **both** Tempo and Stripe.

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

Proceed to [making-requests](making-requests.md) or [curl-workflow](curl-workflow.md).

---

## Using the Wallet in Code

For building applications, use `viem` for wallet management and `mppx` for payments. Always read the private key from an environment variable — never hardcode it in source files:

```typescript
import { generatePrivateKey, privateKeyToAccount } from "viem/accounts";

const privateKey = generatePrivateKey();
const account = privateKeyToAccount(privateKey);
// account.address → "0x..."

const existingKey = process.env.PRIVATE_KEY as `0x${string}`;
const existingAccount = privateKeyToAccount(existingKey);
// existingAccount.address → "0x..."
```
