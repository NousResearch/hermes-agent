# Wallet Setup

> **Skip this entirely if `ALCHEMY_API_KEY` is set.** Wallet setup is only needed for the x402 gateway flow.

Use this rule when the x402 gateway flow needs a wallet. If a wallet file (e.g. `wallet-key.txt`) already exists on disk, use it and proceed directly to authentication.

**This is the mandatory entry point for x402 gateway requests when no API key is available.** No data can be fetched until wallet setup is complete. If the user asks a blockchain question and no wallet is configured (and no API key is set), redirect them here first. Do NOT mention the API key or suggest obtaining one — go straight to wallet setup.

## Determine Wallet Type and Source (Hard Requirement)

### If wallet files already exist on disk (e.g. `wallet-key.txt`)

Use the existing wallet and proceed directly to authentication. If the wallet type (EVM or Solana) is not already known, ask the user.

### If no wallet is configured

You MUST ask the user a **single combined question** that presents all wallet options. Do not skip, assume, or infer the answer. Wait for an explicit response before taking any wallet action.

Present **all four options** in a single prompt — both EVM and Solana options MUST be included as equal choices:

> 1. **EVM — create a new wallet** (pays USDC on Base)
> 2. **EVM — import an existing private key** (pays USDC on Base)
> 3. **Solana — create a new wallet** (pays USDC on Solana)
> 4. **Solana — import an existing private key** (pays USDC on Solana)

**Do NOT assume EVM. Do NOT omit the Solana options.** Wallet type determines auth and payment method ONLY — it has absolutely nothing to do with which chains can be queried. A Solana wallet can query Ethereum, and an EVM wallet can query Solana.

### Anti-pattern: DO NOT do this

NEVER use the query chain to justify, suggest, or default to a wallet type. The following are **all wrong**:

- "Since we're querying Ethereum, we'll create an EVM wallet" — **WRONG**
- "You're looking up Solana data, so let's set up a Solana wallet" — **WRONG**
- "For this ETH mainnet query, an EVM wallet makes sense" — **WRONG**
- Skipping the wallet type question because the query chain "implies" the answer — **WRONG**
- Presenting only EVM options without Solana (or vice versa) — **WRONG**
- Splitting into two separate questions where the second question (Solana) can be dropped — **WRONG**

The query chain and the wallet type are **completely independent choices**. Never correlate them in your reasoning or your response to the user.

Do not generate a wallet, import a key, or proceed to any other step until the user answers.

**Record the user's choice:** Once the user answers, set `ARCHITECTURE = evm` or `ARCHITECTURE = svm`. Use the matching wallet path for auth token generation, payment signing, and wallet management commands.

Based on the answer, follow one of the three paths below.

---

## Path A: Use an Existing Connected Wallet

If the user already has a wallet available (e.g. a private key in an environment variable or config file), proceed directly to [Fund the Wallet](#fund-the-wallet).

---

## Path B: Import an Existing Wallet

Ask the user where their private key file is located (e.g. a `.env` file, a keystore export, a text file). Once you have the file path, extract the key into `wallet-key.txt` using a shell pipe so it never appears on screen or in tool output:

```bash
# Example: extract from a .env file
grep PRIVATE_KEY /path/to/.env | cut -d '=' -f2 > wallet-key.txt

# Example: key is already the sole content of a file
cp /path/to/keyfile wallet-key.txt
```

> **Important:** Never use agent tools (Read, Write, Edit) on ANY file that may contain a private key — including `wallet.json`, `wallet-key.txt`, `.env` files, or keystore exports. Always use shell pipes to move keys between files. Never echo or print key contents to stdout.

Verify the imported key:

### EVM Path

```bash
npx @alchemy/x402 wallet import --private-key ./wallet-key.txt
```

Output:
```json
{ "address": "0xYourChecksummedAddress" }
```

### Solana Path

```bash
npx @alchemy/x402 wallet import --architecture svm --private-key ./wallet-key.txt
```

> **Note:** Solana private keys can be in base58 format or JSON array format (e.g. from Solana CLI's `id.json`). Both are supported.

Output:
```json
{ "address": "YourBase58SolanaAddress" }
```

Add the key file to `.gitignore`:

```bash
echo "wallet-key.txt" >> .gitignore
```

Proceed to [Fund the Wallet](#fund-the-wallet).

---

## Path C: Create a New Wallet

Generate a wallet and pipe the private key directly to a file so it never appears on screen:

### EVM Path

```bash
npx @alchemy/x402 wallet generate | jq -r .privateKey > wallet-key.txt
echo "wallet-key.txt" >> .gitignore
```

Retrieve the wallet address (safe to display):

```bash
npx @alchemy/x402 wallet import --private-key ./wallet-key.txt
```

### Solana Path

```bash
npx @alchemy/x402 wallet generate --architecture svm | jq -r .privateKey > wallet-key.txt
echo "wallet-key.txt" >> .gitignore
```

Retrieve the wallet address (safe to display):

```bash
npx @alchemy/x402 wallet import --architecture svm --private-key ./wallet-key.txt
```

> **Important:** Never run `wallet generate` without piping to a file — it prints the private key to stdout.

Proceed to [Fund the Wallet](#fund-the-wallet).

---

## Fund the Wallet

### EVM Wallets

#### Testnet (Base Sepolia)

1. Go to the [Circle USDC faucet](https://faucet.circle.com/)
2. Select **Base Sepolia**
3. Paste your wallet address
4. Request testnet USDC

The USDC will arrive at your address on Base Sepolia (`0x036CbD53842c5426634e7929541eC2318f3dCF7e`).

#### Mainnet

Transfer USDC to your wallet address on Base Mainnet.

### Solana Wallets

#### Devnet

1. Go to the [Circle USDC faucet](https://faucet.circle.com/)
2. Select **Solana Devnet**
3. Paste your Solana wallet address
4. Request testnet USDC

#### Mainnet

Transfer USDC to your wallet address on Solana Mainnet.

## Using the Wallet in Code

For building applications, use the `@alchemy/x402` library. Always read the private key from an environment variable — never hardcode it in source files:

### EVM Path

```typescript
import { generateWallet, getWalletAddress } from "@alchemy/x402";

// Generate a new wallet (in a setup script, save privateKey to a secure location)
const wallet = generateWallet();
// wallet.address → "0x..."

// Or derive address from an existing key
const privateKey = process.env.PRIVATE_KEY as `0x${string}`;
const address = getWalletAddress(privateKey);
```

### Solana Path

```typescript
import { generateSolanaWallet, getSolanaWalletAddress } from "@alchemy/x402";

// Generate a new Solana wallet (save privateKey to a secure location)
const wallet = generateSolanaWallet();
// wallet.address → "Base58SolanaAddress"

// Or derive address from an existing key
const privateKey = process.env.PRIVATE_KEY as string; // base58-encoded
const address = getSolanaWalletAddress(privateKey);
```

Use the private key for auth token generation (see [authentication](authentication.md)) and payment signing (see [making-requests](making-requests.md)).
