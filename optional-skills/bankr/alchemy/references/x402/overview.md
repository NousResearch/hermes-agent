# Gateway Overview

The Alchemy Agentic Gateway lets agents easily access Alchemy's developer platform.

**If `ALCHEMY_API_KEY` is set**, use standard Alchemy endpoints directly (e.g. `https://{chainNetwork}.g.alchemy.com/v2/$ALCHEMY_API_KEY`). No wallet, auth token, or payment setup is needed. Skip the rest of this document.

**If `ALCHEMY_API_KEY` is NOT set**, proceed directly with wallet-based authentication. Do NOT mention the API key, suggest obtaining one, or list it as an alternative — go straight to x402 wallet setup. The gateway authenticates with SIWE (EVM wallet) or SIWS (Solana wallet) and handles per-request payments with USDC via the x402 protocol.

> **Wallet type vs query chain:** Your wallet type (EVM or Solana) determines how you authenticate and pay. It does NOT restrict which chains you can query — a SIWE or SIWS token works with any supported chain URL. NEVER suggest a wallet type based on the chain being queried (e.g. "Since we're querying Ethereum, we'll use EVM" is wrong). Always ask the user which wallet type they prefer without reference to the query chain.

## Base URL

```
https://x402.alchemy.com
```

The gateway exposes four API routes:

| Route | Example | Description |
|-------|---------|-------------|
| `/:chainNetwork/v2` | `/eth-mainnet/v2` | Node JSON-RPC + Token API + Transfers API |
| `/:chainNetwork/nft/v3/*` | `/eth-mainnet/nft/v3/getNFTsForOwner?owner=0x...` | NFT API v3 (REST) |
| `/data/v1/*` | `/data/v1/assets/tokens/by-address` | Portfolio API (not chain-specific) |
| `/prices/v1/*` | `/prices/v1/tokens/by-symbol?symbols=ETH` | Prices API (current + historical) |

See [reference](reference.md) for all endpoints, supported chains, and available methods.

## End-to-End Flow

1. **Set up a wallet (REQUIRED — must complete before any data request)** — Ask the user to either create a new wallet or provide an existing private key. EVM: `npx @alchemy/x402 wallet generate`; Solana: `npx @alchemy/x402 wallet generate --architecture svm`. See [wallet-bootstrap](wallet-bootstrap.md).
2. **Fund the wallet** — Load USDC on a supported payment network (Base for EVM wallets, Solana for SVM wallets).
3. **Create an auth token** — EVM: `npx @alchemy/x402 sign --private-key ./wallet-key.txt` or `signSiwe()` in code; Solana: `npx @alchemy/x402 sign --architecture svm --private-key ./wallet-key.txt` or `signSiws()` in code.
4. **Send a request** — Call any gateway route with your auth token (`Authorization: SIWE <token>` for EVM wallets, `Authorization: SIWS <token>` for Solana wallets). The chain in the URL is independent of your wallet type. For quick queries without an npm project, see the [curl-workflow](curl-workflow.md) for a lightweight curl-based alternative.
5. **Handle 402 Payment Required** — If the gateway returns 402, create an x402 payment with `npx @alchemy/x402 pay` (add `--architecture svm` for Solana) or `createPayment()` / `createSolanaPayment()` and retry with a `Payment-Signature` header.
6. **Receive the result** — After payment, the gateway proxies the request to Alchemy and returns the result. Subsequent requests with the same auth token do not require payment again.

## Packages

### `@alchemy/x402` — CLI + Library (recommended)

```bash
npm install @alchemy/x402
```

Provides both CLI commands and library utilities for wallet management, SIWE/SIWS authentication, and x402 payments:

| CLI command | Library function | Purpose |
|-------------|-----------------|---------|
| `npx @alchemy/x402 wallet generate` | `generateWallet()` | Create a new EVM wallet |
| `npx @alchemy/x402 wallet generate --architecture svm` | `generateSolanaWallet()` | Create a new Solana wallet |
| `npx @alchemy/x402 wallet import` | `getWalletAddress()` | Import / verify an EVM wallet |
| `npx @alchemy/x402 wallet import --architecture svm` | `getSolanaWalletAddress()` | Import / verify a Solana wallet |
| `npx @alchemy/x402 sign --private-key <key>` | `signSiwe()` | Generate a SIWE auth token (EVM) |
| `npx @alchemy/x402 sign --architecture svm --private-key <key>` | `signSiws()` | Generate a SIWS auth token (Solana) |
| `npx @alchemy/x402 pay` | `createPayment()` | Create an EVM x402 payment from a 402 response |
| `npx @alchemy/x402 pay --architecture svm` | `createSolanaPayment()` | Create a Solana x402 payment from a 402 response |
| — | `buildX402Client()` | Create an EVM x402 client for use with `@x402/fetch` or `@x402/axios` |
| — | `buildSolanaX402Client()` | Create a Solana x402 client for use with `@x402/fetch` or `@x402/axios` |

### Additional packages for app development

For building applications with automatic payment handling, also install a fetch/axios wrapper:

```bash
npm install @alchemy/x402 @x402/fetch   # or @x402/axios
```

| Package | Purpose |
|---------|---------|
| `@x402/fetch` | `wrapFetchWithPayment` — auto-handles 402 → sign → retry with `fetch` |
| `@x402/axios` | `wrapAxiosWithPayment` — auto-handles 402 → sign → retry with `axios` |

## Building Your Own x402 Endpoints                                                                                                                                                     
                                                                                                                                                                                          
If you want to deploy your own paid API endpoints using the same x402 protocol, [Bankr x402 Cloud](https://x402.bankr.bot) lets you write a handler, set a price in USDC, and deploy with one command — callers pay per request using the same @x402/fetch pattern you used above.
