# Browser SDK Reference

Complete reference for `@phantom/browser-sdk` — for vanilla JS, non-React frameworks, or lightweight integrations.

## Prerequisites

All Phantom Connect integrations require:

1. **Phantom Portal Account** — Register at phantom.com/portal
2. **App ID** — Get from Portal (required when using Google or Apple auth providers)
3. **Allowlisted URLs** — Add your domains and redirect URLs in Portal settings

## Auth Providers

| Provider      | Description                     | Requires appId |
| ------------- | ------------------------------- | -------------- |
| `"injected"`  | Phantom browser extension       | No             |
| `"google"`    | Google OAuth (embedded wallet)  | Yes            |
| `"apple"`     | Apple ID (embedded wallet)      | Yes            |
| `"deeplink"`  | Phantom mobile app via deeplink | Yes            |

Use `"injected"` for extension-only flows (no appId needed). Add `"google"` and/or `"apple"` for social login (requires appId from Phantom Portal). Add `"deeplink"` to support connecting to the Phantom mobile app on devices where the extension is not available.

## Installation

```bash
npm install @phantom/browser-sdk

# For Solana support
npm install @solana/kit @solana-program/system @solana-program/compute-budget
```

## Quick Start Template

Generate a project with the Phantom Embedded JS Starter:

```bash
npx -y create-solana-dapp@latest -t solana-foundation/templates/community/phantom-embedded-js
```

## SDK Initialization

### Injected Provider Only (Extension)

```ts
import { BrowserSDK, AddressType } from "@phantom/browser-sdk";

const sdk = new BrowserSDK({
  providers: ["injected"],
  addressTypes: [AddressType.solana],
});
```

### Multiple Auth Methods

```ts
const sdk = new BrowserSDK({
  providers: ["google", "apple", "injected"],
  appId: "your-app-id",
  addressTypes: [AddressType.solana],
  authOptions: {
    authUrl: "https://connect.phantom.app/login", // optional
    redirectUrl: "https://yourapp.com/callback",  // required for OAuth
  },
  autoConnect: true,
});
```

## Connection

### Basic Connection

```ts
// Connect with specific provider
const { addresses } = await sdk.connect({ provider: "google" });
const { addresses } = await sdk.connect({ provider: "apple" });
const { addresses } = await sdk.connect({ provider: "injected" });

console.log("Connected:", addresses);
// [{ address: "...", addressType: "solana" }]
```

### Auto-Connect

```ts
const sdk = new BrowserSDK({
  providers: ["google", "apple"],
  appId: "your-app-id",
  addressTypes: [AddressType.solana],
  autoConnect: true, // Automatically reconnect existing sessions
});

// Or manually trigger
await sdk.autoConnect();
```

### Disconnect

```ts
await sdk.disconnect();
```

## Solana Operations (sdk.solana)

```ts
// Sign message
const { signature, rawSignature } = await sdk.solana.signMessage("Hello Solana!");

// Sign transaction (without sending) — recommended for Helius Sender flow
const signedTx = await sdk.solana.signTransaction(transaction);
// Then submit to Helius Sender — see references/transactions.md

// Network switching
await sdk.solana.switchNetwork("devnet"); // "mainnet-beta", "testnet", "devnet"

// Utilities
const publicKey = await sdk.solana.getPublicKey();
const isConnected = sdk.solana.isConnected();
```

## Auto-Confirm (Injected Provider Only)

```ts
import { NetworkId } from "@phantom/browser-sdk";

// Enable for specific chains
await sdk.enableAutoConfirm({
  chains: [NetworkId.SOLANA_MAINNET]
});

// Enable for all supported chains
await sdk.enableAutoConfirm();

// Disable
await sdk.disableAutoConfirm();

// Get status
const status = await sdk.getAutoConfirmStatus();

// Get supported chains
const chains = await sdk.getSupportedAutoConfirmChains();
```

## Extension Detection

```ts
import { waitForPhantomExtension } from "@phantom/browser-sdk";

const isAvailable = await waitForPhantomExtension(5000); // 5s timeout

if (isAvailable) {
  console.log("Phantom extension installed");
} else {
  console.log("Extension not found - offer OAuth login");
}
```

## Wallet Discovery

Discover all injected wallets using Wallet Standard (Solana):

```ts
// Async discovery
const wallets = await sdk.discoverWallets();
console.log(wallets);
// [
//   { id: "phantom", name: "Phantom", icon: "...", addressTypes: [...] },
//   { id: "backpack", name: "Backpack", icon: "...", addressTypes: [...] },
// ]

// Get already discovered (sync)
const cachedWallets = sdk.getDiscoveredWallets();
```

## Event Handlers

```ts
// Connection started
sdk.on("connect_start", (data) => {
  console.log("Starting:", data.source); // "auto-connect" | "manual-connect"
});

// Connection successful
sdk.on("connect", (data) => {
  console.log("Connected:", data.addresses);
  console.log("Provider:", data.provider);
});

// Connection failed
sdk.on("connect_error", (data) => {
  console.error("Failed:", data.error);
});

// Disconnected
sdk.on("disconnect", (data) => {
  console.log("Disconnected");
});

// General errors
sdk.on("error", (error) => {
  console.error("SDK Error:", error);
});

// Remove listener
sdk.off("connect", handleConnect);
```

### Events with Auto-Connect

```ts
const sdk = new BrowserSDK({
  providers: ["google"],
  appId: "your-app-id",
  addressTypes: [AddressType.solana],
  autoConnect: true,
});

// Set up listeners BEFORE autoConnect triggers
sdk.on("connect", (data) => {
  updateUI(data.addresses);
});

sdk.on("connect_error", (data) => {
  showLoginButton();
});

await sdk.autoConnect();
```

## Debug Configuration

```ts
import { DebugLevel } from "@phantom/browser-sdk";

// Enable/disable at runtime
sdk.enableDebug();
sdk.disableDebug();

// Set level
sdk.setDebugLevel(DebugLevel.INFO);
// Levels: ERROR (0), WARN (1), INFO (2), DEBUG (3)

// Set callback
sdk.setDebugCallback((message) => {
  console.log(`[${message.level}] ${message.category}: ${message.message}`);
});

// Configure all at once
sdk.configureDebug({
  enabled: true,
  level: DebugLevel.DEBUG,
  callback: (msg) => console.log(msg),
});
```

## AddressType Values

| AddressType            | Chains                   |
| ---------------------- | ------------------------ |
| `AddressType.solana`   | Mainnet, Devnet, Testnet |

## Supported Solana Networks

| Network | Cluster      |
| ------- | ------------ |
| Mainnet | mainnet-beta |
| Devnet  | devnet       |
| Testnet | testnet      |

## Complete Example

```ts
import { BrowserSDK, AddressType, waitForPhantomExtension } from "@phantom/browser-sdk";

// Initialize
const sdk = new BrowserSDK({
  providers: ["google", "apple", "injected"],
  appId: "your-app-id",
  addressTypes: [AddressType.solana],
  autoConnect: true,
});

// Set up event handlers
sdk.on("connect", ({ addresses }) => {
  document.getElementById("status").textContent = `Connected: ${addresses[0].address}`;
});

sdk.on("connect_error", ({ error }) => {
  document.getElementById("status").textContent = `Error: ${error.message}`;
});

// Connect button
document.getElementById("connectBtn").addEventListener("click", async () => {
  const hasExtension = await waitForPhantomExtension(2000);
  const provider = hasExtension ? "injected" : "google";
  await sdk.connect({ provider });
});

// Sign message button
document.getElementById("signBtn").addEventListener("click", async () => {
  const { signature } = await sdk.solana.signMessage("Hello!");
  console.log("Signature:", signature);
});

// Disconnect button
document.getElementById("disconnectBtn").addEventListener("click", async () => {
  await sdk.disconnect();
  document.getElementById("status").textContent = "Disconnected";
});
```

## Common Mistakes

- **Using `signAndSendTransaction` instead of `signTransaction` + Helius Sender** — `signAndSendTransaction` submits through standard RPC. Use `signTransaction` to get the signed bytes, then POST to `https://sender.helius-rpc.com/fast` for better landing rates. See `references/transactions.md`.
- **Missing `appId` when using Google or Apple providers** — register at phantom.com/portal and add the appId to the BrowserSDK config.
- **Redirect URL not allowlisted** — go to phantom.com/portal, open app settings, and add the exact redirect URL (including protocol and path) to the allowlist.
- **Phantom extension not detected** — use `waitForPhantomExtension(5000)` with a timeout. If not found, fall back to social login providers (`"google"` or `"apple"`).
- **Exposing RPC endpoint with API key** — use a proxy URL instead of `https://mainnet.helius-rpc.com/?api-key=SECRET`. See `references/frontend-security.md`.
