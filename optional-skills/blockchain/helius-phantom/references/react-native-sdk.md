# React Native SDK Reference

Complete reference for `@phantom/react-native-sdk` — integrate Phantom into mobile apps built with Expo.

## Prerequisites

All Phantom Connect integrations require:

1. **Phantom Portal Account** — Register at phantom.com/portal
2. **App ID** — Get from Portal (required when using Google or Apple auth providers)
3. **Allowlisted URLs** — Add your redirect URLs in Portal settings

## Auth Providers

| Provider     | Description                     | Requires appId |
| ------------ | ------------------------------- | -------------- |
| `"google"`   | Google OAuth (embedded wallet)  | Yes            |
| `"apple"`    | Apple ID (embedded wallet)      | Yes            |
| `"deeplink"` | Phantom mobile app via deeplink | Yes            |

React Native does not support the `"injected"` provider (no browser extension on mobile). Use `"google"` and/or `"apple"` for social login, or `"deeplink"` to connect to the Phantom mobile app directly.

## Installation

```bash
npm install @phantom/react-native-sdk

# Expo peer dependencies
npx expo install expo-secure-store expo-web-browser expo-auth-session expo-router react-native-svg

# Required polyfill
npm install react-native-get-random-values

# For Solana support
npm install @solana/kit @solana-program/system @solana-program/compute-budget
```

## Critical Setup

### 1. Polyfill (MUST BE FIRST IMPORT)

```tsx
// App.tsx or _layout.tsx - THIS MUST BE THE VERY FIRST IMPORT
import "react-native-get-random-values";

import { PhantomProvider } from "@phantom/react-native-sdk";
// ... other imports
```

### 2. Configure app.json (Expo)

```json
{
  "expo": {
    "name": "My Wallet App",
    "slug": "my-wallet-app",
    "scheme": "mywalletapp",
    "plugins": [
      "expo-router",
      "expo-secure-store",
      "expo-web-browser",
      "expo-auth-session"
    ]
  }
}
```

## PhantomProvider Configuration

```tsx
import "react-native-get-random-values";
import { PhantomProvider, AddressType, darkTheme } from "@phantom/react-native-sdk";

export default function App() {
  return (
    <PhantomProvider
      config={{
        providers: ["google", "apple"],
        appId: "your-app-id",
        scheme: "mywalletapp",
        addressTypes: [AddressType.solana],
        authOptions: {
          redirectUrl: "mywalletapp://phantom-auth-callback",
        },
      }}
      theme={darkTheme}
      appIcon="https://yourapp.com/icon.png"
      appName="Your App"
    >
      <App />
    </PhantomProvider>
  );
}
```

## Available Hooks

| Hook           | Purpose                  | Returns                                |
| -------------- | ------------------------ | -------------------------------------- |
| `useModal`     | Control connection modal | `{ open, close, isOpened }`            |
| `usePhantom`   | Access wallet/user state | `{ isConnected, isLoading }`           |
| `useConnect`   | Connect to wallet        | `{ connect, isConnecting, error }`     |
| `useAccounts`  | Get wallet addresses     | `{ addresses, isConnected, walletId }` |
| `useDisconnect`| Disconnect wallet        | `{ disconnect, isDisconnecting }`      |
| `useSolana`    | Solana operations        | `{ solana, isAvailable }`              |

## Hook Examples

### useModal (Recommended Approach)

```tsx
import { View, Button, Text } from "react-native";
import { useModal, useAccounts } from "@phantom/react-native-sdk";

export function WalletScreen() {
  const { open, close, isOpened } = useModal();
  const { isConnected, addresses } = useAccounts();

  if (!isConnected) {
    return (
      <View style={{ padding: 20 }}>
        <Button title="Connect Wallet" onPress={open} />
      </View>
    );
  }

  return (
    <View style={{ padding: 20 }}>
      <Text>Connected!</Text>
      {addresses.map((addr, i) => (
        <Text key={i}>{addr.addressType}: {addr.address}</Text>
      ))}
      <Button title="Manage Wallet" onPress={open} />
    </View>
  );
}
```

### useConnect (Direct Connection)

```tsx
import { View, Button, Text, Alert } from "react-native";
import { useConnect, useAccounts, useDisconnect } from "@phantom/react-native-sdk";

export function WalletScreen() {
  const { connect, isConnecting, error } = useConnect();
  const { addresses, isConnected } = useAccounts();
  const { disconnect } = useDisconnect();

  const handleConnect = async () => {
    try {
      await connect({ provider: "google" });
      Alert.alert("Success", "Wallet connected!");
    } catch (err) {
      Alert.alert("Error", err.message);
    }
  };

  if (!isConnected) {
    return (
      <View>
        <Button
          title={isConnecting ? "Connecting..." : "Connect with Google"}
          onPress={handleConnect}
          disabled={isConnecting}
        />
        {error && <Text style={{ color: "red" }}>{error.message}</Text>}
      </View>
    );
  }

  return (
    <View>
      {addresses.map((addr, i) => (
        <Text key={i}>{addr.addressType}: {addr.address}</Text>
      ))}
      <Button title="Disconnect" onPress={disconnect} />
    </View>
  );
}
```

### useSolana

```tsx
import { Alert } from "react-native";
import { useSolana } from "@phantom/react-native-sdk";

function SolanaActions() {
  const { solana, isAvailable } = useSolana();

  if (!isAvailable) return null;

  const signMessage = async () => {
    try {
      const { signature } = await solana.signMessage("Hello from Solana!");
      Alert.alert("Signed!", signature.slice(0, 20) + "...");
    } catch (err) {
      Alert.alert("Error", err.message);
    }
  };

  // For transactions: use signTransaction, then submit via Helius Sender
  // See references/transactions.md for the full sign → Sender flow
  const sendTransaction = async (transaction: any) => {
    try {
      const signedTx = await solana.signTransaction(transaction);
      // Submit to Helius Sender — see references/helius-sender.md
      const serialized = signedTx.serialize();
      const base64Tx = Buffer.from(serialized).toString("base64");

      const response = await fetch("https://sender.helius-rpc.com/fast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: "1",
          method: "sendTransaction",
          params: [base64Tx, {
            encoding: "base64",
            skipPreflight: true,
            maxRetries: 0,
          }],
        }),
      });
      const result = await response.json();
      Alert.alert("Sent!", `TX: ${result.result}`);
    } catch (err) {
      Alert.alert("Error", err.message);
    }
  };

  return <Button title="Sign Message" onPress={signMessage} />;
}
```

## Authentication Flow

1. User taps "Connect Wallet"
2. System browser opens (Safari iOS / Chrome Android)
3. User authenticates with Google or Apple
4. Browser redirects back via custom URL scheme
5. SDK processes auth result automatically
6. Wallet connected and ready

### Redirect URL Format

```
{scheme}://phantom-auth-callback?wallet_id=...&session_id=...
```

## Security Features

- **iOS**: Keychain Services with hardware security
- **Android**: Android Keystore with hardware-backed keys
- Uses system browser (not in-app webview)
- Verifies redirect origins automatically

## Debug Configuration

```tsx
<PhantomProvider
  config={config}
  debugConfig={{
    enabled: true,
  }}
>
  ...
</PhantomProvider>
```

## Solana Configuration

```tsx
<PhantomProvider
  config={{
    providers: ["google", "apple"],
    appId: "your-app-id",
    scheme: "mycompany-wallet",
    addressTypes: [AddressType.solana],
    authOptions: {
      redirectUrl: "mycompany-wallet://auth/success",
    },
  }}
  theme={darkTheme}
>
  ...
</PhantomProvider>
```

## Supported Solana Networks

| Network | Cluster      |
| ------- | ------------ |
| Mainnet | mainnet-beta |
| Devnet  | devnet       |
| Testnet | testnet      |

## RPC Configuration

For RPC calls, use your backend API URL. Never hardcode a Helius API key in mobile app code — it can be extracted from the binary.

```tsx
// Use your backend proxy for all RPC calls
const RPC_PROXY = "https://yourapi.com/api/rpc";
```

See `references/frontend-security.md` for backend proxy patterns.

## Common Mistakes

- **`react-native-get-random-values` not imported first** — the app will crash on startup. This polyfill must be the very first import in your entry file.
- **Using `signAndSendTransaction` instead of `signTransaction` + Helius Sender** — use `signTransaction` to sign, then POST to `https://sender.helius-rpc.com/fast`. See `references/transactions.md`.
- **Missing `appId`** — required for Google/Apple providers. Get it from phantom.com/portal.
- **Auth redirect not working** — verify `scheme` in app.json matches config, ensure all Expo plugins are configured, run `npx expo prebuild` after changes.
- **Hardcoding API keys in mobile code** — mobile app binaries can be decompiled. Always use a backend proxy for Helius API calls.
