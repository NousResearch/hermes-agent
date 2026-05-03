# Trails x Bankr: Integration Recipes

Three end-to-end flows combining **Trails REST API** with **Bankr** for agent funding, swaps, and DeFi.

**Bankr package**: `@bankr/cli` — `submit()` broadcasts the on-chain step using the Bankr agent wallet.
Note: `submit` is not re-exported from the main `@bankr/cli` entry point — import it directly:

```javascript
const { submit } = require('@bankr/cli/dist/lib/api.js');
```

**Trails**: REST API at `https://trails-api.sequence.app/rpc/Trails/<MethodName>` (header: `X-Access-Key: $TRAILS_API`).

---

## What is Trails?

Trails is an orchestration protocol . An agent specifies the *action* (e.g. "swap USDC on Polygon -> ETH on Base"), and Trails automatically determines the optimal multi-step path:

---

## Recipe 1: Fund a Bankr Agent Wallet (On-Ramp via Trails Widget)

Point the Trails widget at the Bankr agent wallet. Users pick any source token on any chain — Trails routes it through to the Bankr wallet on Polygon. No code required on the user's side.

### Step 1: Get the Bankr wallet address

```bash
curl -s https://api.bankr.bot/agent/me \
  -H "X-API-Key: $BANKR_API_KEY" \
  | jq '.wallets[] | select(.chain == "evm") | .address'
# -> "0x3ef200c4b8b153553b62906151a71c3ae82bfd5c"
```

### Step 2: Build the Trails funding URL

```
https://demo.trails.build/?mode=swap
  &toAddress=<BANKR_EVM_WALLET>
  &toChainId=137
  &toToken=0x3c499c542cef5e3811e1192ce70d8cC03d5c3359
  &apiKey=<TRAILS_API>
  &theme=light
```

| Param       | Value                                        | Notes                                     |
| ----------- | -------------------------------------------- | ----------------------------------------- |
| `toAddress` | Bankr EVM wallet                             | Destination — Bankr agent wallet receives |
| `toChainId` | `137`                                        | Polygon mainnet                           |
| `toToken`   | `0x3c499c542cef5e3811e1192ce70d8cC03d5c3359` | USDC on Polygon                           |
| `apiKey`    | `$TRAILS_API`                                | Sequence project key                      |

### Step 3: Share the URL

The user opens the link, connects any wallet on any chain, picks a source token, and Trails handles the rest — selects optimal bridges, handles multi-hop paths (e.g. SOL -> bridge -> USDC on Polygon), and executes the full sequence. USDC arrives in the Bankr wallet.

---

## Recipe 2: Swap from Bankr Wallet via Trails

Trails REST API provides the quote and routing. Bankr `submit()` broadcasts the `depositTransaction` on-chain. Trails handles the rest — same-chain or cross-chain.

### Step 1: Get Bankr wallet address

```bash
BANKR_WALLET=$(curl -s https://api.bankr.bot/agent/me \
  -H "X-API-Key: $BANKR_API_KEY" \
  | jq -r '.wallets[] | select(.chain == "evm") | .address')
```

### Step 2: Quote — Trails selects the optimal route

```bash
QUOTE=$(curl -s https://trails-api.sequence.app/rpc/Trails/QuoteIntent \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d "{
    \"ownerAddress\": \"$BANKR_WALLET\",
    \"originChainId\": 137,
    \"originTokenAddress\": \"0x3c499c542cef5e3811e1192ce70d8cC03d5c3359\",
    \"originTokenAmount\": \"10000000\",
    \"destinationChainId\": 8453,
    \"destinationTokenAddress\": \"0x0000000000000000000000000000000000000000\",
    \"destinationTokenAmount\": \"0\",
    \"tradeType\": \"EXACT_INPUT\",
    \"options\": { \"slippageTolerance\": 0.005 }
  }")

# Inspect the deposit tx and quote
echo $QUOTE | jq '.intent.depositTransaction, .intent.quote.toAmount, .intent.quote.estimatedDuration'
```

`originTokenAmount` is in base units — `10000000` = 10 USDC (6 decimals).

### Step 3: Commit — lock in the intent

```bash
INTENT_BODY=$(echo $QUOTE | jq '{intent: .intent}')
COMMIT=$(curl -s https://trails-api.sequence.app/rpc/Trails/CommitIntent \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d "$INTENT_BODY")
INTENT_ID=$(echo $COMMIT | jq -r '.intentId')
```

### Step 4: Submit the depositTransaction from Bankr wallet

```javascript
import { submit } from '@bankr/cli';

const depositTx = JSON.parse(process.env.DEPOSIT_TX); // from $QUOTE above
const result = await submit({
  transaction: {
    to: depositTx.to,
    chainId: depositTx.chainId,
    data: depositTx.data,
    value: depositTx.value ?? '0',
  },
  description: 'Trails swap deposit',
  waitForConfirmation: true,
});
const TX_HASH = result.transactionHash;
```

### Step 5: Execute — tell Trails the deposit is confirmed

```bash
curl -s https://trails-api.sequence.app/rpc/Trails/ExecuteIntent \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d "{\"intentId\": \"$INTENT_ID\", \"depositTransactionHash\": \"$TX_HASH\"}"
```

### Step 6: Poll until complete

```bash
curl -s https://trails-api.sequence.app/rpc/Trails/WaitIntentReceipt \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d "{\"intentId\": \"$INTENT_ID\"}" | jq '{done: .done, status: .intentReceipt.status}'
```

> **Send output to a different wallet**: Add `"destinationToAddress": "<other-wallet>"` to the QuoteIntent body. `ownerAddress` (Bankr) pays; the other wallet receives the output tokens.

---

## Recipe 3: Deposit into a Yield Vault from Bankr Wallet

Trails `GetEarnPools` discovers active vault addresses and APYs — no need to hard-code pool addresses or ABIs. Bankr `submit()` sends the approve and deposit transactions.

### Step 1: Discover pools

```bash
POOLS=$(curl -s https://trails-api.sequence.app/rpc/Trails/GetEarnPools \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d '{"chainIds": [137]}')

# Show active USDC pools sorted by TVL
echo $POOLS | jq '[.pools[] | select(.isActive and .token.symbol == "USDC")] | sort_by(-.tvl) | .[0] | {protocol, name, apy, tvl, depositAddress}'
```

Example output:

```json
{
  "protocol": "aave-v3",
  "name": "Aave v3 USDC",
  "apy": 0.0198,
  "tvl": 40700000,
  "depositAddress": "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
}
```

### Step 2: Approve the pool contract

```javascript
import { submit } from '@bankr/cli';
import { encodeFunctionData } from 'viem';

const USDC_ADDRESS = '0x3c499c542cef5e3811e1192ce70d8cC03d5c3359';
const DEPOSIT_AMOUNT = 50_000_000n; // 50 USDC (6 decimals)

const approveData = encodeFunctionData({
  abi: [{ name: 'approve', type: 'function', inputs: [{ name: 'spender', type: 'address' }, { name: 'amount', type: 'uint256' }], outputs: [{ type: 'bool' }] }],
  functionName: 'approve',
  args: [depositAddress, DEPOSIT_AMOUNT],
});

await submit({
  transaction: {
    to: USDC_ADDRESS,
    chainId: 137,
    data: approveData,
    value: '0',
  },
  description: 'Approve USDC for Aave deposit',
  waitForConfirmation: true,
});
```

### Step 3: Deposit into the vault

```javascript
const supplyData = encodeFunctionData({
  abi: [{ name: 'supply', type: 'function', inputs: [{ name: 'asset', type: 'address' }, { name: 'amount', type: 'uint256' }, { name: 'onBehalfOf', type: 'address' }, { name: 'referralCode', type: 'uint16' }], outputs: [] }],
  functionName: 'supply',
  args: [USDC_ADDRESS, DEPOSIT_AMOUNT, BANKR_WALLET, 0],
});

const result = await submit({
  transaction: {
    to: depositAddress, // pool.depositAddress from GetEarnPools
    chainId: 137,
    data: supplyData,
    value: '0',
  },
  description: 'Deposit 50 USDC into Aave v3',
  waitForConfirmation: true,
});
console.log('txHash:', result.transactionHash);
```

> For Morpho vaults, use `encodeFunctionData` with the `deposit(uint256 assets, address receiver)` ABI. `depositAddress` from `GetEarnPools` is always the correct contract to call.

---

## Trails REST API Reference

Base URL: `https://trails-api.sequence.app/rpc/Trails/<MethodName>`
Auth header: `X-Access-Key: $TRAILS_API`

| Group                 | Method                                   | Description                                                       |
| --------------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| **Intent lifecycle**  | `QuoteIntent`                            | Get quote + `depositTransaction` for a swap/bridge                |
|                       | `CommitIntent`                           | Lock the intent, receive `intentId`                               |
|                       | `ExecuteIntent`                          | Notify Trails the deposit tx is mined                             |
|                       | `WaitIntentReceipt`                      | Poll until intent is complete (`done: true`)                      |
| **Intent management** | `GetIntent`                              | Look up intent by ID                                              |
|                       | `GetIntentReceipt`                       | Get final receipt                                                 |
|                       | `SearchIntents`                          | List intents by owner/status                                      |
|                       | `GetIntentHistory`                       | Paginated history                                                 |
|                       | `AbortIntent`                            | Cancel a pending intent                                           |
| **Discovery**         | `GetEarnPools`                           | Active yield pools (Aave, Morpho) with APY, TVL, `depositAddress` |
|                       | `GetChains`                              | Supported chains (returns `null` if none configured)              |
|                       | `GetTokenList`                           | Tokens per chain — body: `{"chainIds":[137]}` (array, not scalar) |
|                       | `GetTokenPrices`                         | USD prices                                                        |
|                       | `GetExactInputRoutes`                    | Preview routes for exact-in                                       |
|                       | `GetExactOutputRoutes`                   | Preview routes for exact-out                                      |
| **Reference**         | `GetExchangeRate`                        | Fiat conversion                                                   |
|                       | `GetTrailsContracts`                     | Contract addresses per chain                                      |
|                       | `GetCountryList` / `GetFiatCurrencyList` | Supported countries/currencies                                    |
| **Utility**           | `Ping` / `RuntimeStatus` / `Clock`       | Health + server time                                              |

### QuoteIntent — key request fields

| Field                       | Type              | Notes                                                 |
| --------------------------- | ----------------- | ----------------------------------------------------- |
| `ownerAddress`              | string            | Wallet that pays and signs the deposit tx             |
| `originChainId`             | number            | Source chain ID                                       |
| `originTokenAddress`        | string            | Source token contract address                         |
| `originTokenAmount`         | string            | Base units (e.g. `"10000000"` = 10 USDC)              |
| `destinationChainId`        | number            | Destination chain ID (can differ for cross-chain)     |
| `destinationTokenAddress`   | string            | Destination token (`"0x000...000"` for native)        |
| `destinationTokenAmount`    | string            | `"0"` for `EXACT_INPUT`                               |
| `destinationToAddress`      | string (optional) | Send output to a different wallet than `ownerAddress` |
| `tradeType`                 | string            | `"EXACT_INPUT"` or `"EXACT_OUTPUT"`                   |
| `options.slippageTolerance` | number            | e.g. `0.005` = 0.5%                                   |

### GetEarnPools — pool object fields

| Field                            | Notes                                    |
| -------------------------------- | ---------------------------------------- |
| `protocol`                       | e.g. `"aave-v3"`, `"morpho"`             |
| `chainId`                        | Chain the pool is on                     |
| `apy`                            | e.g. `0.0198` = 1.98%                    |
| `tvl`                            | USD total value locked                   |
| `token.symbol` / `token.address` | Underlying asset                         |
| `depositAddress`                 | Contract to approve and call for deposit |
| `isActive`                       | Filter to `true` only                    |

---

## Environment Variables

```bash
export TRAILS_API=<your-sequence-project-access-key>   # from https://sequence.build
export BANKR_API_KEY=<your-bankr-key>                  # from bankr.bot/api (Agent API access required)
```

## References

- Bankr SDK: `npm install @bankr/cli` — `submit()`, `getUserInfo()`, `getBalances()`
- Bankr Agent API: `GET https://api.bankr.bot/agent/me` -> `wallets[]`
- Trails REST base: `https://trails-api.sequence.app/rpc/Trails/<MethodName>` (header: `X-Access-Key`)
- Trails widget: `https://demo.trails.build/`
