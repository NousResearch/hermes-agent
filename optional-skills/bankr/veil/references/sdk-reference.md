# Veil SDK Reference

Full documentation: https://github.com/veildotcash/veildotcash-sdk

## Installation

```bash
npm install -g @veil-cash/sdk
# or local: npm install @veil-cash/sdk  (yarn/pnpm equivalent)
```

## Supported Assets

| Asset | Decimals | Token Contract |
|-------|----------|---------------|
| ETH   | 18       | Native ETH (via WETH) |
| USDC  | 6        | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |

## CLI Commands

### `veil init`

Generate a new Veil keypair.

```bash
# Recommended for Bankr: derive from a pre-computed EIP-191 signature
veil init --json --signature 0x...

# Derive from wallet key (same keypair as frontend login)
veil init --json --sign-message --wallet-key 0x...

# Random keypair (no wallet derivation)
veil init --json

# Other options
veil init                                  # Interactive, saves to .env.veil
veil init --force                          # Overwrite existing without prompting
veil init --no-save                        # Print keypair without saving
```

JSON output:
```json
{
  "veilKey": "0x...",
  "veilPrivateKey": "0x...",
  "depositKey": "0x...",
  "derivation": "provided-signature"
}
```

`derivation` can be: `"wallet-signature"`, `"provided-signature"`, or `"random"`.

> **Note on key field names:** `veilKey` and `veilPrivateKey` are the same value (the Veil private key). Both are included for backward compatibility; `veil keypair` uses `veilPrivateKey`. Scripts should prefer `veilPrivateKey` with a fallback: `.veilPrivateKey // .veilKey`.

### `veil keypair`

Show current Veil keypair as JSON (from VEIL_KEY env).

```bash
veil keypair
# {"veilPrivateKey":"0x...","depositKey":"0x..."}
```

### `veil status`

Check configuration and service status.

```bash
veil status
```

Output:
```json
{
  "walletKey": { "found": true, "address": "0x..." },
  "veilKey": { "found": true },
  "depositKey": { "found": true, "key": "0x1234...abcd" },
  "rpcUrl": { "found": false, "url": "https://mainnet.base.org" },
  "registration": {
    "checked": true,
    "registered": true,
    "matches": true,
    "onChainKey": "0x..."
  },
  "relay": {
    "checked": true,
    "healthy": true,
    "status": "ok",
    "network": "mainnet"
  }
}
```

### `veil register`

Register or update your deposit key on-chain.

```bash
veil register                              # Register (first time)
veil register --json                       # JSON output
veil register --unsigned --address 0x...   # Unsigned payload for agents

# Change deposit key (if already registered with a different key)
veil register --force                      # Change to local deposit key
veil register --force --unsigned           # Unsigned change payload for agents
```

If already registered with the same key, the command exits successfully. If registered with a different key (e.g. after `veil init --sign-message`), use `--force` to update it on-chain.

### `veil deposit <asset> <amount>`

Deposit ETH or USDC into the privacy pool. For USDC, the CLI automatically handles ERC20 approval before depositing.

```bash
veil deposit ETH 0.1                    # Deposit ETH
veil deposit USDC 100                   # Approve + deposit USDC
veil deposit ETH 0.1 --unsigned         # Unsigned payload for agents
veil deposit USDC 100 --unsigned        # Unsigned: outputs [approve, deposit] array
veil deposit ETH 0.1 --quiet            # Suppress progress output
```

Output (signed mode):
```json
{
  "success": true,
  "hash": "0x...",
  "asset": "ETH",
  "amount": "0.1",
  "blockNumber": "12345678",
  "gasUsed": "150000"
}
```

Output (`--unsigned`, ETH — single object):
```json
{
  "step": "deposit",
  "to": "0x...",
  "data": "0x...",
  "value": "100000000000000000",
  "chainId": 8453
}
```

Output (`--unsigned`, USDC — array):
```json
[
  {
    "step": "approve",
    "to": "0x...",
    "data": "0x...",
    "value": "0",
    "chainId": 8453
  },
  {
    "step": "deposit",
    "to": "0x...",
    "data": "0x...",
    "value": "0",
    "chainId": 8453
  }
]
```

### `veil balance`

Show both queue and private balances.

```bash
veil balance                        # All pools (default)
veil balance --pool eth             # ETH pool only
veil balance --pool usdc            # USDC pool only
veil balance --quiet                # Suppress progress output
```

Output:
```json
{
  "address": "0x...",
  "pool": "ETH",
  "symbol": "ETH",
  "depositKey": "0x...",
  "totalBalance": "0.15",
  "totalBalanceWei": "150000000000000000",
  "private": {
    "balance": "0.10",
    "balanceWei": "100000000000000000",
    "utxoCount": 2,
    "utxos": [
      { "index": 5, "amount": "0.05" },
      { "index": 8, "amount": "0.05" }
    ]
  },
  "queue": {
    "balance": "0.05",
    "balanceWei": "50000000000000000",
    "count": 1,
    "deposits": [
      { "nonce": 42, "amount": "0.05", "status": "pending" }
    ]
  }
}
```

### `veil withdraw <asset> <amount> <recipient>`

Withdraw from the privacy pool to any public address.

```bash
veil withdraw ETH 0.05 0xRecipientAddress
veil withdraw USDC 50 0xRecipientAddress
veil withdraw ETH 0.05 0xRecipientAddress --quiet
```

Output:
```json
{
  "success": true,
  "transactionHash": "0x...",
  "blockNumber": 12345678,
  "asset": "ETH",
  "amount": "0.05",
  "recipient": "0x..."
}
```

### `veil transfer <asset> <amount> <recipient>`

Transfer privately to another registered Veil user.

```bash
veil transfer ETH 0.02 0xRecipientAddress
veil transfer USDC 25 0xRecipientAddress
veil transfer ETH 0.02 0xRecipientAddress --quiet
```

Output:
```json
{
  "success": true,
  "transactionHash": "0x...",
  "blockNumber": 12345678,
  "asset": "ETH",
  "amount": "0.02",
  "recipient": "0x...",
  "type": "transfer"
}
```

### `veil merge <asset> <amount>`

Consolidate multiple small UTXOs into one (self-transfer).

```bash
veil merge ETH 0.1
veil merge USDC 100
veil merge ETH 0.1 --quiet
```

Output:
```json
{
  "success": true,
  "transactionHash": "0x...",
  "blockNumber": 12345678,
  "asset": "ETH",
  "amount": "0.1",
  "type": "merge"
}
```

## Environment Variables

The CLI uses two config files:

| File       | Purpose                                                                 |
| ---------- | ----------------------------------------------------------------------- |
| `.env.veil` | Veil keypair (VEIL_KEY, DEPOSIT_KEY) — created by `veil init`           |
| `.env`      | Wallet config (WALLET_KEY, RPC_URL) — your existing config              |

| Variable     | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| VEIL_KEY     | Your Veil private key (for ZK proofs, withdrawals, transfers) |
| DEPOSIT_KEY  | Your Veil deposit key (public, for register/deposit)          |
| WALLET_KEY   | Ethereum wallet private key (for signing transactions)        |
| RPC_URL      | Base RPC URL (optional, defaults to public RPC)               |

## Error Handling

Commands that fail output JSON with standardized error codes:

```json
{
  "success": false,
  "errorCode": "VEIL_KEY_MISSING",
  "error": "VEIL_KEY required. Use --veil-key or set VEIL_KEY env"
}
```

## Error Codes

| Code                  | Description                      |
| --------------------- | -------------------------------- |
| VEIL_KEY_MISSING      | VEIL_KEY not provided            |
| WALLET_KEY_MISSING    | WALLET_KEY not provided          |
| DEPOSIT_KEY_MISSING   | DEPOSIT_KEY not provided         |
| INVALID_ADDRESS       | Invalid Ethereum address format  |
| INVALID_AMOUNT        | Invalid or below minimum amount  |
| INSUFFICIENT_BALANCE  | Not enough ETH balance           |
| USER_NOT_REGISTERED   | Recipient not registered in Veil |
| NO_UTXOS              | No unspent UTXOs available       |
| RELAY_ERROR           | Error from relayer service       |
| RPC_ERROR             | RPC/network error                |
| CONTRACT_ERROR        | Smart contract reverted          |
| UNKNOWN_ERROR         | Unexpected error                 |

## For AI Agents

All commands output JSON and support non-interactive usage.

### Key initialization (recommended: Bankr signature flow)

```bash
# 1. Get a personal_sign signature from Bankr's /agent/sign endpoint
#    (sign the VEIL_SIGNED_MESSAGE constant)

# 2. Derive keypair from that signature
veil init --json --signature 0x<BANKR_SIGNATURE>
# Returns: { "veilKey": "0x...", "veilPrivateKey": "0x...", "depositKey": "0x...", "derivation": "provided-signature" }

# Fallback: random keypair (no wallet derivation)
veil init --json
```

Parse the private key defensively: `.veilPrivateKey // .veilKey`

### Unsigned transaction payloads

```bash
veil register --unsigned --address 0x...
veil deposit ETH 0.1 --unsigned
veil deposit USDC 100 --unsigned    # Outputs [approve, deposit] array
```

Use `--unsigned` to get [Bankr-compatible transaction payloads](https://github.com/BankrBot/moltbot-skills/blob/main/bankr/references/arbitrary-transaction.md):

```bash
veil deposit ETH 0.1 --unsigned
# {"step":"deposit","to":"0x...","data":"0x...","value":"100000000000000000","chainId":8453}

veil deposit USDC 100 --unsigned
# [{"step":"approve","to":"0x...","data":"0x...","value":"0","chainId":8453},{"step":"deposit","to":"0x...","data":"0x...","value":"0","chainId":8453}]
```

### Querying

```bash
veil balance --quiet
veil balance --pool usdc --quiet
veil withdraw ETH 0.05 0xRecipient --quiet
```

Use `--quiet` to suppress progress output for clean JSON parsing.
