# Single-Sided Liquidity

Deposit a single token into an auto-managed strategy on Hydrex to earn oHYDX yields and provide deep pool liquidity — without needing to supply both sides of a pair.

**Vault Deposit Guard:** `0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8` (Base)
**Vault Deployer:** `0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE` (Base)

## How It Works

1. You deposit a single asset (e.g., 100% BNKR)
2. An automated liquidity manager handles the position — no need to balance both sides yourself
3. You earn oHYDX yield on your deposited value
4. On withdrawal, you receive your position back as a mix of the deposit token and counter token — typically up to 70/30 by value (e.g., roughly 70% BNKR and 30% WETH), depending on where the price sits relative to when you entered
5. The strategy address is per-pair; get all available strategies from the API

**Note on displayed values**: APR, USD balances, and TVL figures from the API are estimates based on recent activity. They're useful for comparing strategies directionally but will shift as market conditions change. As with any liquidity position, token price movements relative to each other can affect the mix you receive on withdrawal.

## Discovering Opportunities

All single-sided strategies are available from:

```
https://api.hydrex.fi/strategies?strategist=ichi
```

Filter by deposit token address:

```
https://api.hydrex.fi/strategies?strategist=ichi&depositTokens=TOKEN_ADDRESS,TOKEN_ADDRESS
```

**Example — find BNKR deposit opportunities:**

```bash
bankr agent "What single-sided liquidity vaults can I deposit BNKR into on Hydrex?"
```

The API fetches from: `https://api.hydrex.fi/strategies?strategist=ichi&depositTokens=0x22af33fe49fd1fa80c7149773dde5890d3c76f3b`

**Key API fields per strategy:**

| Field | Description |
|-------|-------------|
| `address` | Vault address (used for deposit, withdraw, and balance calls) |
| `title` | `"DEPOSIT/COUNTER"` format — e.g., `"BNKR/WETH"` means deposit BNKR |
| `depositToken` | Address of the token you deposit |
| `childAPR` | Current average APR in oHYDX for depositors |
| `lpPriceUsd` | USD value per LP share |
| `tvlUsd` | Total value locked in this vault |

**Note**: Not all tokens have single-sided strategies. If you'd like a strategy for a specific token, reach out to the Hydrex team on [Discord](https://discord.gg/hydrexfi) or [Telegram](https://t.me/larrettgee).

## Depositing

### Natural Language

Always specify the strategy by title (e.g., `"BNKR/WETH"`) or vault address so Bankr can unambiguously resolve which vault to use.

```bash
bankr agent "Deposit 100 BNKR into the BNKR/WETH strategy on Hydrex"
bankr agent "Deposit 500 USDC into the USDC/HYDX strategy on Hydrex"
bankr agent "Deposit 1000 HYDX into vault 0xABC...123 on Hydrex"
```

### Steps Bankr Executes

1. **Resolve vault** — match the strategy by `title` (e.g., `"BNKR/WETH"`) or `address` from `https://api.hydrex.fi/strategies?strategist=ichi`; if ambiguous, ask the user to confirm the strategy title or address before proceeding
2. **Check allowance**: `allowance(userAddress, DEPOSIT_GUARD)` on the deposit token contract
3. **Approve if needed**: `approve(DEPOSIT_GUARD, amount)` on the deposit token
4. **Deposit**: call `forwardDepositToICHIVault` on Deposit Guard

### Deposit

**Function**: `forwardDepositToICHIVault(address vault, address vaultDeployer, address token, uint256 amount, uint256 minimumShares, address userAddress)`
**Contract**: `0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8` (Deposit Guard, Base)

```
Send transaction to 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8 on Base calling forwardDepositToICHIVault with vault [VAULT_ADDRESS], vaultDeployer 0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE, token [DEPOSIT_TOKEN_ADDRESS], amount [AMOUNT_IN_WEI], minimumShares 0, userAddress [USER_ADDRESS]
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| `vault` | From API `address` field |
| `vaultDeployer` | `0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE` (always) |
| `token` | Deposit token address (from API `depositToken`) |
| `amount` | Amount in wei (18 decimals for most tokens, 6 for USDC) |
| `minimumShares` | `0` (acceptable for most cases; use slippage calc for large deposits) |
| `userAddress` | User's wallet address |

**Result**: User receives LP shares (vault tokens) minted to `userAddress`.

## Withdrawing

### Natural Language

```bash
bankr agent "Withdraw my BNKR/WETH single-sided position on Hydrex"
bankr agent "Remove 50% of my BNKR single-sided liquidity on Hydrex"
bankr agent "Exit my Hydrex BNKR vault position"
```

### Steps Bankr Executes

1. **Get LP balance**: `balanceOf(userAddress)` on the vault contract
2. **Check LP allowance**: `allowance(userAddress, DEPOSIT_GUARD)` on the vault contract
3. **Approve LP if needed**: `approve(DEPOSIT_GUARD, shares)` on the vault contract
4. **Withdraw**: call `forwardWithdrawFromICHIVault` on Deposit Guard

### Withdraw Call

**Function**: `forwardWithdrawFromICHIVault(address vault, address vaultDeployer, uint256 shares, address userAddress, uint256 minAmount0, uint256 minAmount1)`
**Contract**: `0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8` (Deposit Guard, Base)

```
Send transaction to 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8 on Base calling forwardWithdrawFromICHIVault with vault [VAULT_ADDRESS], vaultDeployer 0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE, shares [LP_SHARES], userAddress [USER_ADDRESS], minAmount0 0, minAmount1 0
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| `vault` | Vault address (from API `address` or user's existing position) |
| `vaultDeployer` | `0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE` (always) |
| `shares` | LP token balance from `balanceOf` (partial withdrawals: multiply by fraction) |
| `userAddress` | User's wallet address |
| `minAmount0` | `0` (or calculate slippage on token0) |
| `minAmount1` | `0` (or calculate slippage on token1) |

**Result**: User receives both token0 and token1 from the vault (up to 70/30 deposit/counter split depending on vault position).

## Viewing Your Position

### Natural Language

```bash
bankr agent "Show my Hydrex single-sided liquidity positions"
bankr agent "What's my BNKR/WETH vault balance on Hydrex?"
bankr agent "How much is my Hydrex BNKR single-sided position worth?"
```

### Calculating Underlying Tokens

**Step 1 — Get user LP shares:**
**Function**: `balanceOf(address)` — standard ERC20, selector `0x70a08231`
**Contract**: Vault address (from API `address` field)

To read directly — encode user address as 32-byte padded hex and call `eth_call` on the vault. Returns `uint256` LP share balance.

**Step 2 — Get vault totals:**

**`totalSupply()`** — selector `0x18160ddd`, no parameters. Returns total LP shares outstanding.

**`getTotalAmounts()`** — returns `(uint256 totalToken0, uint256 totalToken1)`. Total underlying tokens in the vault.

**Step 3 — Calculate user's underlying:**

```
userToken0 = (userShares × totalToken0) / totalSupply
userToken1 = (userShares × totalToken1) / totalSupply
```

Multiply by `lpPriceUsd / totalSupply` from the API to get USD value.

## Function Reference

| Function | Contract | Parameters | Returns |
|----------|----------|------------|---------|
| `forwardDepositToICHIVault(address,address,address,uint256,uint256,address)` | Deposit Guard | vault, vaultDeployer, token, amount, minShares, user | — |
| `forwardWithdrawFromICHIVault(address,address,uint256,address,uint256,uint256)` | Deposit Guard | vault, vaultDeployer, shares, user, min0, min1 | — |
| `balanceOf(address)` | Vault | user address | uint256 LP shares |
| `totalSupply()` | Vault | — | uint256 total shares |
| `getTotalAmounts()` | Vault | — | (uint256 token0, uint256 token1) |
| `allowance(address,address)` | Token / Vault | owner, spender | uint256 |
| `approve(address,uint256)` | Token / Vault | spender, amount | bool |

## Contracts (Base Mainnet)

| Contract | Address |
|----------|---------|
| Vault Deposit Guard | `0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8` |
| Vault Deployer | `0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE` |

Vault addresses are per-pair — always retrieve from `https://api.hydrex.fi/strategies?strategist=ichi` (`address` field).

## Complete Workflow Examples

### Deposit BNKR into BNKR/WETH Vault

```bash
# 1. Find available BNKR vaults
bankr agent "What single-sided liquidity vaults can I deposit BNKR into on Hydrex?"

# 2. Check BNKR balance
bankr agent "What's my BNKR balance on Base?"

# 3. Deposit
bankr agent "Deposit 500 BNKR into the BNKR/WETH single-sided vault on Hydrex"

# 4. Confirm position
bankr agent "Show my Hydrex single-sided liquidity positions"
```

### Withdraw from BNKR/WETH Vault

```bash
# 1. Check current position
bankr agent "What's my BNKR/WETH vault balance on Hydrex?"

# 2. Full withdrawal
bankr agent "Withdraw my full BNKR/WETH single-sided position on Hydrex"

# 3. Partial withdrawal
bankr agent "Withdraw 25% of my BNKR single-sided position on Hydrex"
```

## Implementation Guide for Bankr

When a user requests single-sided liquidity operations:

### Resolving the Vault

1. Fetch strategies: `GET https://api.hydrex.fi/strategies?strategist=ichi`
2. Match by user intent:
   - By name: `title == "BNKR/WETH"` (first token = deposit token)
   - By token: `depositToken == userSpecifiedTokenAddress`
   - List all: return all strategies with `childAPR`, `tvlUsd`, `title`
3. Extract `address` (vault) and `depositToken` for the selected strategy

### Deposit Flow

```
1. GET https://api.hydrex.fi/strategies?strategist=ichi → find vault
2. eth_call allowance(userAddress, 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8) on depositToken
3. If allowance < amount:
     approve(0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8, amount) on depositToken
4. forwardDepositToICHIVault(vault, 0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE, depositToken, amount, 0, userAddress)
   on Deposit Guard 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8
```

### Withdraw Flow

```
1. eth_call balanceOf(userAddress) on vault → get LP shares
2. eth_call allowance(userAddress, 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8) on vault
3. If allowance < shares:
     approve(0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8, shares) on vault
4. forwardWithdrawFromICHIVault(vault, 0x7d11De61c219b70428Bb3199F0DD88bA9E76bfEE, shares, userAddress, 0, 0)
   on Deposit Guard 0x9A0EBEc47c85fD30F1fdc90F57d2b178e84DC8d8
```

### View Position Flow

```
1. eth_call balanceOf(userAddress) on vault → userShares
2. eth_call totalSupply() on vault → totalShares
3. eth_call getTotalAmounts() on vault → (totalToken0, totalToken1)
4. userToken0 = (userShares × totalToken0) / totalShares
   userToken1 = (userShares × totalToken1) / totalShares
```

## Tips

- **Partial withdrawals**: Multiply LP balance by the fraction to withdraw (e.g., 50% = `shares / 2`)
- **Slippage**: `minimumShares = 0` and `minAmount0/1 = 0` is acceptable for most users; for large positions consider calculating 1% slippage
- **Token ordering**: `title` format is always `"DEPOSIT/COUNTER"` — the first token is what you put in
- **APR**: `childAPR` is the oHYDX yield; actual returns also include trading fees from the pool
- **IL risk**: Wider price swings between the two tokens = larger impermanent loss potential
- **No strategy for your token?** Contact Hydrex on [Discord](https://discord.gg/hydrexfi) or [Telegram](https://t.me/larrettgee)
