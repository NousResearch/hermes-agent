# Transfers Reference

Transfer tokens to addresses, ENS names, or social handles.

## CLI Command

```bash
# Transfer with token symbol resolution
bankr wallet transfer --to <recipient> --token <symbol> --amount <amount>
bankr wallet transfer --to <recipient> --token <symbol> --amount <amount> --chain <chain>

# Examples
bankr wallet transfer --to vitalik.eth --token USDC --amount 50 --chain base
bankr wallet transfer --to 0x1234... --token ETH --amount 0.1
bankr wallet transfer --to @friend --token USDC --amount 20
```

The `--token` flag resolves token symbols (e.g. `USDC`) to contract addresses via the search API.

## REST API

```bash
# Direct transfer via Wallet API
curl -X POST "https://api.bankr.bot/wallet/transfer" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"to": "vitalik.eth", "token": "USDC", "amount": "50", "chain": "base"}'
```

The `/wallet/transfer` endpoint is a write endpoint — requires `walletApiEnabled`, `readOnly: false`, and is subject to `allowedRecipients` enforcement and IP allowlist.

## Supported Transfers

- **EVM Chains**: Base, Polygon, Ethereum (mainnet), Unichain, World Chain, Arbitrum, BNB Chain
  - Native tokens: ETH, POL, BNB
  - ERC20 tokens: USDC, USDT, WETH, etc.
- **Solana**: SOL and SPL tokens

## Recipient Formats

| Format | Example | Description |
|--------|---------|-------------|
| Address | `0x1234...abcd` | Direct wallet address (EVM) |
| Address | `9x...abc` | Direct wallet address (Solana) |
| ENS | `vitalik.eth` | Ethereum Name Service |
| Twitter | `@elonmusk` | Twitter/X username |
| Farcaster | `@dwr.eth` | Farcaster username |
| Telegram | `@username` | Telegram handle |

**Social Handle Resolution**: Handles are resolved to linked wallet addresses before sending. User must have linked their wallet to the social platform.

## Amount Formats

| Format | Example | Description |
|--------|---------|-------------|
| USD | `$50` | Dollar amount |
| Percentage | `50%` | Percentage of balance |
| Exact | `0.1 ETH` | Specific amount |

## Prompt Examples

**To addresses:**
- "Send 0.5 ETH to 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
- "Transfer 100 USDC to 9xKc...abc"
- "Send $20 of ETH to 0x1234..."

**To ENS:**
- "Send 1 ETH to vitalik.eth"
- "Transfer $50 of USDC to mydomain.eth"
- "Send 10 USDC to friend.eth"

**To social handles:**
- "Send $20 of ETH to @friend on Twitter"
- "Transfer 0.1 ETH to @user on Farcaster"
- "Send 50 USDC to @buddy on Telegram"

**With chain specified:**
- "Send ETH on Base to vitalik.eth"
- "Send 10% of my ETH to @friend"
- "Transfer USDC on Polygon to 0x..."

## Chain Selection

If not specified, Bankr selects automatically based on:
- Recipient activity patterns
- Gas costs
- Token availability
- Liquidity

Specify chain in prompt if you need a specific network.

## Common Issues

| Issue | Resolution |
|-------|------------|
| ENS not found | Verify the ENS name exists and is registered |
| Social handle not found | Check username spelling and platform |
| No linked wallet | User hasn't linked wallet to their social account |
| Insufficient balance | Reduce amount or ensure enough funds |
| Wrong chain | Specify chain explicitly in prompt |
| Gas required | Ensure you have native token for gas |

## Security Notes

- **Verify recipient** - Always double-check before confirming
- **Address preview** - Social handle resolution shows the resolved address
- **Irreversible** - Blockchain transactions cannot be undone
- **Large transfers** - May require additional confirmation
- **Test first** - Send small amount first for new recipients

## Best Practices

1. **Start small** - Test with small amounts for new recipients
2. **Verify address** - Double-check resolved addresses
3. **Check chain** - Ensure recipient uses the same chain
4. **Gas buffer** - Keep some native token for future transactions
5. **ENS preferred** - More reliable than social handles
6. **Screenshot** - Save transaction hash for records
