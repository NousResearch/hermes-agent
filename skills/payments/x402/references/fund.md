# Funding the Wallet

Use the wallet companion app to fund the wallet with USDC via Coinbase Onramp. This supports multiple payment methods including Apple Pay, debit cards, bank transfers, and funding from a Coinbase account.

## Confirm wallet is initialized and authed

```bash
npx awal@latest status
```

If the wallet is not authenticated, see `authenticate.md`.

## Opening the Funding Interface

```bash
npx awal@latest show
```

This opens the wallet companion window where users can:

1. Select a preset amount ($10, $20, $50) or enter a custom amount
2. Choose their preferred payment method
3. Complete the purchase through Coinbase Pay

## Payment Methods

| Method    | Description                                    |
| --------- | ---------------------------------------------- |
| Apple Pay | Fast checkout with Apple Pay (where available) |
| Coinbase  | Transfer from existing Coinbase account        |
| Card      | Debit card payment                             |
| Bank      | ACH bank transfer                              |

## Alternative

You can also ask your human to send USDC on Base to your wallet address. Get the address with:

```bash
npx awal@latest address
```

## Prerequisites

- Must be authenticated (`npx awal@latest status` to check)
- Coinbase Onramp is available in supported regions (US, etc.)

## Flow

1. Run `npx awal@latest show` to open the wallet UI
2. Instruct the user to click the Fund button
3. User selects amount and payment method in the UI
4. User completes payment through Coinbase Pay (opens in browser)
5. USDC is deposited to the wallet once payment confirms

## Checking Balance After Funding

```bash
npx awal@latest balance
```

## Notes

- Funding goes through Coinbase's regulated onramp
- Processing time varies by payment method (instant for card/Apple Pay, 1-3 days for bank)
- Funds are deposited as USDC on Base network
- If funding is not available, users can also send USDC on Base directly to the wallet address
