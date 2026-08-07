---
name: x402-payment
description: Make autonomous x402 payments on Arc Testnet using USDC. Handles HTTP 402 Payment Required responses automatically.
version: 1.0.0
metadata:
  hermes:
    tags: [web3, payments, arc, usdc, x402]
    category: blockchain
---

# x402 Payment Skill — Arc Testnet

## When to Use
- User wants to access an x402-protected API endpoint
- User wants to make a USDC micropayment on Arc Testnet
- User encounters HTTP 402 Payment Required response
- User wants to test agentic payments on Arc Testnet

## Procedure
1. Load the x402 client tool: `~/.hermes/skills/x402-payment/tools/x402_client.py`
2. Check wallet balance before payment
3. Send request to target endpoint
4. If HTTP 402 received, parse payment requirements
5. Sign USDC payment authorization (EIP-3009)
6. Retry request with X-PAYMENT header
7. Return response to user with payment receipt

## Pitfalls
- Always check wallet balance before attempting payment
- Arc Testnet USDC faucet: https://faucet.circle.com
- Never expose private keys in logs or output
- Max payment per request: $1.00 USDC (safety limit)

## Verification
- Check transaction on Arc explorer: https://testnet.arcscan.app
- Confirm X-PAYMENT-RESPONSE header in successful response
