---
id: references/recipes-simulate-transaction.md
name: 'Recipe: Simulate a Transaction'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-simulation-api.md
updated: 2026-02-05
---
# Recipe: Simulate a Transaction

## Goal
Simulate a transaction to preview asset changes and detect failures before sending.

## Inputs
- Transaction object (`from`, `to`, `data`, `value`)
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Build the transaction object.
2. Call `alchemy_simulateAssetChanges`.
3. Inspect the response for asset changes and revert reasons.

## Example (curl)
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_simulateAssetChanges","params":[{"from":"0x00000000219ab540356cbb839cbe05303d7705fa","to":"0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48","data":"0x"},"latest"]}'
```

## Output
- Simulated asset changes and execution details.

## Related Files
- `data-simulation-api.md`

## Official Docs
- [simulateAssetChanges](https://www.alchemy.com/docs/reference/simulateassetchanges-sdk)
