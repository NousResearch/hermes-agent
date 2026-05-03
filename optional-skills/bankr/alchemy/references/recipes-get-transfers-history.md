---
id: references/recipes-get-transfers-history.md
name: 'Recipe: Get Transfer History'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-transfers-api.md
updated: 2026-02-05
---
# Recipe: Get Transfer History

## Goal
Fetch historical transfers for a wallet or contract.

## Inputs
- `address`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Call `alchemy_getAssetTransfers` with `fromAddress` or `toAddress`.
2. Use `pageKey` to paginate.

## Example (curl)
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_getAssetTransfers","params":[{"fromBlock":"0x0","toBlock":"latest","fromAddress":"0x00000000219ab540356cbb839cbe05303d7705fa","category":["erc20","erc721","erc1155"],"withMetadata":true,"maxCount":"0x3e8"}]}'
```

## Output
- Transfer events with metadata and timestamps.

## Related Files
- `data-transfers-api.md`

## Official Docs
- [alchemy_getAssetTransfers](https://www.alchemy.com/docs/reference/alchemy-getassettransfers)
