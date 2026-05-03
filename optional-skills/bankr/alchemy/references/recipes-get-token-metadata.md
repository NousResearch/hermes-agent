---
id: references/recipes-get-token-metadata.md
name: 'Recipe: Get Token Metadata'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-token-api.md
updated: 2026-02-05
---
# Recipe: Get Token Metadata

## Goal
Resolve token metadata (name, symbol, decimals, logo) for a contract.

## Inputs
- `contractAddress`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Call `alchemy_getTokenMetadata`.
2. Cache results for repeated use.

## Example (curl)
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_getTokenMetadata","params":["0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"]}'
```

## Example (TypeScript)
```ts
const url = `https://eth-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`;
const body = {
  jsonrpc: "2.0",
  id: 1,
  method: "alchemy_getTokenMetadata",
  params: ["0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"],
};
const res = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const json = await res.json();
```

## Output
- Token name, symbol, decimals, logo URI (when available).

## Related Files
- `data-token-api.md`

## Official Docs
- [alchemy_getTokenMetadata](https://www.alchemy.com/docs/reference/alchemy-gettokenmetadata)
