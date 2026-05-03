---
id: references/recipes-get-token-balances.md
name: 'Recipe: Get Token Balances'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-token-api.md
updated: 2026-02-05
---
# Recipe: Get Token Balances

## Goal
Fetch all ERC-20 token balances for a wallet.

## Inputs
- `address`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Call `alchemy_getTokenBalances` for the wallet.
2. Optionally filter by contract addresses.
3. Normalize balances using token decimals.

## Example (curl)
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_getTokenBalances","params":["0x00000000219ab540356cbb839cbe05303d7705fa"]}'
```

## Example (TypeScript)
```ts
const url = `https://eth-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`;
const body = {
  jsonrpc: "2.0",
  id: 1,
  method: "alchemy_getTokenBalances",
  params: ["0x00000000219ab540356cbb839cbe05303d7705fa"],
};
const res = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const json = await res.json();
```

## Output
- List of token contract addresses and balances.

## Pitfalls
- Some balances may be `0x0` or missing metadata; fetch metadata separately if needed.

## Related Files
- `data-token-api.md`

## Official Docs
- [Token API Quickstart](https://www.alchemy.com/docs/docs/reference/token-api-quickstart)
