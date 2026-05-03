---
id: references/recipes-get-nft-ownership.md
name: 'Recipe: Get NFT Ownership'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-nft-api.md
updated: 2026-02-05
---
# Recipe: Get NFT Ownership

## Goal
Fetch NFTs owned by a wallet address.

## Inputs
- `owner`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Call `getNFTsForOwner` via NFT API.
2. Use `pageKey` to paginate.

## Example (curl)
```bash
curl -s "https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTsForOwner?owner=0x00000000219ab540356cbb839cbe05303d7705fa"
```

## Example (TypeScript)
```ts
const url = new URL(
  `https://eth-mainnet.g.alchemy.com/nft/v3/${process.env.ALCHEMY_API_KEY}/getNFTsForOwner`
);
url.searchParams.set("owner", "0x00000000219ab540356cbb839cbe05303d7705fa");

const res = await fetch(url.toString());
const json = await res.json();
```

## Output
- List of NFTs with contract, tokenId, and metadata fields.

## Pitfalls
- Some NFTs have missing or invalid metadata.
- Spam collections may appear unless filtered; allowlist trusted collections when needed.

## Related Files
- `data-nft-api.md`

## Official Docs
- [NFT API Endpoints](https://www.alchemy.com/docs/reference/nft-api-endpoints)
