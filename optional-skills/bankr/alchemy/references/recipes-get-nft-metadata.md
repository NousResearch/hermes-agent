---
id: references/recipes-get-nft-metadata.md
name: 'Recipe: Get NFT Metadata'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-nft-api.md
updated: 2026-02-05
---
# Recipe: Get NFT Metadata

## Goal
Fetch metadata for a specific NFT token.

## Inputs
- `contractAddress`
- `tokenId`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Call `getNFTMetadata`.
2. Cache metadata in your database.

## Example (curl)
```bash
curl -s "https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTMetadata?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d&tokenId=1"
```

## Example (TypeScript)
```ts
const url = new URL(
  `https://eth-mainnet.g.alchemy.com/nft/v3/${process.env.ALCHEMY_API_KEY}/getNFTMetadata`
);
url.searchParams.set("contractAddress", "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d");
url.searchParams.set("tokenId", "1");

const res = await fetch(url.toString());
const json = await res.json();
```

## Output
- Metadata including name, image, attributes (if available).

## Pitfalls
- Treat metadata URLs and images as untrusted input; sanitize and proxy if needed.

## Related Files
- `data-nft-api.md`

## Official Docs
- [NFT API Endpoints](https://www.alchemy.com/docs/reference/nft-api-endpoints)
