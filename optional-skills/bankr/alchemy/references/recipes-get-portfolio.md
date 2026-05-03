---
id: references/recipes-get-portfolio.md
name: 'Recipe: Get Full Portfolio'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-portfolio-apis.md
  - data-nft-api.md
  - data-prices-api.md
updated: 2026-02-05
---
# Recipe: Get Full Portfolio

## Goal
Build a full wallet portfolio by combining token balances, NFT ownership, and optional pricing.

## Inputs
- `address`
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Fetch token balances via Portfolio API or Token API.
2. Fetch token metadata for symbols/decimals.
3. Fetch NFTs via NFT API.
4. Optionally fetch prices to compute USD values.

## Example (Token Balances)
```bash
curl -s "https://api.g.alchemy.com/data/v1/$ALCHEMY_API_KEY/assets/tokens/balances/by-address?address=0x00000000219ab540356cbb839cbe05303d7705fa"
```

## Example (NFTs)
```bash
curl -s "https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTsForOwner?owner=0x00000000219ab540356cbb839cbe05303d7705fa"
```

## Example (Prices)
```bash
curl -s "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/by-symbol?symbols=ETH&symbols=USDC"
```

## Output
- Combined portfolio object with tokens, NFTs, and optional pricing.

## Pitfalls
- Normalize token decimals before valuation.
- Cache metadata and price data.
- NFT metadata can be spam or malformed; allowlist trusted collections where needed.

## Related Files
- `data-portfolio-apis.md`
- `data-nft-api.md`
- `data-prices-api.md`

## Official Docs
- [Portfolio APIs](https://www.alchemy.com/docs/docs/reference/portfolio-apis)
