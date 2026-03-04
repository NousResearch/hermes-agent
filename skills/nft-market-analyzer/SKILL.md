---
name: nft-market-analyzer
description: Solana NFT wallet analytics using Helius API. Detects wash trading, mint-dump patterns, smart money signals, and computes net ROI. Use for requests like "analyze nft wallet", "check wash trading", "nft risk score", "smart money tracker", "profile nft wallet".
version: 1.0.0
author: nftpoetrist
license: MIT
metadata:
  hermes:
    tags: [NFT, Solana, Blockchain, Analytics, Risk, WashTrading]
    related_skills: []
---

# NFT Market Analyzer

Analyzes any Solana wallet and returns a three-panel risk + profiling report using the Helius Enhanced Transaction API and optional Tensor Trade API. Zero third-party dependencies — stdlib only.

## When to Use

Load this skill when the user asks to:
- Analyze an NFT wallet address
- Check wash trading signals for a wallet
- Get NFT risk score for a Solana address
- Track smart money wallet behavior
- Profile NFT trading activity
- Detect mint-dump patterns
- Calculate NFT portfolio ROI

## Quick Reference

| Task | Command |
|---|---|
| Full analysis | `python3 ~/.hermes/skills/nft-market-analyzer/scripts/nft_analytics.py --wallet <ADDR>` |
| JSON output | `python3 ~/.hermes/skills/nft-market-analyzer/scripts/nft_analytics.py --wallet <ADDR> --json` |
| Verbose mode | `python3 ~/.hermes/skills/nft-market-analyzer/scripts/nft_analytics.py --wallet <ADDR> --verbose` |

## Procedure

1. Get the Solana wallet address from the user (base-58, 32-44 chars)
2. Run the analyzer via terminal:
```bash
python3 ~/.hermes/skills/nft-market-analyzer/scripts/nft_analytics.py --wallet <WALLET_ADDRESS>
```

3. Return the three-panel report to the user with interpretation

## Output Panels

**Wallet Risk Analyzer** — Risk score (LOW/MEDIUM/HIGH), mint-dump pattern, wash trading signals, fast flips

**Smart Money Tracker** — Smart entry detection, average ROI, activity duration in months

**Wallet Profiling** — Total NFTs minted, average flip duration, high-risk collections, net ROI

## Requirements

Add to `~/.hermes/.env`:
```
HELIUS_API_KEY=your_key_here    # Required — https://helius.dev (free tier works)
TENSOR_API_KEY=your_key_here    # Optional — improves ROI accuracy
```

## Pitfalls

- **HTTP 403**: Helius API key invalid or expired. Get a new key at https://helius.dev
- **Invalid wallet**: Address must be 32-44 chars, base-58 alphabet only
- **ROI shows 0%**: Wallet may not have matching buy/sell pairs in the last 1000 transactions
- **Wash trading false positives**: Only NFT_SALE events are counted, not transfers or listings

## Verification

A successful run shows three panels ending with `======` separator line. Risk Score will be LOW, MEDIUM, or HIGH.
