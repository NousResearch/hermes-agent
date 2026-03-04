---
name: nft-market-analyzer
description: Solana NFT wallet analytics using Helius API. Detects wash trading, mint-dump patterns, smart money signals, and computes net ROI. Use for requests like "analyze nft wallet", "check wash trading", "nft risk score", "smart money tracker", "profile nft wallet".
version: 2.0.0
author: nftpoetrist
license: MIT
metadata:
  hermes:
    tags: [NFT, Solana, Blockchain, Analytics, Risk, WashTrading]
    related_skills: []
---

# NFT Market Analyzer

## CRITICAL INSTRUCTIONS
- ALWAYS run the Python script directly via terminal
- NEVER interpret or summarize the output
- ALWAYS show the raw terminal output exactly as-is
- Do NOT add any commentary before or after the output

## When to Use

When user asks to analyze a Solana wallet, check wash trading, get NFT risk score, smart money tracker, profile NFT wallet, mint dump detection.

## Procedure

1. Get wallet address from user
2. Run this EXACT command via terminal:
```bash
python3 ~/.hermes/skills/nft-market-analyzer/scripts/nft_analytics.py --wallet <WALLET_ADDRESS>
```

3. Show the raw output EXACTLY as printed — do not modify, summarize, or interpret it

## Requirements

- HELIUS_API_KEY must be set in ~/.hermes/.env
- TENSOR_API_KEY optional

## Pitfalls

- If you summarize instead of showing raw output, you are doing it WRONG
- Always use the script, never make up analysis
