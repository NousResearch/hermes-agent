# PR description draft: Add bsc-mint-guard to the plugin catalog

## What this plugin does

bsc-mint-guard safety-checks a BSC ERC-721 mint before the user mints. Three tools, all read-only:

- `mint_check`: contract exists, name, symbol, total supply, mint price, mint open state, max per wallet, owner, contract balance. Verdict is SAFE_TO_PREVIEW, DO_NOT_MINT, or NEEDS_MANUAL_CHECK.
- `mint_quote`: price times quantity, with gas noted as an estimate only.
- `wallet_mint_status`: balanceOf for a wallet, and whether it already minted.

A bundled skill documents the safe-mint checklist: verify the contract, check the price, preview clean, no allowance change, verify the receipt.

## Disclosures

- Network calls: public BSC JSON-RPC endpoints only (bsc.meowrpc.com, bsc-mainnet.public.blastapi.io, rpc-bsc.48.club, bsc-dataseed.bnbchain.org). No API key. Read-only eth_call and eth_getCode. No funds moved, no signing, no keys.
- No shell commands, no background processes, no stored credentials, no telemetry.
- Dependencies: Python standard library only.
- Tested: 22 unit tests pass, no live network in tests. Live read-only check 2026-11-11 08:02 WIB on JPEG.ai 0xC24BE89bd723547546ABC86e6B9fdD176C21D5e2 returned name JPEG.ai, symbol JAI, supply 5000, mintOpen true, price unknown, verdict NEEDS_MANUAL_CHECK.

## Hackathon entry

Hermes Agent Plugin Hackathon, Nous Research. Demo video and Typeform links to be added after the PR opens.
