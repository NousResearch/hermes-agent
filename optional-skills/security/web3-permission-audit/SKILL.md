---
name: web3-permission-audit
description: Read-only cross-chain wallet permission audit for Solana SPL delegates, XRPL trust lines, and targeted Base/EVM ERC-20 allowances. Surfaces revocable spend rights, issuer-control risk, rippling exposure, freeze authorities, and explicit coverage limits. No private keys or signing.
version: 0.1.0
author: Osraka, with Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Security, Web3, Wallets, Permissions, Solana, XRPL, Base, EVM, Privacy]
    related_skills: [blockchain/solana, blockchain/base]
---

# Web3 Permission Audit Skill

Read-only cross-chain wallet permission audit for high-signal Web3 safety reviews.
The skill answers a narrow but important question:

> Who, besides the wallet owner, can affect this wallet's assets or asset movement?

It focuses on practical permission surfaces that users often forget:

- Solana SPL Token and Token-2022 delegates that can transfer tokens from token accounts
- Solana external close authorities, frozen token accounts, and active mint/freeze authorities
- XRPL trust line limits, freeze flags, rippling flags, negative balances, and issuer exposure
- Base/EVM ERC-20 allowances for explicit owner/spender/token sets, including a Permit2 alias

The helper script uses only Python standard library modules. It never asks for private keys,
never signs transactions, and never submits state-changing RPC calls.

---

## When to Use

Use this skill when the user asks to:

- Audit a wallet before using a new dapp, bridge, marketplace, or agent workflow
- Review whether old approvals or delegates should be revoked
- Understand cross-chain wallet risk without sharing private keys
- Compare EVM allowances with Solana delegate risk and XRPL trust-line risk
- Produce a structured security report for a wallet or incident triage
- Explain why an asset can be frozen, minted, rippled, or transferred by another principal

Do not use this skill to:

- Sign or submit revoke transactions
- Claim full EVM approval discovery without an indexer or log scan
- Give financial advice about whether to hold a token
- Treat issuer-control findings as proof of malicious behavior without context

---

## Safety Model

This skill is intentionally defensive.

| Property | Behavior |
|---|---|
| Private keys | Never requested |
| Signing | Never performed |
| Transaction submission | Never performed |
| RPC operations | Read-only calls only |
| Output | JSON report with findings, evidence, limits, and suggested actions |
| Privacy caveat | Querying public RPC providers reveals the queried addresses to those providers |

The script validates RPC URLs and rejects localhost/private IP RPC endpoints by default.
Use `--allow-private-rpc` or `WEB3_AUDIT_ALLOW_PRIVATE_RPC=1` only when intentionally querying
a trusted local/private node.

---

## Quick Reference

Helper script path after install:

```bash
~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py
```

Commands:

```bash
python3 web3_permission_audit.py explain
python3 web3_permission_audit.py solana <SOLANA_ADDRESS>
python3 web3_permission_audit.py xrpl <XRPL_CLASSIC_ADDRESS>
python3 web3_permission_audit.py evm-allowance --owner <0xOWNER> --spender <0xSPENDER>
python3 web3_permission_audit.py evm-allowance --owner <0xOWNER> --spender-alias permit2
python3 web3_permission_audit.py audit --solana <SOLANA_ADDRESS> --xrpl <XRPL_ADDRESS> --evm-owner <0xOWNER> --evm-spender-alias permit2
```

RPC environment variables:

```bash
export WEB3_AUDIT_SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export WEB3_AUDIT_XRPL_RPC_URL="https://s1.ripple.com:51234/"
export WEB3_AUDIT_BASE_RPC_URL="https://mainnet.base.org"
```

The script also falls back to `SOLANA_RPC_URL`, `XRPL_RPC_URL`, and `BASE_RPC_URL` for compatibility
with existing blockchain skills.

---

## Procedure

### 1. Explain the Risk Model

Start here when preparing a review or PR description.

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py explain
```

The output documents severity definitions, coverage, and non-goals. This is useful when a user
expects the tool to discover everything automatically.

### 2. Audit Solana Token Permissions

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  solana 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

What it checks:

- SPL Token delegates with delegated transfer amounts
- Token-2022 delegates where `jsonParsed` RPC exposes them
- External close authorities
- Frozen token accounts
- Token mint authorities and freeze authorities for the first `--mint-limit` unique mints

Useful flags:

```bash
--rpc-url <URL>       # Override Solana RPC
--mint-limit 50      # Inspect more unique mints for authority risk
--detail-limit 100   # Limit raw token-account evidence rows in large wallets
--allow-private-rpc  # Permit localhost/private RPC targets intentionally
```

### 3. Audit XRPL Trust Lines

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  xrpl rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh
```

What it checks:

- Trust line balances and limits
- Freeze flags
- Rippling flags
- Negative balances
- Issuer/counterparty exposure
- Peer trust limits as informational context

Useful flags:

```bash
--limit 200       # Trust lines per RPC page
--max-pages 10    # Fetch more paginated trust-line pages
--rpc-url <URL>   # Override XRPL JSON-RPC endpoint
```

### 4. Audit Base/EVM ERC-20 Allowances

Plain EVM JSON-RPC cannot enumerate all approvals for an address without scanning logs or using an
indexer. This command is therefore deliberately targeted: provide an owner plus one or more spenders.
If `--token` is omitted, the tool checks a bundled common Base token set.

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  evm-allowance \
  --owner 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 \
  --spender-alias permit2
```

Check a specific spender and token:

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  evm-allowance \
  --owner 0xOWNER \
  --spender 0xSPENDER \
  --token 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913
```

Severity mapping:

- `critical`: unlimited approval on a funded token
- `high`: approval can cover the current balance
- `medium`: non-zero bounded approval
- `info`: contextual findings only

### 5. Run a Cross-Chain Report

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  audit \
  --solana <SOLANA_ADDRESS> \
  --xrpl <XRPL_CLASSIC_ADDRESS> \
  --evm-owner <0xOWNER> \
  --evm-spender-alias permit2
```

The report includes:

- `summary`: finding counts and severity counts
- `findings`: sorted actionable findings with evidence and suggested actions
- `chain_reports`: raw chain-specific evidence used to make the findings
- `limitations`: explicit coverage boundaries
- `safety_model`: confirmation that the tool is read-only and non-signing

---

## Interpreting Results

A finding is not automatically a vulnerability. Many permissions are legitimate while a dapp is in use.
The right question is whether the permission is:

- Still needed
- Granted to the expected principal
- Bounded to the expected asset and amount
- Backed by an issuer or mint authority the user understands
- Safe to leave active for future balances

Suggested review flow:

1. Treat `critical` and `high` findings as revoke-or-confirm items.
2. Treat `medium` findings as context-dependent items that need user confirmation.
3. Treat `low` issuer-control findings as portfolio hygiene, not proof of compromise.
4. Keep the JSON output as evidence if opening a security issue or support ticket.

---

## Limitations

This skill avoids false certainty. Important limits:

- EVM full approval discovery requires an indexer, archive log scan, or wallet-provider API.
- ERC-721, ERC-1155, Permit signatures, account abstraction session keys, and app-specific delegated
  permissions are not discovered in the first version.
- Solana compressed NFTs, DeFi positions, multisig owners, durable nonce authorities, and arbitrary
  program authorities are out of scope.
- XRPL trust lines are not ERC-20 allowances. The XRPL scan focuses on trust limits, rippling, freezes,
  issuer exposure, and balances.
- Public RPC calls can leak address interest to the RPC provider. Use a trusted private RPC when privacy
  requirements are strict.

---

## Verification

Syntax and local smoke tests:

```bash
python3 -m py_compile ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py --help
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py explain
```

Live RPC smoke tests are optional and depend on public endpoint rate limits:

```bash
python3 ~/.hermes/skills/security/web3-permission-audit/scripts/web3_permission_audit.py \
  evm-allowance --owner 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 --spender-alias permit2
```
