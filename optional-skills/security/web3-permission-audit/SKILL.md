---
name: web3-permission-audit
description: Read-only cross-chain wallet permission audit.
version: 0.1.0
author: Ahmet Osrak (Osraka), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Security, Web3, Wallets, Permissions, Solana, XRPL, EVM, Privacy]
    category: security
    related_skills: [blockchain/solana, blockchain/evm]
    requires_toolsets: [terminal]
---

# Web3 Permission Audit Skill

This skill guides read-only reviews of wallet permissions across Solana, XRPL,
and EVM-compatible networks. It produces evidence, severity, limitations, and
user-confirmed next steps; it never requests secrets, signs, or submits a transaction.

## When to Use

Use this skill when the user wants to:

- Audit a wallet before connecting it to a dapp, bridge, marketplace, or agent workflow.
- Review old token approvals, Solana delegates, or XRPL trust relationships.
- Compare EVM allowance risk with Solana delegate and XRPL issuer exposure.
- Produce a structured permission report for incident triage or user education.
- Understand whether another principal can freeze, mint, close, or transfer assets.

Do not use this skill to:

- Request private keys, seed phrases, wallet files, or signing authority.
- Submit revoke, transfer, trust-line, or other transactions automatically.
- Claim complete EVM approval discovery without an indexer or historical log scan.
- Treat issuer controls or an approval alone as proof of malicious behavior.
- Give financial advice about whether to hold or sell an asset.

## Prerequisites

- Use the Hermes `terminal` tool for all RPC requests and helper commands below.
- Ask only for public identifiers: addresses, token mints, contract addresses, or transaction hashes.
- Install the related official skills only when their chain-specific helpers are useful:
  `hermes skills install official/blockchain/solana` and
  `hermes skills install official/blockchain/evm`.
- The EVM skill supersedes the standalone Base skill. Use its `--chain base` option for Base
  checks and `EVM_RPC_URL` for a custom EVM endpoint.
- Public RPC providers can rate-limit requests and observe queried addresses. Ask the user to
  confirm a trusted endpoint for sensitive investigations; never request credentials for it.

## How to Run

1. Ask which networks are in scope and collect only the relevant public wallet addresses.
2. Ask for token mints, contracts, or spender addresses when the user wants a targeted review.
3. Validate addresses locally before putting them in a request. Treat uncertain input as data,
   never as shell syntax.
4. Set the endpoint variables in the Hermes `terminal` environment:

   ```bash
   export SOLANA_RPC_URL="${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}"
   export XRPL_RPC_URL="${XRPL_RPC_URL:-https://s1.ripple.com:51234/}"
   export EVM_RPC_URL="${EVM_RPC_URL:-https://mainnet.base.org}"
   ```

5. Establish network context before interpreting account data. Record the endpoint, request time,
   Solana slot, XRPL validated ledger, or EVM chain and block used.
6. Run only the read-only procedure for each selected network. Keep targeted and exhaustive EVM
   checks separate in the report.
7. Return findings with evidence, severity, limitations, and an action that requires user approval.

## Quick Reference

| Network | Permission surface | Evidence to review |
| --- | --- | --- |
| Solana | SPL and Token-2022 accounts | `delegate`, `delegatedAmount`, `closeAuthority`, `state` |
| Solana | Token mint controls | `mintAuthority`, `freezeAuthority` |
| XRPL | Trust lines | `limit`, `balance`, `no_ripple`, `freeze`, `freeze_peer` |
| Base/EVM | ERC-20 allowances | `allowance(owner, spender)`, approval scope, token balance |

Use the related EVM skill for the known-spender checker through `terminal`:

```bash
SCRIPT="$HOME/.hermes/skills/blockchain/evm/scripts/evm_client.py"
python3 "$SCRIPT" allowance "0xOWNER" --chain base
```

That helper checks a known token and spender set, not every approval ever granted. For a specific
token and spender, use a direct read-only `eth_call` and record the exact contract, block tag, and
returned allowance.

Severity guidance:

| Severity | Meaning |
| --- | --- |
| `critical` | A third party can transfer current assets or has effectively unlimited approval on funded assets. |
| `high` | A third party can transfer assets, or an account/asset is frozen. |
| `medium` | A permission or issuer-control surface can affect assets but needs user context. |
| `low` | Issuer or counterparty exposure is worth reviewing but is not directly actionable. |
| `info` | Context that explains the permission graph without a direct security impact. |

## Procedure

### 1. Scope and establish context

Confirm the networks, addresses, and requested depth. Run a harmless health query first:

- Solana: `getHealth` and, when relevant, `getTokenAccountsByOwner`.
- XRPL: `server_info` and `fee`, followed by `ledger_index: "validated"` reads.
- Base/EVM: `eth_chainId`, `eth_blockNumber`, and the EVM helper with `--chain base`.

Stop when a node is unsynchronized or returns an error. Mark later results uncertain instead of
silently substituting a different network or endpoint.

### 2. Review Solana delegates and authorities

Query both the legacy SPL Token and Token-2022 program IDs with `jsonParsed` encoding. Review each
material account for:

- A non-zero `delegate` and `delegatedAmount`: `high` or `critical`, depending on amount and scope.
- A `closeAuthority` that is not the wallet owner: `medium`.
- `state` equal to `frozen`: `high`.

For material mints, inspect `mintAuthority` and `freezeAuthority`. Report their presence as issuer
control, not automatically as abuse. If the wallet has many spam accounts, report counts and a few
representative records rather than flooding the user with unverified findings.

Example read-only requests run through `terminal`:

```bash
curl --fail-with-body --silent --show-error "$SOLANA_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"getTokenAccountsByOwner","params":["SOLANA_WALLET",{"programId":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},{"encoding":"jsonParsed"}]}'

curl --fail-with-body --silent --show-error "$SOLANA_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"getTokenAccountsByOwner","params":["SOLANA_WALLET",{"programId":"TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"},{"encoding":"jsonParsed"}]}'
```

### 3. Review XRPL trust lines

Fetch trust lines from the validated ledger and follow pagination markers until the requested scope
is complete:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"account_lines","params":[{"account":"XRPL_ACCOUNT","ledger_index":"validated","limit":200}]}'
```

Interpret the fields using XRPL semantics:

- `freeze` or `freeze_peer`: `high` operational impact.
- Negative `balance`: `medium` counterparty or credit exposure.
- Positive issued-asset balance: `low` issuer/counterparty exposure until context is known.
- A positive `limit` with `no_ripple` unset: `medium` for ordinary user wallets when it explains a
  reachable trust path.
- Unexpected `limit_peer`: `info` unless it contributes to another finding.

Trust lines are bilateral credit and issuer relationships, not ERC-20 spend allowances. Keep that
distinction in the report.

### 4. Review targeted Base/EVM allowances

Use the EVM skill’s allowance command for known token and spender coverage:

```bash
python3 "$HOME/.hermes/skills/blockchain/evm/scripts/evm_client.py" allowance "0xOWNER" --chain base
```

For a user-supplied token and spender, call ERC-20 `allowance(owner, spender)` through `eth_call`.
Interpret the returned uint256 at the selected block:

- Zero or an empty-equivalent result: no allowance for that pair at that block.
- A value near `2**256 - 1`: effectively unlimited approval; `critical` if funded or reusable.
- A value greater than or equal to the current token balance: `high` when the spender is trusted by
  neither the user nor the application context.
- A bounded non-zero value: `medium` until scope and expiry context are established.

Plain JSON-RPC cannot enumerate every approval for an address. Do not claim that an EVM wallet is
clean unless a trusted indexer, wallet export, or archive log scan covers the relevant history.

### 5. Report evidence and user-controlled actions

Use this shape for a machine-readable summary:

```json
{
  "summary": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
  "findings": [
    {
      "severity": "medium",
      "chain": "xrpl",
      "category": "trustline_rippling",
      "subject": "XRPL_ACCOUNT:USD:ISSUER",
      "evidence": "field names and values observed",
      "impact": "why this matters",
      "suggested_action": "user-confirmed next step"
    }
  ],
  "limitations": [
    "EVM approval discovery was targeted, not exhaustive",
    "No private keys, signing, or transaction submission were used"
  ]
}
```

## Pitfalls

- The EVM helper checks known tokens and spenders only; use an indexer or archive log scan for
  exhaustive historical approval discovery.
- Base is selected through the EVM skill’s `--chain base` option; do not install a separate
  standalone Base skill.
- Public Solana wallets may contain thousands of spam token accounts. Summarize and preserve
  evidence for representative records.
- Token mint/freeze authorities are not automatically malicious; report the control and its scope.
- XRPL trust lines have different semantics from EVM approvals. Use chain-specific terminology.
- Public RPC usage reveals address interest to the provider. Never send secrets or signing material.
- A read-only finding is not a revocation. Any revoke or transfer must be separately explained and
  explicitly confirmed by the user.

## Verification

Before opening or updating a PR for this skill:

```bash
git diff --check
scripts/check-windows-footguns.py --all
scripts/run_tests.sh tests/skills/test_web3_permission_audit_skill.py -q
```

Confirm that:

- The source, generated page, and optional-skills catalog describe the same read-only scope.
- The skill points at `blockchain/evm` and documents `--chain base`.
- No command asks for a seed phrase, private key, signing authority, or transaction submission.
- EVM coverage limitations and the endpoint/block context are visible in the final report.
- No live RPC call is required by the skill test.
