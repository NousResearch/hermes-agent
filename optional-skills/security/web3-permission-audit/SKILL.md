---
name: web3-permission-audit
description: Instruction-first cross-chain wallet permission audit playbook for Solana, XRPL, and Base/EVM. Guides read-only review of delegates, trust lines, allowances, freeze controls, issuer exposure, and explicit coverage limits without bundling executable code.
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

Instruction-first playbook for read-only cross-chain wallet permission reviews.
The skill answers one narrow question:

> Who, besides the wallet owner, can affect this wallet's assets or asset movement?

This is intentionally shipped as a Markdown-only optional skill. Per the Hermes contribution guide,
new skills should prefer instructions, shell commands, and existing tools before adding executable
repo code. If a reusable parser is needed later, add it as a focused follow-up after the workflow is
accepted and tested.

---

## When to Use

Use this skill when the user asks to:

- Audit a wallet before connecting to a dapp, bridge, marketplace, or agent workflow
- Review whether old approvals, delegates, or trust relationships should be revoked
- Compare EVM allowances with Solana delegate risk and XRPL trust-line risk
- Produce a structured wallet-safety report for incident triage or user education
- Explain why an asset can be frozen, minted, rippled, or transferred by another principal

Do not use this skill to:

- Ask for private keys, seed phrases, or signing authority
- Submit revoke transactions automatically
- Claim complete EVM approval discovery without an indexer or historical log scan
- Treat issuer-control findings as proof of malicious behavior without context
- Give financial advice about whether to hold a token

---

## Safety Model

| Property | Behavior |
|---|---|
| Private keys | Never requested |
| Signing | Never performed |
| Transaction submission | Never performed |
| RPC operations | Read-only calls only |
| Output | Findings, evidence, caveats, and suggested user-confirmed actions |
| Privacy caveat | Public RPC providers can learn which addresses were queried |

If privacy is strict, ask the user whether they have a trusted private RPC endpoint. Never use a
private or localhost RPC URL supplied by an untrusted third party.

---

## Quick Reference

Install related official optional skills when useful:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/blockchain/base
```

Default public RPCs for read-only checks:

```bash
export SOLANA_RPC_URL="${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}"
export XRPL_RPC_URL="${XRPL_RPC_URL:-https://s1.ripple.com:51234/}"
export BASE_RPC_URL="${BASE_RPC_URL:-https://mainnet.base.org}"
```

Core audit surfaces:

| Chain | Permission surface | What to look for |
|---|---|---|
| Solana | SPL / Token-2022 delegates | `delegate`, `delegatedAmount`, external `closeAuthority`, `state=frozen` |
| Solana | Mint controls | `mintAuthority`, `freezeAuthority` on token mints |
| XRPL | Trust lines | `limit`, `balance`, `no_ripple`, `freeze`, `freeze_peer`, negative balances |
| Base/EVM | ERC-20 allowances | `allowance(owner, spender)`, especially unlimited or balance-covering approvals |

---

## Procedure

### 1. Scope the Audit

Collect only public identifiers:

- Solana wallet address, if relevant
- XRPL classic account address, if relevant
- Base/EVM owner address, if relevant
- EVM spender address or known spender alias, if the user wants allowance checks
- Specific token contracts or mints, if the user wants targeted checks

Confirm the audit is read-only and that no private key or seed phrase is needed.

### 2. Use a Consistent Severity Model

| Severity | Meaning |
|---|---|
| `critical` | A third party can transfer current assets or has effectively unlimited approval on funded assets |
| `high` | A third party can transfer assets, or the account/asset is frozen |
| `medium` | Permission or issuer-control surface can affect assets but needs user context |
| `low` | Issuer/counterparty exposure or supply-control fact worth reviewing |
| `info` | Context that explains the permission graph but is not directly actionable |

### 3. Solana Review

Fetch SPL Token and Token-2022 accounts with `jsonParsed` encoding. Use both token program IDs:

```bash
curl -s "$SOLANA_RPC_URL" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getTokenAccountsByOwner","params":["SOLANA_WALLET",{"programId":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},{"encoding":"jsonParsed"}]}'

curl -s "$SOLANA_RPC_URL" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getTokenAccountsByOwner","params":["SOLANA_WALLET",{"programId":"TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"},{"encoding":"jsonParsed"}]}'
```

Review each parsed token account:

- `delegate` present with non-zero `delegatedAmount`: `high` or `critical`
- `closeAuthority` present and not the wallet owner: `medium`
- `state` is `frozen`: `high`
- Very large numbers of token accounts: summarize counts and sample evidence instead of flooding output

For material mints, fetch mint account info:

```bash
curl -s "$SOLANA_RPC_URL" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getAccountInfo","params":["TOKEN_MINT",{"encoding":"jsonParsed"}]}'
```

Flag mint controls:

- `freezeAuthority` present: `medium`
- `mintAuthority` present: `low` or `medium`, depending on asset context

### 4. XRPL Review

Fetch trust lines from the validated ledger:

```bash
curl -s "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  -d '{"method":"account_lines","params":[{"account":"XRPL_ACCOUNT","ledger_index":"validated","limit":200}]}'
```

Review each line:

- `freeze` or `freeze_peer`: `high`
- Negative `balance`: `medium`
- Positive issued-asset `balance`: `low` issuer/counterparty exposure
- `limit` greater than zero and `no_ripple` not set: `medium` for ordinary user wallets
- Large or unexpected `limit_peer`: `info`, unless it explains another finding

Important: XRPL trust lines are not ERC-20 approvals. Do not describe them as spend allowances.
They represent bilateral credit/issuer relationships with different semantics.

### 5. Base/EVM Targeted Allowance Review

Plain EVM JSON-RPC cannot enumerate every approval for an address without an indexer or log scan.
Only check explicit owner/spender/token sets unless the user provides an indexer export.

Common spender alias:

```text
permit2 = 0x000000000022d473030f116ddee9f6b43ac78ba3
```

Use `eth_call` for ERC-20 `allowance(address,address)`:

```bash
python3 - <<'PY'
import json
import os
import urllib.request

rpc = os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")
owner = "0xOWNER".lower().replace("0x", "").zfill(64)
spender = "0xSPENDER".lower().replace("0x", "").zfill(64)
token = "0xTOKEN"
selector = "dd62ed3e"  # allowance(address,address)
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "eth_call",
    "params": [{"to": token, "data": "0x" + selector + owner + spender}, "latest"],
}
req = urllib.request.Request(
    rpc,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=20) as resp:
    body = json.load(resp)
print(json.dumps(body, indent=2))
PY
```

Interpretation:

- Result is `0x0` or empty-equivalent: no allowance for that token/spender pair
- Result near `2**256 - 1`: unlimited approval, `critical` if the token has current/future value
- Result greater than or equal to current token balance: `high`
- Any non-zero bounded result: `medium`

Do not say the EVM wallet is fully clean unless historical approvals were checked via a trusted
indexer, wallet provider export, or archive log scan.

### 6. Report Format

Use this structure for final answers or issue comments:

```json
{
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "info": 0
  },
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

---

## Pitfalls

- EVM approvals are event/history based; plain JSON-RPC only answers targeted allowance questions.
- Public Solana wallets can have thousands of spam token accounts. Summarize counts and show samples.
- Token mint/freeze authorities are not automatically malicious. Many legitimate tokens retain controls.
- XRPL trust lines use different semantics from EVM approvals. Keep the language chain-specific.
- Public RPC usage leaks address interest to the RPC provider.
- Never ask the user to paste a seed phrase or private key to revoke anything.

---

## Verification

Before opening or updating a PR for this skill:

```bash
git diff --check
scripts/check-windows-footguns.py
```

Manual verification checklist:

- The PR changes stay focused on the skill and docs.
- The skill is instruction-first and has no new runtime dependency.
- Commands are read-only and do not submit transactions.
- EVM limitations are explicit.
- Security-sensitive behavior is described in the PR body.
- Any live RPC example is labeled as optional and read-only.
