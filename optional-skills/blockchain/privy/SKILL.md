---
name: privy-wallet
description: "Read-only Privy app user and wallet inspection."
version: 0.1.0
author: Patrick (@leprincep35700), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: PRIVY_APP_ID
    prompt: "Privy application ID"
    help: "Available in the Privy dashboard; credentials may authorize write operations elsewhere"
  - name: PRIVY_APP_SECRET
    prompt: "Privy application secret"
    help: "Privileged server credential; may authorize write operations elsewhere; never expose it"
metadata:
  hermes:
    tags: [Privy, Wallet, Web3, Blockchain, Embedded Wallets, Read-only]
    category: blockchain
    related_skills: [evm, solana]
    requires_toolsets: [terminal]
---

# Privy Wallet Skill

Read-only Privy application user and wallet inspection for Hermes. This skill
helps an administrator understand which wallets are linked to a Privy user without exposing
private keys, raw identity data, or transaction-signing capabilities.

This skill is intentionally **read-only**. It **does not sign** messages,
prepare transactions, send transactions, export keys, or bypass Privy's user
approval flows. Any future signing or transaction capability should be built as
a separate, human-approved flow with simulation, policy limits, and explicit
confirmation.

"Read-only" describes this helper's deliberately limited command surface. The
Privy credentials themselves are not read-only and may authorize privileged
application operations elsewhere. Store them as server secrets, never expose
them to prompts or client-side code, and rotate them if disclosure is suspected.

---

## When to Use

- User asks which wallet(s) are linked to a Privy account.
- User wants to inspect embedded or external wallet addresses for a Privy user.
- User wants a safe identity-to-wallet bridge before using EVM/Solana skills.
- User is designing a wallet-aware Hermes workflow and needs read-only context.

Do **not** use this skill to sign messages, approve transactions, or move funds.

---

## Prerequisites

The helper script uses only Python standard library modules.

Privy server credentials are required:

```bash
export PRIVY_APP_ID="your-privy-app-id"
export PRIVY_APP_SECRET="your-privy-app-secret"
```

Keep `PRIVY_APP_SECRET` out of prompts, chat transcripts, and committed files.

---

## How to Run

Use the `terminal` tool for every command. The helper performs only read-only
Privy API requests and prints PII-minimized JSON.

### Quick Reference

Helper script path after installation:

```text
~/.hermes/skills/blockchain/privy/scripts/privy_client.py
```

Commands:

```bash
python3 privy_client.py user    <did:privy:...>
python3 privy_client.py wallets <did:privy:...>
python3 privy_client.py users   [--limit N] [--cursor CURSOR] [--email EMAIL]
```

Output is JSON and intentionally PII-minimized. Email addresses and phone
numbers are not included in summaries; linked-account types and wallet metadata
are retained.

---

## Procedure

### 1. Check setup

```bash
python3 ~/.hermes/skills/blockchain/privy/scripts/privy_client.py users --limit 1
```

If credentials are missing, the helper exits with a clear message naming
`PRIVY_APP_ID` and/or `PRIVY_APP_SECRET`.

### 2. Inspect a Privy user safely

```bash
python3 ~/.hermes/skills/blockchain/privy/scripts/privy_client.py \
  user did:privy:example
```

The result includes:

- Privy user id
- creation timestamp if returned by Privy
- linked account types, e.g. `email`, `wallet`, `phone`
- wallet linked accounts with address, chain type, wallet client type,
  connector type, and verification timestamp

It does not include raw email addresses or phone numbers.

### 3. List wallets only

```bash
python3 ~/.hermes/skills/blockchain/privy/scripts/privy_client.py \
  wallets did:privy:example
```

Use the resulting EVM or Solana addresses with the `evm` or `solana` optional
skills for chain-specific portfolio, token, or transaction inspection.

### 4. Search/list users

```bash
python3 ~/.hermes/skills/blockchain/privy/scripts/privy_client.py users --limit 20
python3 ~/.hermes/skills/blockchain/privy/scripts/privy_client.py users --email alice@example.com
```

The `--email` filter uses Privy's documented `POST /v1/users/email/address`
endpoint with an `{"address": "..."}` JSON body. Returned summaries stay
redacted, and the address never appears in a request URL or error output.

---

## Safety Model

This skill is safe by default because it only performs server-side read calls.
For any future write/signing extension, require all of the following before
implementation:

1. human-approved confirmation in the active platform;
2. transaction or message simulation/preview;
3. explicit chain and contract allowlists;
4. amount/value limits;
5. no private-key or app-secret exposure to the LLM prompt;
6. audit-friendly logs that redact tokens and personal data.

Recommended future shape: keep write operations in a separate MCP server or
separate toolset so deployments can enable read-only Privy inspection without
enabling signing.

---

## Pitfalls

- **Server credentials are sensitive** — `PRIVY_APP_SECRET` is an app-level
  secret. Store it in environment/config secret storage only.
- **Wallet ownership can change** — always fetch fresh data before using a
  wallet address for authorization decisions.
- **Read-only is not authorization** — seeing a wallet linked to a user does
  not by itself grant permission to spend or act on behalf of that wallet.
- **Privy API shapes can evolve** — the helper preserves a small stable summary
  instead of passing raw API payloads into the agent transcript.
- **No signing path** — this skill deliberately avoids signing and transactions.
  Add those only with human-approved policy controls.

---

## Verification

Run the targeted tests from the Hermes repo root:

```bash
python -m pytest tests/skills/test_privy_wallet_skill.py -q -o addopts=''
```
