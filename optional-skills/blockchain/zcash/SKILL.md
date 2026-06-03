---
name: zcash
description: Use the published zcash-mcp server for Zcash public chain context, ZAP1 attestation receipts, anchor status, and receipt verification. This is an attestation/proof skill, not a wallet or signer.
version: 1.4.0
author: Zk-nd3r
license: MIT
metadata:
  hermes:
    tags: [Zcash, Blockchain, Privacy, ZEC, Shielded, MCP, ZAP1, Attestation, Receipt]
    related_skills: [base, solana]
---

# Zcash Attestation Skill

Use this skill when an agent needs Zcash context or a verifiable ZAP1 receipt for a workflow.

Built on the published `@frontiercompute/zcash-mcp` MCP server.

ZAP1 rule: observe state, bound the claim, hash evidence, issue a receipt, verify later.

This is not a wallet skill. It does not hold keys, scan balances, sign transactions, build shielded spends, broadcast wallet transactions, or replace lightwalletd/Zaino/wallet SDKs.

## When to Use

- The user asks for Zcash public chain context needed to interpret a receipt or anchor.
- The user wants to decode a Zcash memo payload.
- The user wants to create a ZAP1 attestation leaf for an agent/workflow event.
- The user wants a receipt template for an agent action, agent eval verdict, wallet action, external action, payment, operator event, or policy attestation.
- The user wants to fetch or verify a ZAP1 proof bundle, Merkle inclusion proof, anchor status, or EVM verifier result.
- The user wants to convert an external eval/action result into a hash-only receipt request without exposing raw prompts, transcripts, secrets, or payment credentials.

## Setup

Requires Node.js 18+.

```bash
npm install -g @frontiercompute/zcash-mcp
```

Or add the stdio MCP server to your MCP config:

```json
{
  "mcpServers": {
    "zcash": {
      "command": "npx",
      "args": ["@frontiercompute/zcash-mcp"]
    }
  }
}
```

The package runs as a stdio MCP server. Agents should call tools through their MCP client, not as direct shell subcommands.

## Core Tools

| Tool | Use |
|------|-----|
| `zcash_capability_manifest` | Check the supported boundary before composing with wallet-layer tools. |
| `zcash_receipt_template` | Get a customer-ready workflow for a receipt type. |
| `attest_event` | Create a typed ZAP1 event leaf. |
| `get_anchor_status` | Check current Merkle root, pending leaves, and anchor state. |
| `get_anchor_history` | Inspect published ZAP1 anchors. |
| `verify_proof` | Verify Merkle inclusion for a ZAP1 leaf. |
| `zap1_prove_receipt` | Fetch a proof bundle for a leaf hash. |
| `zap1_finalize_external_receipt` | Assemble a final receipt packet from a receipt request and proof bundle. |
| `zap1_verify_external_receipt` | Validate an external-action receipt packet without trusting the external rail. |
| `zap1_agent_eval_verdict_request` | Convert an external eval verdict into a hash-only receipt request. |
| `zap1_attest_external_action` | Convert an external rail action into a hash-only receipt request. |
| `zap1_wallet_receipt_request` | Convert a wallet-layer action result into hashes; wallet custody stays outside ZAP1. |
| `zap1_extract_proof_artifact` | Extract portable proof fields from a receipt. |
| `zap1_verify_receipt_chain` | Validate a sequence of receipt packets. |
| `zap1_compare_receipt_claims` | Compare subject, claim, evidence, type, and anchor context across receipts. |
| `zap1_audit_event_log` | Replay a small receipt sequence against an expected event policy. |
| `zap1_verify_evm` | Verify a proof against the deployed EVM verifier. |
| `decode_memo` | Decode Zcash memo payloads, including ZAP1 typed memos. |
| `get_block_height` | Read current public Zcash height from Zebra. |
| `lookup_transaction` | Read public transaction context by txid. |

## Safety Rules

- Do not send private keys, seeds, PCZTs, wallet scan state, raw prompts, transcripts, secrets, card data, or credentials to ZAP1.
- Do not describe an unanchored leaf as final settlement evidence.
- Do not treat a quote, intent, route, memo, or payment URI as proof by itself.
- Use wallet-layer tools for balance, signing, sync, and spend construction; use ZAP1 before or after those actions to produce receipts.
- State the claim narrowly: a receipt proves a bounded event/claim was committed and can be verified; it is not proof that an external system behaved correctly.

## Links

- npm: https://www.npmjs.com/package/@frontiercompute/zcash-mcp
- GitHub: https://github.com/Frontier-Compute/zcash-mcp
- ZAP1 proof rail docs: https://github.com/Frontier-Compute/zcash-mcp/blob/main/docs/zap1-proof-rail.md
- Verify page: https://verify.frontiercompute.cash
- API health: https://api.frontiercompute.cash/health
