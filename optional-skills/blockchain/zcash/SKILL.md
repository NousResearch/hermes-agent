---
name: zcash
description: Create and verify Zcash attestation receipts.
version: 1.4.0
author: Zk-nd3r, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Zcash, Blockchain, Privacy, ZEC, MCP, ZAP1, Attestation, Receipt]
    category: blockchain
    related_skills: [solana]
---

# Zcash Attestation Skill

Connect Hermes to the published `@frontiercompute/zcash-mcp` server for
public Zcash context and verifiable ZAP1 receipts. This is an attestation
skill, not a wallet, signer, balance scanner, or transaction broadcaster.

## When to Use

- Create a bounded receipt for an agent, evaluation, operator, payment, or
  external-action event.
- Verify a receipt, Merkle inclusion proof, proof bundle, or receipt chain.
- Inspect anchor status or public Zcash context needed to interpret a receipt.
- Decode a public Zcash memo, including a ZAP1 typed memo.
- Convert external evidence into a hash-only receipt request without exposing
  the underlying prompt, transcript, secret, or payment credential.

## Prerequisites

- Node.js 18 or newer with `npx` available.
- Hermes MCP support enabled through `~/.hermes/config.yaml`.
- No wallet seed, private key, PCZT, raw prompt, transcript, card data, or
  credential may be supplied to this MCP server.

Add the server using Hermes' native YAML contract:

```yaml
mcp_servers:
  zcash:
    command: "npx"
    args: ["-y", "@frontiercompute/zcash-mcp@1.4.0"]
```

Restart Hermes after changing `config.yaml` so it can discover the MCP tools.
Review the package version deliberately before changing the pin.

## How to Run

Use the discovered `zcash` MCP tools directly from Hermes. Do not invoke the
package with invented shell flags; it is a stdio MCP server.

Start with `zcash_capability_manifest` to confirm the active boundary. Then
call only the receipt, proof, anchor, memo, or public-chain tool required for
the user's request.

## Quick Reference

| Tool | Purpose |
| --- | --- |
| `zcash_capability_manifest` | Confirm supported and excluded capabilities. |
| `zcash_receipt_template` | Select a bounded receipt workflow. |
| `attest_event` | Create a typed ZAP1 event leaf. |
| `zap1_agent_eval_verdict_request` | Hash-bound an external evaluation result. |
| `zap1_attest_external_action` | Hash-bound an external action result. |
| `zap1_wallet_receipt_request` | Receipt a wallet result without taking custody. |
| `get_anchor_status` | Inspect pending and anchored state. |
| `get_anchor_history` | Inspect published ZAP1 anchors. |
| `zap1_prove_receipt` | Fetch a portable proof bundle. |
| `verify_proof` | Verify Merkle inclusion. |
| `zap1_finalize_external_receipt` | Assemble a final external receipt packet. |
| `zap1_verify_external_receipt` | Validate an external receipt packet. |
| `zap1_verify_receipt_chain` | Validate an ordered receipt sequence. |
| `zap1_compare_receipt_claims` | Compare bounded claims and evidence hashes. |
| `zap1_audit_event_log` | Replay a small sequence against an event policy. |
| `zap1_verify_evm` | Verify against the public EVM verifier. |
| `decode_memo` | Decode a public memo payload. |
| `get_block_height` | Read public Zebra chain height. |
| `lookup_transaction` | Read public transaction context by txid. |

## Procedure

1. **Confirm capability.** Call `zcash_capability_manifest`. Continue only if
   the requested operation is explicitly supported; otherwise state the
   boundary and stop.
2. **Bound the claim.** Use `zcash_receipt_template` and identify the exact
   subject, event, evidence hashes, and receipt type. Remove raw sensitive
   material before any MCP call.
3. **Create the request.** Use `attest_event` or the matching specialized
   request builder. Check that the returned claim describes only the event
   actually observed.
4. **Check anchor state.** Call `get_anchor_status`. Treat pending or unanchored
   leaves as pending; never describe them as final settlement evidence.
5. **Prove and verify.** Fetch with `zap1_prove_receipt`, then verify with
   `verify_proof` or the appropriate receipt verifier. Completion requires a
   matching leaf hash, root, proof result, and anchor context.
6. **Report narrowly.** State what was committed and verified, the anchor
   status, and what the receipt does not prove about external correctness.

## Pitfalls

- A receipt proves a bounded commitment and verification result; it does not
  prove that an external service, evaluator, wallet, or payment rail behaved
  correctly.
- An intent, quote, route, memo, payment URI, or unanchored leaf is not final
  settlement evidence.
- Public transaction context is not wallet state. Use a wallet-layer tool for
  balances, synchronization, signing, PCZTs, spends, or broadcasting.
- Do not send raw prompts, transcripts, secrets, private keys, seeds, wallet
  scan state, card data, or credentials to ZAP1.
- Do not replace `mcp_servers` with a generic camel-case JSON key; Hermes reads
  its MCP configuration from `config.yaml`.

## Verification

1. Confirm `~/.hermes/config.yaml` contains `mcp_servers.zcash` with the pinned
   `npx` command and package argument shown above.
2. Restart Hermes and confirm `zcash_capability_manifest` is discoverable.
3. Call the manifest and verify wallet custody, signing, balance scanning,
   shielded-spend construction, and broadcasting remain out of scope.
4. Create a non-sensitive test receipt request, inspect its anchor status, and
   verify its proof only when a proof bundle is available.
5. Report pending status honestly if the test leaf is not anchored.
