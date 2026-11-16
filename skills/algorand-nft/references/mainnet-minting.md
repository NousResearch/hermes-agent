# ARC-69 MainNet creation and recovery

Read workspace/README.md for current controls and workspace/VERIFICATION.md for local evidence. Local verification is not a real mint.

## Transaction profile

Create one ASA: total 1, decimals 0, default frozen false. Embed compact UTF-8 ARC-69 JSON directly in the creation acfg transaction note (this workspace caps notes at 1024 bytes; larger fee-bearing notes in newer protocols are outside scope). Require standard="arc69". Description, external_url, media_url, properties object, and mime_type are optional. Name/unit are ASA parameters. The asset URL points directly to media; omit assetMetadataHash entirely. No metadata.json upload, #arc3, SHA-256 commitment, remote metadata comparison, or attestations.

Validate name <=32 bytes, unit <=8 bytes, asset URL <=96 bytes, and note <=1024 bytes in UTF-8. Syntax validation is not media availability verification. The JSON note is public on-chain; media remains off-chain. ARC-69 is currently marked superseded by ARC-89 in its normative source; this workflow intentionally implements requested ARC-69.

## Network and account

Pera chain ID 416001. Pin node https://mainnet-api.algonode.cloud and verify BOTH genesis ID mainnet-v1.0 and hash wGHE2Pwdvd7S12BL5FaOP20EGYesN73ktiC1qzkkit8=. An approved wallet connection alone is not chain verification. Support unrekeyed ordinary single-Ed25519 accounts only; reject foreign auth addresses, multisig, LogicSig, post-quantum signatures, unexpected groups/rekey/lease/payment/transfer/close fields.

Read amount/minBalance using exact integers. Spendable balance must cover the actual fee plus 100000 microALGOs additional asset minimum balance. The latter is a balance commitment, not a transaction fee. Use current suggested parameters, bounded validity and a 10000-microALGO fee cap; do not silently raise the cap.

## Explicit button actions, no attestation checkboxes

1. Prepare/review reads node parameters/account, constructs unsigned bytes and displays sender, name/unit, direct media URL, note JSON/bytes, roles, supply/decimals, fee, minimum-balance commitment, rounds and transaction ID. No media or metadata-gateway lookup.
2. Sign reviewed mint calls Pera for exactly the reviewed transaction. It must not broadcast. Decode wallet output, require exact unsigned payload/txid equality and ordinary signature, and cryptographically verify Ed25519 over txn.bytesToSign() (TX domain, not SDK verifyBytes MX domain). No media/metadata SHA-256 verification.
3. Broadcast once is a separate explicit spending action. Recheck network/rounds/account/balance and unchanged reviewed/signed context. Persist and read back a public recovery record before the single POST. Do not store signed bytes or secrets. Explain recovery storage without adding a checkbox gate.
4. Check confirmation explicitly queries the exact saved transaction. No polling or automatic reconnect/query on reload. Require a confirmed round, nonzero asset ID, empty pool error, matching exact creation including note, and matching asset parameters. Absent metadata hash may be returned as zero-filled SDK bytes; interpret protocol absence correctly without accepting a nonzero hash.

Edits/account changes/disconnect invalidate unsigned review and signatures, never a submitted recovery receipt. Review freshness/validity failures require prepare again. Wallet rejection is not success.

## Roles

Immutable: no manager/reserve/freeze/clawback; no manager update or destruction, so asset minimum balance cannot be reclaimed by destroying it. Creator-manager: only manager=creator, permitting later control-address changes and ARC-69 metadata note updates with an additional authorized acfg. Later management is not implemented. Neither policy can rewrite immutable ASA fields or historical notes.

## Once-only recovery

Store public canonical unsigned bytes, expected txid, network and review before POST. Validate restored unsigned bytes and recompute txid; never trust a cached 'confirmed' string. Detect pre-migration ARC-3 pending/corrupt receipts and block replacement mints for investigation, not silent deletion.

After POST, timeouts, bad responses or mismatched IDs mean OUTCOME UNKNOWN. Never resend/re-sign/re-mint automatically. Preserve txid and query explicitly. Missing history, expired rounds or pool errors alone must not unlock another mint. Only fresh verified confirmation allows a deliberate Start another NFT button. Local storage is not a global uniqueness guarantee; use one tab and retain receipts.

Report accurately: local tests passed; prepared unsigned; signed/not submitted; submitted/unconfirmed; outcome unknown; or confirmed with asset ID/txid/round. Never fabricate wallet compatibility, funds, uploads or confirmations.

## Sources

https://raw.githubusercontent.com/algorandfoundation/ARCs/main/ARCs/arc-0069.md
https://dev.algorand.co/concepts/transactions/reference/
https://dev.algorand.co/concepts/assets/overview/
https://dev.algorand.co/concepts/assets/asset-operations/
https://docs.perawallet.app/references/pera-connect/
