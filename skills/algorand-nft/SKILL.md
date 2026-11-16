---
name: algorand-nft
description: "Use for Algorand NFTs. Generate ARC-69 JSON and mint."
version: 0.4.1
author: Amy Lee
license: MIT
metadata:
  hermes:
    tags: [algorand, nft, mainnet, mint, pera, arweave, ardrive, arc69, metadata]
---

# Algorand NFT: ARC-69 on-chain metadata + Pera

## When to use

Use for generating ARC-69 JSON and minting one Algorand NFT through Pera MainNet. Read workspace/README.md for the installed controls and references/mainnet-minting.md for transaction/recovery behavior. The browser workspace is the implementation; scripts/nft.py is an obsolete TestNet-only ARC-3 helper, not this workflow. Never use that helper for MainNet or bypass its network guard.

## User workflow

1. Connect Pera explicitly and select a shared public account. MainNet chain ID is 416001.
2. Upload only the media through ArDrive's official app into a public drive, or use an existing media reference. Storage authentication/payment remains in ArDrive. A raw File Data Transaction ID is not an ArFS metadata record or share URL. Preserve literal IDs; syntax-check before use.
3. Use the ARC-69 form to generate compact JSON with required standard="arc69" and optional description, external_url, media_url, mime_type, and properties object. Show JSON and its UTF-8 byte count. Embed these exact bytes in the asset creation transaction's note. Enforce this workspace's conservative 1024-byte standard-fee note cap on encoded bytes, not JavaScript character count. Current node source also defines larger fee-bearing notes; those are outside this implementation. Do not require a metadata.json file, download, second upload, remote JSON reference, or SHA-256 check.
4. Set the on-chain asset name, unit, direct media URL, and administrative policy. Name and unit belong in ASA parameters, not required ARC-69 JSON fields. The asset URL points directly to media, not JSON and not #arc3; ARC-69 recommends #i/#v/#a/#p/#h for image/video/audio/PDF/HTML. Media stays off-chain; only JSON is on-chain.
5. Prepare/review, sign in Pera, broadcast once, and check confirmation through separate labeled actions. Buttons express their operation; do not add attestation or consent-checkbox gates. Connection or signing must never automatically broadcast. Keep the exact-transaction review and wallet approval.

## Simplification requirements

- Remove ARC-3 exact-file attestations, integrity bindings, media-file hashing, metadata hashing, upload-reference review, replacement attestations, and all extra checkboxes.
- ARC-69's media hash is optional; omit assetMetadataHash. Do not compute or verify SHA-256 for metadata or media. SDK transaction IDs and wallet signature verification are transaction-integrity mechanisms, not metadata hashing, and remain.
- Do not fetch media or remote metadata while generating or preparing ARC-69 JSON. No JSON gateway availability/CORS/equality check is necessary.
- Keep a short public/permanent-data notice. A transaction note is public and is not encrypted or private storage.
- Keep byte limits, supported-account checks, network identity checks, fee/balance checks, exact signed-payload verification, and once-only submission recovery. Simplifying metadata UX is not authorization to sign or spend autonomously.

## Administration

One ASA, total 1, decimals 0, default frozen false. Immutable means manager/reserve/freeze/clawback absent: no later manager metadata updates or destruction to recover the asset minimum balance. Creator-manager means only manager=creator; later ARC-69 metadata updates would require another asset configuration transaction. This workspace creates assets only; later administration is not implemented. Changing a note later does not rewrite transaction history or immutable ASA name/unit/URL.

## Proven workflow and evidence

The user reports that the simplified ARC-69 browser workflow worked perfectly in live use. Treat it as user-reported successful use, not an independently verified on-chain receipt: no asset ID, transaction ID or confirmed round was supplied to the agent. Local verification separately passed all 29 tests and the production build. Do not describe this workflow as only locally exercised, and do not claim independent MainNet confirmation without an exact transaction readback.

Preserve the successful UX: media-only ArDrive storage, Step 3 ARC-69 JSON form, metadata in the on-chain note, direct media ASA URL, no separate metadata.json upload, no SHA-256 metadata/media verification, and no attestation checkboxes. Retain separate Prepare / Sign / Broadcast / Confirm buttons and transaction-integrity/recovery checks. Do not reintroduce ARC-3 steps when helping with another mint.

## Run and verify

Resolve workspace under this skill's directory in the active profile; do not modify another profile. Read package.json before commands.

    npm --prefix "/root/.hermes/skills/blockchain/algorand-nft/workspace" test
    npm --prefix "/root/.hermes/skills/blockchain/algorand-nft/workspace" run build

Local tests use injected services and ephemeral fixture signatures, not live wallet/network evidence. Code changes authorize local tests/build; they do not authorize connecting real wallets, live account queries, uploads, signatures, or broadcast. Start the server when the user asks to use the site:

## Packaging and publishing

When publishing or distributing this skill (e.g. via `hermes skills publish`), keep `.skillignore` present to exclude `.venv/`, `workspace/node_modules/`, `workspace/dist/`, and `__pycache__/`. `skills_guard` scans all unignored files; local binaries and minified build bundles trigger critical scanner rules and cause a `DANGEROUS` verdict that blocks publishing.

    npm --prefix "/root/.hermes/skills/blockchain/algorand-nft/workspace" run dev

Verify the site is reachable before saying it is ready; a previously started dev server may have stopped. Check http://127.0.0.1:5173/ for HTTP 200 and the ARC-69 form. If unavailable and site launch is authorized, start the dev server as a background process and check again. Do not treat a background-process ID as readiness evidence.

Serve loopback only at http://127.0.0.1:5173. Remote access needs an approved SSH loopback forward or local copy; no public bind/tunnel. On the user's local workstation, forward with:

    ssh -N -L 5173:127.0.0.1:5173 <ssh-user>@<server-host>

Use the actual SSH user/host if known; otherwise retain explicit placeholders, never invent a hostname. Keep SSH running, open http://127.0.0.1:5173 locally, and refresh after workspace changes. Use one workspace tab, HTTP/HTTPS origin and supported Ed25519 WebCrypto. Never request seeds, private keys, JWKs, or tokens in chat or tools; Pera signs and ArDrive authenticates in its official app.

## Confirmation and recovery

MainNet genesis must match both mainnet-v1.0 and wGHE2Pwdvd7S12BL5FaOP20EGYesN73ktiC1qzkkit8=. Ordinary unrekeyed single-Ed25519 accounts only. Prepared transactions display fee, 100000-microALGO extra asset minimum-balance commitment, validity, roles, direct URL, and exact note JSON. The balance commitment is not a fee. Keep the actual fee capped at 10000 microALGOs and enforce suggested-parameter constraints.

Persist a public recovery record before sending; never persist wallet secrets or signed bytes. A POST timeout means OUTCOME UNKNOWN, not safe failure. Preserve the expected txid, block replacement mints and check that exact transaction explicitly; never automatically resend or clear recovery storage. Old ARC-3 receipts must not be silently discarded during migration. Confirm the exact creation transaction including ARC-69 note and read back asset parameters before reporting confirmed minted with exact-string asset ID, txid, and round. A successful build, wallet signature, node acceptance, or cached receipt is not confirmation.

## Primary sources and caveats

- ARC-69 normative specification: https://raw.githubusercontent.com/algorandfoundation/ARCs/main/ARCs/arc-0069.md
- Transaction fields and limits: https://dev.algorand.co/concepts/transactions/reference/
- Pera SDK: https://docs.perawallet.app/references/pera-connect/

The current ARC-69 document is Final and marked superseded by ARC-89. Honor an explicit request for ARC-69; do not silently implement ARC-89 or claim universal marketplace support. Consult references/pera-arweave.md for storage/wallet boundaries. Keep workflow lessons here, not global memory.
