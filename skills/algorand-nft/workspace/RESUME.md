# Algorand NFT workspace — ARC69 checkpoint

## Completed state

The incomplete ARC-3 → ARC69 conversion is finished in this workspace. Parent skill/reference documentation is maintained separately and was not edited. Package/lockfile remain 0.3.0 with unchanged dependencies; ARC69 recovery format is version 4.

- Step 3 is a compact ARC69 JSON form: required standard="arc69"; optional description/media_url/external_url/mime_type/properties object; exact output and UTF-8 byte count.
- Note application cap is 1024 UTF-8 bytes for conservative standard-fee compatibility, not an absolute protocol maximum. JSON goes directly in the ASA creation note.
- ASA URL points directly to media; assetMetadataHash omitted. No metadata.json upload/export prerequisite, hashes, remote metadata fetch, local file inspection, attestations or extra checkboxes.
- Defaults: Hermes / HRMS01 / immutable, PNG media `wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4`, direct URL `https://arweave.net/wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4#i`.
- Explicit Pera MainNet connection and account selection; no automatic restoration or query.
- Separate Prepare / Sign / Broadcast once / Confirm buttons. Pinned network, account/rekey/balance/fee/round checks, exact unsigned/signed payload and actual Ed25519 TX signature verification remain.
- Exact ARC69 creation note is verified at confirmation, followed by strict ASA parameter/role readback. No metadata digest comparison. SDK absent-hash zero32 normalization is handled.
- Public unsigned recovery persists/readbacks before POST; uncertain outcome locks new mint. Old ARC-3 recovery key is explicitly detected and preserved, never silently ignored. Failed fresh confirmation revokes reset eligibility.

## Last actual verification

Executed locally in this workspace:

```sh
npm test && npm run build && npm ls --depth=0
```

**Exit 0; 29 tests passed, 0 failed.** Vite 8.3.8 built 184 modules in 276ms, producing `dist/`. Expected warning: SDK-containing minified chunk >500 kB. See `VERIFICATION.md` for exact output, coverage and observed RED→GREEN slices.

Dependencies: @perawallet/connect 1.10.0; algosdk 3.9.0; vite 8.3.8; happy-dom 21.0.4. Tests inject services/fake wallets and use ephemeral local cryptographic fixtures, not live integration evidence.

## Recovery keys

- ARC69: `hermes.algorand.mainnet.mint.v0.4`.
- Legacy ARC-3: `hermes.algorand.mainnet.mint.v0.3`.

Any legacy value blocks new mints and remains intact for original-workflow/manual investigation. Current restore recomputes safe review/txid from canonical unsigned bytes, treats saved confirmed status as unverified, and never automatically queries. Unknown/corrupt/unconfirmed storage has no override. Do not clear it, resend or mint a replacement. Only fresh confirmed readback enables the explicit **Start another NFT — may create a duplicate** action. Use one tab; local storage is not global duplicate prevention.

## Permission boundary

This task authorized workspace edits and local finite tests/build only. **No server/live browser launch, real wallet connection/signature, MainNet/account query, gateway/media lookup, storage upload or broadcast occurred.** No real minted asset ID/txid is available. Do not start live work automatically. A future user-requested server should bind loopback only (`npm run dev`, 127.0.0.1:5173); remote access requires separately approved loopback SSH forwarding or a local copy. Live connection/preparation/signing/broadcast remain separate user actions.

Never request or handle recovery phrases, private keys, JWKs, storage tokens or wallet credentials in chat/tools. Pera and official ArDrive handle their own authentication.

## Files and limits

Operational guide: `README.md`; evidence: `VERIFICATION.md`; built artifact: `dist/`. Core source: `src/core.js`, `app.js`, `mint.js`, `mint-ui.js`, `services.js`, `main.js`, `index.html`. Tests include migrated ARC69 core/app/mint/UI coverage plus the new finite service-adapter test; wallet/download utility regression tests remain.

Only ordinary unrekeyed single-Ed25519 accounts and one ASA creation at a time. No multisig/LogicSig/PQ, payments, groups, rekey, transfers, royalties, later administration, automatic uploads or media verification. Pera/mobile/QR, live node/ArDrive/media/marketplace behavior remain unverified. The requested ARC69 format was implemented despite the parent-provided source noting ARC89 supersession.
