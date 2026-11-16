# ARC69 conversion · local verification record

## Scope and boundary

Only `/root/.hermes/skills/blockchain/algorand-nft/workspace` was edited. Loaded the test-driven-development and algorand-nft skills, read the incomplete source/tests, and exercised local tests/build. No server or live browser was launched. No real wallet connection/signature, node/account/MainNet/gateway/media query, upload or broadcast occurred. Tests use injected services, disabled external-resource DOM fixtures and ephemeral SDK-generated local signing fixtures. No user secrets or signed payloads were persisted.

## Final execution evidence

Executed in the workspace:

```sh
npm test && npm run build && npm ls --depth=0
```

Result: **exit 0**.

A subsequent local static artifact check also returned **exit 0**:

```text
Static artifact checks passed: source/build have no checkboxes; no obsolete metadata/hash APIs; both recovery namespaces retained; default media and bundled assets verified.
```

It checked the actual source and built HTML and verified every linked bundle asset exists, without serving or launching the application.

### Full suite

```text
node --test tests/*.test.js
ℹ tests 29
ℹ suites 0
ℹ pass 29
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms 1681.224096
```

Coverage:

- Compact ARC69 form with required standard and optional fields/properties, standard-only JSON, no ASA name fields in JSON, exact generated output/byte counter.
- Exact 1024-byte acceptance, 1025-byte and multibyte overflow rejection; invalid standard, types, properties, MIME, BOM, UTF-8 and JSON rejection.
- Literal direct media references/IDs, no whitespace repair, 96-byte URL cap, HTTPS/ipfs, allowed media fragments, ARC-3 and unsafe URL rejection, external page anchors.
- No checkboxes/metadata upload/export/hash/local media-inspection dependency; default Hermes/HRMS01/immutable/PNG URL; text-only rendering and invalid-form fail-closed behavior.
- Explicit Pera MainNet construction, account selection, disconnect/rejection/network-change cleanup, no automatic restoration.
- Real SDK MainNet asset creation with exact note, direct media URL, total 1/decimals 0/frozen false, role restrictions and no encoded assetMetadataHash (`apar.am` absent).
- Network/account/rekey/balance/fee/validity/UTF-8 checks; creator-manager option changes only manager.
- Actual Ed25519 TX-domain signature verification; bad signature, changed transaction, foreign signer, unsupported WebCrypto, multisig and oversized payload rejection.
- Immediate separate signing without node/metadata calls, no automatic submission, wallet rejection/invalid payload count, stale review and pending context-change rejection.
- Public unsigned receipt persisted/read back before exactly one POST; storage failure/readback mismatch prevents submission; timeout/wrong response ID locks outcome unknown; edits/disconnect/reload do not unlock.
- Legacy v0.3 key detection preserves the original value and blocks new prepare/sign/broadcast without network calls. Current receipt restoration derives review/txid from safe canonical unsigned bytes, rejects corrupt/unsafe data, and never trusts cached confirmed status.
- Exact creation/note confirmation, large exact-string IDs/rounds, full ASA field/role mismatch table, confirmation storage errors, pool errors/unconfirmed/expired/missing history lock retention.
- Omitted/empty/SDK zero32 metadata-hash representations accepted as absent; nonzero commitments rejected without computing/comparing a metadata digest.
- A failed fresh confirmation clears eligibility to start another NFT; explicit button-only confirmed reset; reload requires confirmation again.
- Finite credential-free/referrer-free fixed-node adapter, no metadata service/export, and no POST retry. End-to-end injected DOM Prepare → Sign → Broadcast → Confirm path.

Old ARC-3-specific assertions/APIs were replaced, not retained as requirements: exact metadata export/upload attestations, local media hashing and gateway equality checks are no longer this workflow. The standalone generic download helper retains its own unit test but is not imported by the application.

### Production build

```text
vite v8.3.8 building client environment for production...
transforming...
✓ 184 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                         8.26 kB │ gzip:   3.29 kB
dist/assets/index-9fFYQpoG.css          3.79 kB │ gzip:   1.48 kB
dist/assets/App-CiuL_0zi-V9YrY_dI.js  323.36 kB │ gzip:  87.53 kB
dist/assets/index-DaGGpzy_.js         646.35 kB │ gzip: 156.36 kB
✓ built in 276ms
```

Vite also emitted its **>500 kB minified chunk warning**. This is a warning, not a build failure; it was not suppressed. Static wallet/SDK loading preserves direct user gestures. `dist/` was rebuilt, not served or live-browser-tested.

### Installed direct dependencies

```text
hermes-algorand-nft-workspace@0.3.0
├── @perawallet/connect@1.10.0
├── algosdk@3.9.0
├── happy-dom@21.0.4
└── vite@8.3.8
```

No dependency install or manifest/lockfile changes were needed. No package audit is claimed.

## Observed incremental RED → GREEN evidence

1. Existing broadcast regression failed on the obsolete separate-spend-approval gate. Removed attestation parameters and remote metadata call; real local signed broadcast/persist-before-POST/timeout/reload tests passed.
2. New legacy receipt test failed because current and legacy keys were identical. Separated ARC69 v0.4 from ARC-3 v0.3 and added fail-closed legacy detection; test passed without touching original storage.
3. New ARC69 confirmation test failed at `from64(r.hash)` because ARC69 has no metadata digest. Replaced digest equality with absent-field validation and kept exact transaction/note and ASA readback; omitted/empty/zero32 cases passed. Removed the old extra reset acknowledgment parameter, keeping the explicit verified-confirmation action.
4. New failed-confirmation regression exposed that prior successful confirmation still allowed resetting after a failed fresh check. Clearing `confirmedHere` at the start of a new confirmation made the regression pass.
5. Form/DOM tests initially failed to import the old app because it required removed ARC-3 exports. Replaced the entire obsolete form/reference/checkbox integration with compact ARC69 form and four explicit mint buttons; both DOM paths passed.
6. New service test failed because remote metadata was still exposed. Deleted that adapter method and verifier; test passed with a fake SDK client and no network.
7. External page-anchor test failed because a non-media external_url was being treated as an ASA type fragment. Restricted type-fragment checking to media URLs; test passed.
8. Added broad regression tables for preserved safety behavior. One initial multisig fixture attempted to construct two auth forms and was rejected by the SDK constructor before validation; corrected it to a valid SDK multisig fixture. No production weakening was needed; full suite/build passed.

## Migration/workflow lessons

- Change the receipt namespace only with explicit legacy-key detection. Silently ignoring an old key can cause duplicate mints.
- Confirm the original transaction's exact note, not only current ASA parameters. No off-chain digest can substitute for ARC69 note equality.
- SDK decoding may represent an absent metadata hash as zero32; require absence semantically, never invent a digest for the on-chain note.
- Removing attestation checkboxes must not remove the separate spend action, exact signature verification, network/account/fee checks or persist-before-POST lock.
- A failed confirmation recheck must revoke prior reset eligibility even when an earlier result succeeded.
- Use direct first-class patch/write tools for edits and terminal for local execution. No approval-wrapper workaround or fabricated output was used.

## Unverified / intentional limits

Live Pera mobile/extension/QR, MainNet node behavior, ArDrive payment/uploads, actual media availability/MIME and marketplace rendering remain unverified. No real asset ID/txid exists from this task. No later administration/transfer, direct storage upload or remote media integrity feature is implemented. Unknown/corrupt/legacy receipts need manual investigation, never a replacement mint. Browser-local recovery cannot guarantee uniqueness across tabs/origins/profiles/deleted storage; use one tab and retain public receipts. The 1024-byte note cap is an application standard-fee cap, not an absolute protocol limit. Parent skill/reference files were not edited.
