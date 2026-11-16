# Hermes Algorand NFT workspace · ARC69

Browser application for **Algorand MainNet ARC69**: compact on-chain JSON note, direct off-chain media URL, Pera signature, separate once-only submission, and explicit verified confirmation. Implemented and locally tested; **not live-tested**. No server, real wallet connection, real signature, node/gateway query, upload or broadcast was performed during this conversion.

## Local verification

```sh
cd /root/.hermes/skills/blockchain/algorand-nft/workspace
npm test
npm run build
```

Tests inject wallet/node services and use ephemeral local SDK fixture signatures only. DOM tests disable script evaluation and external resources. `dist/` is the generated production artifact; see `VERIFICATION.md` for actual output. No dependency changes were needed. The package version remains 0.3.0; the ARC69 recovery schema/key is version 4, distinct from the legacy ARC-3 format.

Manual server commands are documentation only; they were not run:

```sh
npm run dev      # loopback 127.0.0.1:5173, strictPort
npm run preview  # loopback 127.0.0.1:4173, strictPort
```

Use an HTTP/HTTPS browser origin, not file://, and a browser supporting Ed25519 WebCrypto. Do not expose a wallet-facing development server publicly. Remote access requires a user-approved loopback SSH forward or local copy. Use one workspace tab.

## Workflow

1. **Wallet:** explicitly connect Pera, requesting MainNet chain ID 416001, and select a returned public address. No automatic reconnect/session restore. Connection alone does not query the node, request a signature or spend ALGO. Never enter seed phrases, private keys, JWKs or tokens here.
2. **Media:** manually open ArDrive's official app and upload media into a Public Drive, or reuse an existing direct URL. ArDrive handles storage authentication/payment. This app never uploads or fetches media. Use the raw File Data Tx ID, not a share page or ArFS metadata record. Public publication is intended to be permanent; do not publish private data or content you may not distribute.
3. **ARC69 JSON form:** click **Generate ARC69 JSON** to produce compact UTF-8 JSON. Required `standard` is always `"arc69"`. Optional fields are `description`, `media_url`, `external_url`, `mime_type`, and a JSON-object `properties`. Empty optional fields are omitted. Name/unit are ASA settings, not required JSON fields. The read-only output and byte counter show the exact note. This workspace enforces **1024 UTF-8 bytes as a conservative application standard-fee cap**, not an absolute protocol limit. Editing form fields invalidates the generated note and review/signature; generate again. The note is public transaction history, not encrypted/private storage.
4. **ASA settings:** choose name (1–32 UTF-8 bytes), unit (1–8), direct media URL (≤96), and administration. A literal 43-character base64url File Data Tx ID expands to `https://arweave.net/ID`; direct HTTPS/ipfs URLs are preserved, never silently trimmed or repaired. Optional media type fragments: `#i`, `#v`, `#a`, `#p`, `#h`. The URL points to media, never a metadata JSON reference or `#arc3`. No `assetMetadataHash` is set, and no media/metadata SHA-256 is calculated. No metadata.json download, separate JSON upload, remote JSON availability/CORS/equality test, attestation, replacement acknowledgment or checkbox gate is required.
5. **Mint:** use the separate Prepare, Sign, Broadcast and Confirm buttons below. Buttons express their operations; no action automatically advances to the next financial operation.

### Current default NFT

- Name: **Hermes**; unit: **HRMS01**.
- Policy: **immutable** (manager/reserve/freeze/clawback absent).
- PNG media ID: `wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4`.
- Direct ASA/media URL: `https://arweave.net/wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4#i`.
- Default JSON form: description `Hermes`, that `media_url`, MIME `image/png`, properties `{}` (omitted until nonempty).

Media existence, content and MIME are user-supplied, not remotely verified.

## Four explicit mint actions

**Prepare / review (read-only):** reads suggested parameters and the selected public account from the fixed `mainnet-api.algonode.cloud` node. No media or remote metadata request exists. Pins both `mainnet-v1.0` and genesis hash `wGHE2Pwdvd7S12BL5FaOP20EGYesN73ktiC1qzkkit8=`. Requires matching account, no foreign rekey, sufficient spendable balance for SDK-calculated fee plus 100000 microALGO additional asset minimum balance. Actual fee must be 1000–10000 microALGO; balance commitment is not a fee. Creates exactly one ASA, total 1, decimals 0, default frozen false, with exact generated note and direct media URL. Validity is bounded by node suggestion and at most 200 rounds. Review displays sender, fields/roles, note, URL, fee, balance commitment, rounds, txid and canonical unsigned bytes. Context binds wallet instance/generation/address, document version/bytes, URL, name/unit/policy. Relevant changes, two-minute expiry or backwards clock movement reject further action.

**Sign reviewed mint in Pera:** invokes the statically loaded wallet SDK directly from the action, without an awaited import or pre-sign request. Accepts exactly one signed byte payload. SDK decoding must reproduce exact unsigned transaction bytes/ID; only ordinary single 64-byte Ed25519 signatures are supported. Foreign signer, multisig, LogicSig and PQ signatures reject. WebCrypto verifies the actual transaction's domain-separated `bytesToSign()` (TX), not an unrelated message signature. Context is checked after wallet completion/verification. Signed bytes remain only in memory. Signing never broadcasts.

**Broadcast signed mint once:** this separate button authorizes submission. Rechecks pinned genesis, current validity, account/balance, context/freshness and exact signature. Before any POST, persists and reads back a **public unsigned recovery receipt**. Storage failure or mismatched readback prevents POST. Receipt contains public sender, txid, canonical unsigned bytes, note/review/roles/fee/validity; never signed bytes or secrets. Installs the in-memory lock before the single POST. SDK adapter requests have finite 15-second abort deadlines; POST is never automatically retried. Timeout or wrong returned ID means **OUTCOME UNKNOWN**, not safe failure. Node acceptance is not confirmation.

**Confirm saved transaction (read-only):** a finite explicit check, no polling. Pins genesis, queries the exact saved txid, requires no pool error, positive confirmed round/asset ID, exact canonical creation transaction **including ARC69 note**, and round inside validity. Reads ASA parameters back and checks index, creator, total, decimals, frozen flag, name/unit/direct URL and all administrative roles. There is **no metadata digest comparison**: the field must remain absent; undefined/empty/SDK-normalized zero32 representations count as absent, nonzero commitments reject. Results use exact decimal-string asset ID and round plus txid. Only verified readback reports **Confirmed minted**. Confirmation does not depend on the current wallet or edited form. Failed checks keep the lock; a failed recheck revokes eligibility to start another NFT.

## Administration

Immutable default: no later manager metadata updates or destruction to reclaim minimum balance. Creator-manager option: manager=creator only; reserve/freeze/clawback absent. Later ARC69 updates would need another asset configuration transaction; prior history and ASA name/unit/URL remain unchanged. Creation only is implemented; no later administration, transfers, royalties, groups, smart contracts, multisig/LogicSig/PQ, rekeys or direct uploads.

## Recovery and migration

- Current ARC69 key: `hermes.algorand.mainnet.mint.v0.4` (schema version 4).
- Legacy ARC-3 key: `hermes.algorand.mainnet.mint.v0.3`. **Any existing value blocks new mints.** It is not deleted, overwritten, trusted as confirmed or silently discarded. Preserve its original storage/public txid and investigate through the original workflow/manual recovery. This version does not convert or confirm legacy receipts.
- Current restoration decodes canonical unsigned bytes, recomputes txid, validates pinned network and supported creation/note/roles, and derives review instead of trusting stored review. Cached confirmed labels are downgraded to pending until explicitly checked again. Corrupt/unavailable storage fails closed.
- Receipts survive edits, disconnect and reload. Reload does not query anything. Unknown/unconfirmed outcomes never unlock because of expiry, pool errors, 404/missing history or later readback mismatch. **Never resend or mint a replacement.** There is no unsafe in-app override.
- After verified confirmation in the current session, **Start another NFT — may create a duplicate** explicitly clears only the confirmed current receipt. Retain your public receipt first. No extra checkbox is required. Browser-local storage is not a global uniqueness guarantee: deleted storage, other browsers/origins/apps or simultaneous tabs can bypass it.

## Implementation and verification limits

`src/core.js`: compact ARC69/schema/byte/reference validation. `app.js` and `index.html`: form and explicit wallet/manual storage handoff. `mint.js`: SDK construction, exact payload/signature, context, recovery, once-only submission and strict confirmation. `mint-ui.js`: action buttons and safe text-only review/status rendering. `services.js`: finite fixed-node SDK adapter, with no metadata fetch API. `wallet.js`: explicit Pera lifecycle. `download.js`: standalone generic local download utility, not required by minting.

Local tests do not establish live Pera/mobile/QR, node availability, ArDrive payment/upload, remote media, marketplace behavior or real on-chain minting. Pera's available disconnect/network events are handled; no public accountsChanged API is assumed. Static SDK loading preserves user gestures and causes Vite's expected >500 kB bundle warning. No server was started. Parent skill/reference documentation is outside this task's edit scope.

The parent-provided ARC69 normative source is `https://raw.githubusercontent.com/algorandfoundation/ARCs/main/ARCs/arc-0069.md`; it is marked superseded by ARC89. This implementation honors the requested ARC69 format rather than silently changing standards. Marketplace support is not claimed.
