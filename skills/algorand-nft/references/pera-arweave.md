# Pera + ArDrive for ARC-69

The active workflow is ARC-69. Read workspace/README.md and references/mainnet-minting.md. Do not restore the old ARC-3 two-upload/hash/attestation workflow.

## Wallet

Use Pera chainId 416001 for MainNet and connect only on an explicit user action. Never request recovery phrases, private keys, storage keys/JWKs or tokens through agent tools/chat. Keep SDK statically available before connect/sign clicks to retain user gestures. Inspect installed SDK types before changing APIs; do not assume latest docs and the installed package match. Current package versions are in workspace/package.json.

Connection is not permission to broadcast. Separate Prepare, Sign, Broadcast and Confirm buttons with a transaction review, no extra attestation checkboxes. Keep exact-payload and Ed25519 signature checks, network checks and once-only recovery. Metadata SHA-256 is not needed.

Handle disconnect/networkChanged events according to installed Pera APIs; do not invent accountsChanged support. Clear selection on a network change. Require explicit disconnect/reconnect for accounts outside the shared list; no automatic session restoration. Do not claim wallet-side revocation succeeded if disconnect failed.

## Storage

Only media needs off-chain storage. The user signs in and pays/uploads through official ArDrive https://app.ardrive.io using a public drive. Do not assume ALGO in Pera funds ArDrive. Keep a short notice: media uploads and on-chain notes are public; public storage is intended to be permanent. Review rights/privacy and actual quote for any agent-assisted upload. Direct uploads are not implemented.

Paste the media's raw File Data Transaction ID or supported direct media URI. A literal Arweave ID is 43 base64url characters. Validate format before using it and preserve exact characters. An ArFS metadata transaction, ArDrive share page or manifest is not necessarily a raw media file. A format-valid ID does not prove existence, content type, ownership, availability or storage finality. Never claim an HTTP 200 or MIME header establishes confirmed permanent storage.

For Arweave use the direct https://arweave.net/ID media URL; ARC-69 recommends #i/#v/#a/#p/#h by media kind. For IPFS prefer ipfs:// rather than a gateway URL. Do not append #arc3. Do not download media or query gateways automatically.

## On-chain JSON

Step 3 generates compact JSON: required standard="arc69"; optional description, external_url, media_url (high-resolution media), mime_type and properties object. Name and unit live in ASA parameters. Show actual UTF-8 byte size and enforce this workspace's conservative 1024-byte standard-fee cap, not a claimed universal protocol maximum. Insert exact bytes in creation note. No separate metadata.json download/upload or JSON Data Tx ID is required. Omit optional assetMetadataHash, all SHA-256/integrity features and exact-file attestations.

ARC-69 source: https://raw.githubusercontent.com/algorandfoundation/ARCs/main/ARCs/arc-0069.md . The document is Final but superseded by ARC-89; honor requested ARC-69 and avoid universal marketplace-compatibility claims.

## Local execution

Read package.json and README before commands. Local tests/build are not live wallet integration. Keep loopback server HTTP origin, no file:// or public binding; remote access via approved SSH forwarding. No unapproved uploads, real signatures, account queries or broadcast just to smoke-test. User's explicit action in the site starts the chosen operation.

Legacy scripts/nft.py is TestNet-only ARC-3, obsolete for this workspace. Leave its guards intact and never use it as a MainNet fallback.
