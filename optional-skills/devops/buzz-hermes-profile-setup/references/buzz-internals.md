# Buzz relay and membership: how it works under the hood

Notes for debugging the installer or for anyone who wants the standalone building
blocks. Source of truth: the `block/buzz` repository (relay, desktop, mobile, CLI) and the
Hermes Buzz adapter at `plugins/platforms/buzz/` (`adapter.py`, `nostr_auth.py`,
`plugin.yaml`).

## Relay info (NIP-11)

`GET https://<community>.communities.buzz.xyz/` returns NIP-11 JSON: `name`,
`description`, `version`, `supported_nips` and `auth_required: true`,
`restricted_writes: true`. The `wss://` endpoint is the Nostr WebSocket relay. The
`https://` base URL is what `relay_url` / `BUZZ_RELAY_URL` expects.

## Membership: relay vs channel

- Channel membership (`buzz channels add-member --channel <uuid> --pubkey <hex> --role bot`)
  writes a channel-member event and returns `accepted:true` immediately.
- It does **not** grant relay membership. Acting as that key afterwards returns
  `{"error":"auth_error","message":"relay error 403: relay_membership_required"}`.
- Relay membership for an agent is granted one of two ways:
  - **Invite mint+claim (the Hermes path, fully programmatic).** An owner/admin mints an
    invite (`POST /api/invites`, NIP-98 signed by the owner); the joining key claims it
    (`POST /api/invites/claim`, NIP-98 signed by the joining key), after accepting the join
    policy if the community has one. Result: a `member`-role relay member, which is exactly
    what the Hermes adapter needs.
  - **Native agent / NIP-OA attestation (the Buzz Desktop path).** Buzz Desktop's
    "Create a new agent" wizard generates a key plus a NIP-OA auth tag for a Buzz-native
    harness (goose/Codex). A Hermes gateway does not use this flow.
- **Self-echo suppression by pubkey.** The adapter ignores inbound events whose pubkey
  equals its own. A profile that runs on the owner's own key *is* the owner and ignores the
  owner's messages. Always use a dedicated keypair. `buzz users get` reveals the identity
  (`display_name`, `role: owner`).

## Invite mint+claim: the HTTP contract

Relative to the relay's HTTPS base URL:

| Step | Method + path | Auth | Body | Returns |
|---|---|---|---|---|
| 1. mint | `POST /api/invites` | NIP-98 (owner/admin) | `{}` or `{"ttl_secs":N,"max_uses":N}` | `{code, expires_at, max_uses, url}` |
| 2. accept-policy | `POST /api/invites/accept-policy` | none | `{code, policy_version, age_confirmed:true}` | `{receipt}` |
| 3. claim | `POST /api/invites/claim` | NIP-98 (joining key) | `{code, policy_receipt}` | `{status:"joined", role:"member", community_id, host}` |

Step 2 is only required when `GET /api/join-policy` returns `{"policy":{...}}`. If it
returns `{}` there is no policy and the claim body is just `{code}`. When
`age_attestation_required` is true, `age_confirmed` must be `true` and `policy_version`
must match `policy.version` exactly, otherwise accept-policy returns
`join_policy_not_accepted` and claim returns `join_policy_required`.

### NIP-98 signing (kind 27235)

`Authorization: Nostr <base64(json)>` where the JSON event is:

```json
{"id":"<hex>","pubkey":"<hex>","created_at":<unix>,"kind":27235,
 "tags":[["u","<url>"],["method","POST"],["payload","<sha256hex(body)>"]],
 "content":"","sig":"<64-byte BIP-340 schnorr hex>"}
```

- `id` = sha256 of the canonical serialization `[0, pubkey_hex, created_at, 27235, tags, ""]`
  as compact JSON (`separators=(",", ":")`, lowercase hex pubkey).
- `sig` = BIP-340 Schnorr over the 32-byte `id`. The installer implements this with the
  standard library (same construction as Hermes' `nostr_auth.py`). ECDSA-only libraries
  cannot produce it.
- `u` is the exact request URL (HTTPS base, not `wss://`), `method` is `POST`, `payload` is
  the sha256 hex of the exact body bytes.
- `created_at` must be within about 60 seconds of server time.
- No nonce tag; replay protection is by event `id`, which is unique per body.

## Cloudflare (why the script uses curl)

The relay sits behind Cloudflare. Python `urllib` requests are rejected with error `1010`
(browser-signature ban) while `curl` with a browser User-Agent passes. The installer shells
out to `curl` for every relay HTTP call.

## Keypair / bech32 padding

When encoding 32 bytes to bech32 (8-bit to 5-bit groups), the final partial group **must be
padded**. A non-padding conversion drops the last bit and produces an nsec that decodes to a
*different* private key, silently. The installer encodes with padding and verifies that
`decode(encode(key)) == key` before using any key.

## Verification: state files, not logs

- `~/.hermes/profiles/<name>/gateway_state.json` contains
  `"platforms": {"buzz": {"state": "connected"}}` when healthy.
- `~/.hermes/profiles/<name>/channel_directory.json` gains a `"buzz"` entry once channels
  surface (empty is normal right after connect).
- The Buzz adapter logs to a different sink than Telegram/Slack, so its lines do not appear
  in `journalctl -u hermes-gateway-<name>`. Absence there is not a failure.
