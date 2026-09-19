# Hermes Browser Extension Pairing

The gateway exposes a loopback-only pairing flow so the Hermes Browser
extension (or any local extension) can obtain a **scoped bearer token** with
one approval click. The token is scoped to the default profile and grants
access only to local chat, model-list, and browser-context requests — it never
exposes the raw `API_SERVER_KEY`.

## Flow

```
Extension                    Gateway (127.0.0.1)
    |  POST /api/browser-extension/pair/start      |
    |---------------------------------------------->|  mint short-lived pairing (180s)
    |  {pairing_id, approval_url, ttl_seconds}      |
    |<----------------------------------------------|
    |  user opens approval_url (approve page)       |
    |                                              |
    |  POST /pair/grant/{id}   (Approve button)     |  status -> approved, mint token
    |  POST /pair/deny/{id}    (Deny button)        |  status -> denied
    |                                              |
    |  GET  /pair/status/{id}  (poll until terminal)|
    |<----------------------------------------------|  {status, token} once approved
    |  use token as Bearer for /v1/* requests       |
```

## Endpoints

All pairing endpoints are **loopback-only** (403 for any non-loopback peer):

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/browser-extension/pair/start` | Mint a pending pairing (180s TTL) |
| GET | `/api/browser-extension/pair/approve/{id}` | Branded approval page (Approve/Deny buttons) |
| POST | `/api/browser-extension/pair/grant/{id}` | Approve; marks the pairing approved and mints the token |
| POST | `/api/browser-extension/pair/deny/{id}` | Reject the pairing |
| GET | `/api/browser-extension/pair/status/{id}` | Poll: `pending` → `approved` (with token) or 410 `denied`/`expired` |

`/v1/capabilities` advertises `browserPairing: true` and the
`browser_pair_*` endpoints so fresh browser installs can discover pairing
before they hold any token.

## Tokens

- Minted by `gateway/browser_pairing.py` (`BrowserPairingStore`) at grant time.
- Persisted under `HERMES_HOME/state/browser_pairing.json`, so pairings and
  tokens survive gateway restarts.
- Accepted by the API-server auth check **only for the default profile** — a
  named profile can never be reached with a token minted against the listener
  key.
- Max token age is 365 days; tokens are not revocable per-session (the
  browser clears them locally when it disconnects).

## Extension contract

The approval page keeps a stable DOM contract for automated tests:

- `#approveButton` submits `POST /api/browser-extension/pair/grant/{id}`
- `#denyButton` submits `POST /api/browser-extension/pair/deny/{id}`

The page ships in the gateway wheel under `gateway/assets/browser_pairing/`
(fonts + artwork loaded as data URIs at request time). If the assets are
missing the page degrades gracefully to fallback fonts with no artwork.
