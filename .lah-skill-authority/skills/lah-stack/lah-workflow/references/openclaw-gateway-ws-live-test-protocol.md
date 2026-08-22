# OpenClaw Gateway WS Live-Test Protocol (dashboard path)

Established during CLOE_DASHBOARD_REPLY_SESSION_CONFLICT_REPAIR_V1 phase-2
final activation (2026-08-03). Enables driving the REAL dashboard gateway
(127.0.0.1:18789) exactly like the SPA — proving transcript persistence and
reply behavior through the actual chat.send pipeline instead of unit mocks.

## When to use
- Live verification of a CLOE dashboard fix (transcript dedup, reply init,
  session persistence) against the deployed gateway.
- Protocol-level proof that complements (never replaces) the operator's
  real-browser test. Once the deployed loader is verified and a live transcript
  proves the guarantees, STOP — hand the browser test to the operator.

## Gateway facts
- Gateway: systemd USER unit `openclaw-gateway.service` (not Docker).
  `systemctl --user` needs `export XDG_RUNTIME_DIR=/run/user/999`.
  MainPID via `systemctl --user show openclaw-gateway.service -p MainPID --value`.
  NODE_OPTIONS via `tr '\0' '\n' < /proc/<PID>/environ | grep NODE_OPTIONS`.
- Listens on 127.0.0.1:18789. Health: `curl -s http://127.0.0.1:18789/health`
  → `{"ok":true,"status":"live"}`. Health can lag ~8s after restart (HTTP 000
  right after start is startup lag, not failure — re-check after sleep).
- WS upgrade: `GET / HTTP/1.1` with `Upgrade: websocket` — ANY path works
  (the HTTP handler returns early for `upgrade: websocket`; plugin paths are
  checked first but a bare `/` passes through to wss.handleUpgrade).

## Handshake (reverse-engineered from dist, all values verified live)

### 1. WS upgrade
- `Sec-WebSocket-Key` MUST be a real base64 of 16 random bytes
  (`randomBytes(16).toString('base64')`, 24 chars). A UUID-derived key
  (22 chars) → `400 Missing or invalid Sec-WebSocket-Key header`.
- `Origin` header: CONTROL_UI client requires an origin from
  `gateway.controlUi.allowedOrigins` (e.g. `https://studio.leanframeworklab.com`).
  CLI/webchat clients pass without Origin.

### 2. Connect params (protocol version = 4)
```json
{
  "minProtocol": 4, "maxProtocol": 4,
  "client": {"id": "cli", "version": "2026.6.11", "platform": "linux",
             "deviceFamily": "linux", "mode": "cli"},
  "role": "operator",
  "scopes": ["operator.read", "operator.write"],
  "auth": {"deviceToken": "<device-token>"},
  "device": {"id": "<deviceId>", "publicKey": "<raw-b64url>",
             "signature": "<sig>", "signedAt": <ms>, "nonce": "<server-nonce>"}
}
```
- Client IDs (schema enum): `openclaw-control-ui`, `webchat-ui`, `openclaw-tui`,
  `webchat`, `cli`, `gateway-client`, ... Modes: `webchat`, `cli`, `ui`,
  `backend`, `node`, `probe`, `test`.
- Protocol version is 4 (1 → `PROTOCOL_MISMATCH`, expectedProtocol=4).
- Plain `auth.token` from the gateway config grants scopes: `[]` for webchat
  and cli clients — NOT enough for chat.send. The gateway token from
  `~/.openclaw/openclaw.json` (`gateway.auth.token`, 64 chars) differs from the
  `.env` token (48 chars) → `AUTH_TOKEN_MISMATCH` if you use the wrong one.
- The ONLY path that grants `operator.write` (required by chat.send) is the
  device token + device identity signature (below). CLI clients do NOT get
  `CLI_DEFAULT_OPERATOR_SCOPES` server-side when connecting from a browser-
  like WS client; the scopes are derived from the device token.

### 3. Device identity handshake (the SPA flow — this is what works)
1. After WS upgrade, server emits event `connect.challenge` with
   `payload.nonce` — WAIT for it; do not send connect immediately.
2. Read device identity:
   - `/home/deploy/.openclaw/identity/device.json` →
     `{version, deviceId, publicKeyPem, privateKeyPem, createdAtMs}`
   - `/home/deploy/.openclaw/identity/device-auth.json` →
     `tokens.operator.token` (scopes `[operator.pairing, operator.read, operator.write]`)
3. Build payload v3 (pipe-joined, scopes comma-joined):
   ```
   v3|deviceId|clientId|clientMode|role|scopes|signedAtMs|signatureToken|nonce|platform|deviceFamily
   ```
   where clientId=`cli`, clientMode=`cli`, role=`operator`,
   scopes=`operator.read,operator.write`, signatureToken=`<device-token>`,
   platform=`linux`, deviceFamily=`linux`. `signedAtMs = Date.now()`.
4. Sign: Ed25519 RAW — `crypto.sign(null, Buffer.from(payload,'utf8'),
   createPrivateKey(privateKeyPem))` then **base64url** encode
   (`+`→`-`, `/`→`_`, strip `=`). Do NOT use `createSign('sha512')` →
   `Unsupported crypto operation`.
5. `publicKey` in device params: raw 32-byte Ed25519 key as base64url —
   strip the 12-byte SPKI AlgorithmIdentifier prefix from the DER export
   (`key.export({type:'spki',format:'der'}).subarray(der.length-32)`), NOT the PEM.
6. Send `connect` with the params above. hello-ok `auth.scopes` must include
   `operator.write`; then chat.send works.

### 4. chat.send contract
- `message` must be a plain STRING (`message: "text"`). An object form
  (`{role:'user', content:[...]}`) → `INVALID_REQUEST: at /message`.
- Params: `sessionKey` (e.g. `agent:cloe-poc:dashboard:<uuid>`),
  `message`, `idempotencyKey` (SPA runId), optional `timeoutMs`.
- Response: `{runId, status}`. `ok:false` with `missing scope: operator.write`
  means auth scopes were empty (fix the device handshake, not the params).

## Transcript verification
- The session key maps to a session FILE via
  `/home/deploy/.openclaw/agents/cloe-poc/sessions/sessions.json`
  (`entry.sessionFile`). The file is `<sessionId>.jsonl` — NOT the dashboard
  UUID. Counting by dashboard UUID returns 0.
- Transcript entries: `{type:'message', message:{role, content, idempotencyKey, timestamp}}`.
- First flush can lag the reply (~5-6s) — POLL for the file/count, don't
  check once.
- Per-session serialization: a send issued while the previous reply is still
  streaming JOINS the in-flight run (gateway content dedupe on sessionKey+
  message for webchat channel). For an intentional-repeat test, wait for the
  previous assistant reply to appear in the transcript BEFORE sending the
  same text again — otherwise the "legitimate repeat" is legitimately joined
  to the still-running turn and never becomes a second entry.

## Safety constraint (operator-mandated, durable)
A bounded content-hash fallback (30s bucket) for keyless user turns must be
gated to dashboard sessions ONLY (`sessionKey.includes(':dashboard:')`).
Telegram/CLI/webchat carry their own transport message IDs — a deliberate
identical repeat there is legitimate and must NEVER be deduplicated.
Implementation: forward `sessionKey` from both transcript append builders
(`appendTranscriptTurnMessages`: `...target.sessionKey ? {sessionKey} : {}`;
`appendTranscriptMessage(scope, options)`: `...scope.sessionKey ? {sessionKey} : {}`),
then gate the fallback with `isCloeDashboardSessionKey(params.sessionKey)`.
The primary key path (message.idempotencyKey) is NOT gated — it only fires on
explicit keys, which are channel-safe.

## Rollback of loader changes
Preserve the pre-change loader before deployment:
```bash
cp /home/deploy/openclaw-runtime/lah-openclaw-mvp/scripts/session-accessor-patch.mjs \
   ~/.lah-secrets/rollback-<mission>/session-accessor-patch.mjs.<tag>
sha256sum <both files>
```
Restore = copy back + `systemctl --user restart openclaw-gateway.service`.
