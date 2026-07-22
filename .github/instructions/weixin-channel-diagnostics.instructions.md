---
description: "Use when diagnosing why the WeChat (weixin) channel is not connecting, why the gateway logs 'Session expired; pausing for 10 minutes', why the dashboard shows weixin as disconnected/error, or when the user needs to re-scan the QR code after token expiry. Covers the iLink Bot API base URL, errcode=-14 session expiry, QR re-login via dashboard or CLI, and data/.env credential checks."
name: "Weixin Channel Diagnostics"
applyTo:
  - "gateway/platforms/weixin.py"
  - "gateway/platforms/weixin_qr_session.py"
  - "docker/hermes-config.yaml"
  - "docker-compose.upstream.yml"
  - "docker-compose.yml"
  - "data/.env"
  - "INSTALL.md"
---

# Weixin (WeChat) Channel Diagnostics

The weixin platform connects Hermes to personal WeChat via Tencent's
**iLink Bot API**. It uses outbound long-poll — no inbound host port
or webhook needed. Most connection failures trace to one of three
causes: wrong base URL, expired session token, or missing/incorrect
env vars.

## Quick triage (run in this order)

1. **Check the dashboard status first:**
   ```bash
   docker exec hermes-gateway curl -s http://hermes-web:9119/api/status | python3 -m json.tool
   ```
   Look at `gateway_platforms.weixin.state`:
   - `connected` — adapter started, but check logs to confirm polling is
     actually succeeding (the dashboard can report `connected` before the
     first long-poll returns).
   - `error` / `disconnected` — check `error_message` and `error_code`.

2. **Check the gateway logs for weixin-specific errors:**
   ```bash
   docker logs hermes-gateway --tail 200 2>&1 | grep -iE "weixin|ilink|expired|errcode|ret=-"
   ```

3. **Check the saved credentials and env vars:**
   ```bash
   docker exec hermes-gateway sh -c 'ls -la /opt/data/weixin/accounts/ && grep WEIXIN /opt/data/.env'
   ```

## Common failure modes

### 1. Wrong `WEIXIN_BASE_URL` (most common silent failure)

The correct iLink API endpoint is `https://ilinkai.weixin.qq.com`.

If `data/.env` contains `WEIXIN_BASE_URL=https://ilinkai.wechat.com`
(missing the `.qq.com` suffix), the adapter will start, report
`connected` on the dashboard, then fail with `Session expired` or
generic API errors on the first long-poll.

**Fix:**
```bash
docker exec hermes-gateway sed -i 's|ilinkai.wechat.com|ilinkai.weixin.qq.com|' /opt/data/.env
docker restart hermes-gateway
```

The adapter's built-in default is correct
(`ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"` in
`gateway/platforms/weixin.py`), so the simplest fix is to **remove** the
`WEIXIN_BASE_URL` line from `.env` entirely and let the default apply.

### 2. Session expired (`errcode=-14`)

Log signature:
```
ERROR gateway.platforms.weixin: [Weixin] Session expired; pausing for 10 minutes
```

This means the iLink bot token has expired. The adapter pauses for 10
minutes then retries, but an expired token will not self-heal — you
must re-scan the QR code.

**Fix — Option A: Web Dashboard (recommended):**
1. Open `http://localhost:9119` in a browser
2. Navigate to **Channels**
3. Find **Weixin / WeChat** → click **Set up with QR**
4. Scan the QR with WeChat on your phone → confirm login
5. The dashboard auto-saves credentials and restarts the gateway

**Fix — Option B: CLI wizard inside the container:**
```bash
docker compose -f docker-compose.upstream.yml run --rm hermes-gateway hermes gateway setup
```
Pick **Weixin / WeChat** (item 13 in the default menu). Scan the QR,
confirm in WeChat. The wizard saves credentials to
`/opt/data/weixin/accounts/<account_id>.json` and updates
`WEIXIN_ACCOUNT_ID` / `WEIXIN_TOKEN` in `data/.env`.

After re-login, restart the gateway:
```bash
docker restart hermes-gateway
```

### 3. Missing or incorrect env vars

Required in `data/.env`:
```
WEIXIN_ACCOUNT_ID=<account_id>@im.bot
WEIXIN_TOKEN=<account_id>@im.bot:<token_hex>
```

`WEIXIN_BASE_URL` is optional — if unset, the adapter defaults to
`https://ilinkai.weixin.qq.com`. Only set it if you need to override.

If `WEIXIN_ACCOUNT_ID` or `WEIXIN_TOKEN` is missing, the adapter fails
at startup with a clear error. Re-run the QR wizard to regenerate both.

### 4. Token lock conflict

```
Another local Hermes gateway is already using this Weixin token
```

Only one gateway instance can long-poll a given token. Stop the other
instance first:
```bash
docker ps | grep hermes-gateway
docker stop <other-container>
```

### 5. Missing Python dependencies (non-Docker only)

```
Weixin startup failed: aiohttp and cryptography are required
```
```bash
pip install aiohttp cryptography
```
The Docker image already includes both — this only affects bare-metal
or venv installs.

## After any fix: verify polling is working

```bash
# Wait 40s for the first long-poll cycle, then check logs
sleep 40
docker logs hermes-gateway --since 1m 2>&1 | grep -iE "weixin|ilink"
```

A healthy adapter produces **no weixin log lines** after startup — the
long-poll is silent when there are no inbound messages. The absence of
`Session expired` or `ERROR` lines is the success signal.

Also re-check the dashboard status:
```bash
docker exec hermes-gateway curl -s http://hermes-web:9119/api/status | python3 -m json.tool
```
`weixin.state` should be `connected` with `error_code: null`.

## Credential file locations

- **Saved accounts:** `data/weixin/accounts/<account_id>.json`
  (one file per QR login; the most recent is used)
- **Context tokens:** `data/weixin/accounts/<account_id>.context-tokens.json`
  (disk-backed reply continuity — safe to delete, will be rebuilt on
  next inbound message)
- **Env vars:** `data/.env` — `WEIXIN_ACCOUNT_ID`, `WEIXIN_TOKEN`,
  `WEIXIN_BASE_URL` (optional)

## iLink bot identity limitations

QR login connects an **iLink bot identity** (e.g. `...@im.bot`), not a
fully scriptable personal WeChat account. Consequences:
- The bot identity generally **cannot be invited into ordinary WeChat
  groups**.
- iLink typically **does not deliver ordinary WeChat group events**.
- `@`-mentioning the personal account used for QR login is NOT the same
  as `@`-mentioning the iLink bot.

If group messages never reach Hermes, the limitation is on the iLink
side, not in Hermes. The gateway logs a `WARNING` at startup when
`WEIXIN_GROUP_POLICY` is set to anything other than `disabled`.

See `website/docs/user-guide/messaging/weixin.md` for the full user
guide and all env vars.