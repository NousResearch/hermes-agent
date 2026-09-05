---
name: weixin-reconnect
description: "Diagnose and fix the WeChat (weixin) channel when it won't connect — wrong base URL, expired session (errcode=-14), missing credentials, or token lock conflicts. Walks through triage, fixes data/.env, and guides QR re-scan via dashboard or CLI. Use when weixin is not connecting, gateway logs 'Session expired', or the dashboard shows weixin as error/disconnected."
argument-hint: "[symptom or error message]"
user-invocable: true
---

# Weixin Reconnect

Use this skill when the WeChat (weixin) channel is not connecting to the
Hermes gateway. The weixin platform uses Tencent's iLink Bot API with
outbound long-poll — no inbound host port needed.

## Read First

- [Weixin Channel Diagnostics](../../instructions/weixin-channel-diagnostics.instructions.md) — the canonical triage reference
- [Weixin user guide](../../../website/docs/user-guide/messaging/weixin.md) — full env var reference and iLink bot limitations
- [gateway/platforms/weixin.py](../../../gateway/platforms/weixin.py) — adapter source (`ILINK_BASE_URL`, retry logic, errcode handling)
- [docker/hermes-config.yaml](../../../docker/hermes-config.yaml) — `platforms.weixin` config section
- [docker-compose.upstream.yml](../../../docker-compose.upstream.yml) — gateway service definition

## When to Use

- Gateway logs `ERROR gateway.platforms.weixin: [Weixin] Session expired; pausing for 10 minutes`
- Dashboard shows `weixin` state as `error` or `disconnected`
- User reports WeChat messages not reaching the agent
- After a stack restart when weixin was previously working
- User says "weixin not connected", "WeChat broken", "微信连不上"

## Triage Procedure

Run these steps in order. Stop at the first fix that resolves the issue.

### Step 1 — Check dashboard status

```bash
docker exec hermes-gateway curl -s http://hermes-web:9119/api/status | python3 -m json.tool
```

Inspect `gateway_platforms.weixin`:
- `state: "connected"`, `error_code: null` → adapter started; proceed to Step 2 to confirm polling is actually working (dashboard can report `connected` before the first long-poll returns).
- `state: "error"` or `error_code` is set → note the `error_message`, proceed to Step 3.

### Step 2 — Check gateway logs for weixin errors

```bash
docker logs hermes-gateway --tail 200 2>&1 | grep -iE "weixin|ilink|expired|errcode|ret=-"
```

Key log signatures and their meanings:

| Log line | Meaning | Go to |
|----------|---------|-------|
| `Session expired; pausing for 10 minutes` | Token expired (errcode=-14) | Step 4 |
| `Weixin startup failed: WEIXIN_TOKEN is required` | Missing token in .env | Step 5 |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | Missing account ID in .env | Step 5 |
| `Another local Hermes gateway is already using this Weixin token` | Token lock conflict | Step 6 |
| `aiohttp and cryptography are required` | Missing Python deps (non-Docker only) | Step 7 |
| No weixin log lines after startup | Healthy — long-poll is silent when idle | Done |

### Step 3 — Check credentials and env vars

```bash
docker exec hermes-gateway sh -c 'ls -la /opt/data/weixin/accounts/ && grep WEIXIN /opt/data/.env'
```

Expected env vars:
```
WEIXIN_ACCOUNT_ID=<account_id>@im.bot
WEIXIN_TOKEN=<account_id>@im.bot:<token_hex>
```

`WEIXIN_BASE_URL` is optional. If present, it **must** be
`https://ilinkai.weixin.qq.com` — see Step 4.

### Step 4 — Fix wrong WEIXIN_BASE_URL (most common silent failure)

If `data/.env` contains `WEIXIN_BASE_URL=https://ilinkai.wechat.com`
(missing `.qq.com`), the adapter starts but fails on the first
long-poll with `Session expired` or generic API errors.

**Fix:**
```bash
docker exec hermes-gateway sed -i 's|ilinkai.wechat.com|ilinkai.weixin.qq.com|' /opt/data/.env
docker restart hermes-gateway
```

Or simply remove the override (the adapter default is correct):
```bash
docker exec hermes-gateway sed -i '/^WEIXIN_BASE_URL=/d' /opt/data/.env
docker restart hermes-gateway
```

After restart, go back to Step 2 to verify the error is gone.

### Step 5 — Missing or incorrect credentials

If `WEIXIN_ACCOUNT_ID` or `WEIXIN_TOKEN` is missing, the token is
stale, or the session expired (errcode=-14), re-scan the QR code.

**Option A — Web Dashboard (recommended):**
1. Open `http://localhost:9119` in a browser
2. Navigate to **Channels**
3. Find **Weixin / WeChat** → click **Set up with QR**
4. Scan the QR with WeChat on your phone → confirm login
5. The dashboard auto-saves credentials and restarts the gateway

**Option B — CLI wizard inside the container:**
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

### Step 6 — Token lock conflict

```
Another local Hermes gateway is already using this Weixin token
```

Only one gateway instance can long-poll a given token. Stop the other:
```bash
docker ps | grep hermes-gateway
docker stop <other-container>
```

Then restart the intended gateway:
```bash
docker restart hermes-gateway
```

### Step 7 — Missing Python dependencies (non-Docker only)

```bash
pip install aiohttp cryptography
```

The Docker image already includes both — this only affects bare-metal
or venv installs.

## After any fix — verify polling is working

```bash
# Wait 40s for the first long-poll cycle
sleep 40
docker logs hermes-gateway --since 1m 2>&1 | grep -iE "weixin|ilink"
```

A healthy adapter produces **no weixin log lines** after startup — the
long-poll is silent when there are no inbound messages. The absence of
`Session expired` or `ERROR` lines is the success signal.

Re-check the dashboard:
```bash
docker exec hermes-gateway curl -s http://hermes-web:9119/api/status | python3 -m json.tool
```
`weixin.state` should be `connected` with `error_code: null`.

## iLink Bot Identity Limitations

QR login connects an **iLink bot identity** (e.g. `...@im.bot`), not a
fully scriptable personal WeChat account. If the user reports that
group messages don't reach Hermes, this is likely an iLink limitation,
not a Hermes bug:

- The bot identity generally **cannot be invited into ordinary WeChat groups**.
- iLink typically **does not deliver ordinary WeChat group events**.
- `@`-mentioning the personal account used for QR login is NOT the same
  as `@`-mentioning the iLink bot.

The gateway logs a `WARNING` at startup when `WEIXIN_GROUP_POLICY` is
set to anything other than `disabled`.

## Credential File Locations

- **Saved accounts:** `data/weixin/accounts/<account_id>.json`
- **Context tokens:** `data/weixin/accounts/<account_id>.context-tokens.json`
  (safe to delete — rebuilt on next inbound message)
- **Env vars:** `data/.env` — `WEIXIN_ACCOUNT_ID`, `WEIXIN_TOKEN`,
  `WEIXIN_BASE_URL` (optional)

## Output Expectations

- Identify the root cause from the triage steps above.
- Apply the fix (sed for base URL, QR re-scan for expired token, stop
  duplicate container for lock conflict).
- Verify with logs and dashboard status that weixin is `connected`
  with no `Session expired` errors.
- Report what was wrong and what was changed.
- If the token was genuinely expired, remind the user they need to
  scan the QR code with their phone — the agent cannot do this
  automatically.