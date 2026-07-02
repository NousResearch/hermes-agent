# Messaging-Adapter Live Smoke Plan

## Overview

This document describes the credential matrix and smoke steps for validating
Hermes messaging-platform adapters against the RuntimeExecutor + run-control
routes (`/v1/runs/{run_id}/stop`, `/approval`, `/clarify`).

Each adapter requires platform-specific credentials (bot tokens, app secrets,
webhook URLs) that are not available in the deterministic CI environment.
This plan documents what is needed so an operator with real credentials can
run the smoke.

---

## Supported Adapters (discovered in `gateway/platforms/`)

| Platform       | Adapter file             | Credential type         |
|----------------|--------------------------|-------------------------|
| Telegram       | `telegram/` (subdir)     | `TELEGRAM_BOT_TOKEN`    |
| Discord        | `discord/` (subdir)      | `DISCORD_BOT_TOKEN`     |
| Slack          | `slack/` (subdir)        | `SLACK_BOT_TOKEN`       |
| Signal         | `signal.py`              | `SIGNAL_PHONE_NUMBER`   |
| WhatsApp Cloud | `whatsapp_cloud.py`      | `WHATSAPP_TOKEN`        |
| Matrix         | `matrix/` (subdir)       | `MATRIX_ACCESS_TOKEN`   |
| Mattermost     | `mattermost/` (subdir)   | `MATTERMOST_TOKEN`      |
| QQ Bot         | `qqbot/` (subdir)        | `QQBOT_APP_TOKEN`       |
| WeChat (WeCom) | `wecom.py`               | `WECOM_CORP_ID`+`SECRET`|
| WeiXin         | `weixin.py`              | `WEIXIN_APP_TOKEN`      |
| Feishu         | `feishu/` (subdir)       | `FEISHU_APP_TOKEN`      |
| DingTalk       | `dingtalk/` (subdir)     | `DINGTALK_TOKEN`        |
| BlueBubbles    | `bluebubbles.py`         | `BLUEBUBBLES_API_KEY`   |
| YuanBao        | `yuanbao.py`             | `YUANBAO_TOKEN`         |
| Email          | `email/` (subdir)        | SMTP/IMAP credentials   |
| SMS            | `sms/` (subdir)          | Twilio/other creds      |
| Webhook        | `webhook.py`             | `WEBHOOK_SECRET`        |
| MS Graph       | `msgraph_webhook.py`     | `MSGRAH_*` OAuth token  |

---

## Credential Matrix

All tokens belong in `~/.hermes/.env` (never in config.yaml or committed).

| Env var                    | Platform     | Minimal permission               |
|----------------------------|--------------|----------------------------------|
| `TELEGRAM_BOT_TOKEN`       | Telegram     | Send messages, read group chats  |
| `DISCORD_BOT_TOKEN`        | Discord      | Send messages, read message hist |
| `SLACK_BOT_TOKEN`          | Slack        | `chat:write`, `channels:history` |
| `SIGNAL_PHONE_NUMBER`      | Signal       | Signal registered phone number   |
| `WHATSAPP_TOKEN`           | WhatsApp     | WhatsApp Business API token      |
| `MATRIX_ACCESS_TOKEN`      | Matrix       | Matrix account access token      |
| `MATTERMOST_TOKEN`         | Mattermost   | Personal access token            |
| `QQBOT_APP_TOKEN`          | QQ Bot       | QQ bot application token         |
| `WECOM_CORP_ID`+`_SECRET`  | WeCom        | Corp ID + agent secret           |
| `WEIXIN_APP_TOKEN`         | WeiXin       | WeChat official account token    |
| `FEISHU_APP_TOKEN`         | Feishu       | Feishu app token                 |
| `DINGTALK_TOKEN`           | DingTalk     | DingTalk custom bot token        |
| `BLUEBUBBLES_API_KEY`      | BlueBubbles  | BlueBubbles server API key       |
| `YUANBAO_TOKEN`            | YuanBao      | YuanBao platform token           |

---

## Safe Test Channel Setup

For each adapter, create a private test channel/chat with only the bot and
the operator. Do NOT use production channels.

1. **Telegram**: Create a private group, add the bot, make it admin (needed to
   read messages).
2. **Discord**: Create a private server, add the bot with appropriate intents.
3. **Slack**: Create a private channel, invite the bot.
4. **Signal**: Use a secondary phone number registered for Signal.
5. **WhatsApp**: Use the WhatsApp Business API sandbox number.
6. **Others**: Follow platform-specific guidance for sandbox/test mode.

---

## Smoke Steps

### Prerequisite

Ensure the Agent runtime server + WebUI agent-runs server are running:

```bash
# Start Agent runtime standalone server
cd hermes-agent
DEEPSEEK_API_KEY=<key> python3 scripts/standalone_runtime_server.py --port 8642

# Start WebUI in agent-runs mode (separate terminal)
cd hermes-webui
HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642 \
HERMES_WEBUI_PASSWORD=test-password \
HERMES_WEBUI_PORT=8789 \
python3 server.py
```

### 1. Gateway adapter connect

```bash
cd hermes-agent
HERMES_GATEWAY=1 python3 -c "
from gateway.run import start_gateway
import asyncio
asyncio.run(start_gateway())
"
```

Verify the adapter connects:
- Log should show `Platform X connected` without errors.
- Bot should appear online on the platform.

### 2. Runtime run visible on platform

Send a direct message to the bot containing text. The gateway routes it through
`RuntimeControlBridge`, which creates a `/v1/runs` run with `execute:true`.

Verify:
- A run is created with `status=queued` or `running`.
- The Agent RuntimeExecutor processes it.
- The bot responds in the platform chat.

### 3. Run status/events via platform

The gateway should mirror run status changes to the platform chat.

- When the run reaches `completed`, the platform should receive a final
  message (the agent's response).
- Events should be streamed if the platform supports it.

### 4. Approval / deny via platform

If the agent requests approval (e.g. for a dangerous command):

1. Verify platform receives the approval prompt.
2. Respond with `/approve` or `/deny`.
3. Verify the approval is resolved via `RuntimeControlBridge.resolve_approval()`.
4. Verify the pending action is removed and a `approval.resolved` event is
   appended.

### 5. Stop / cancel via platform

1. Send a long-running request to the bot.
2. Send `/stop` in the platform chat.
3. Verify the run is cancelled via `POST /v1/runs/{run_id}/stop`.
4. Verify the bot confirms cancellation.

### 6. Slash-command state sync (runtime)

Test `/approve`, `/deny`, and `/new` slash commands:

- `/approve all` should resolve all pending approvals.
- `/deny <id>` should deny a specific approval.
- `/new` should create a new session.

---

## Expected Runtime Run Visibility

When `RuntimeControlBridge` is enabled:
- Every platform message creates a `RunManager` run.
- Run status/events are accessible via `GET /v1/runs/{run_id}` and
  `GET /v1/runs/{run_id}/events`.
- The WebUI `agent-runs` adapter can proxy these status/events.
- The platform adapter can poll or be notified of status changes.

---

## Expected Approval/Deny Slash-Command Behavior

1. Agent requests approval → `approval.requested` event appended.
2. User sends `/approve <id>` or `/approve all`.
3. Gateway dispatches to `RuntimeControlBridge.resolve_approval()` or
   `RunManager.resolve_approval()`.
4. `approval.resolved` event appended.
5. Pending approval ID removed from run status.
6. If no pending approvals/clarifies remain, run status resets to `queued`.

---

## Expected Stop/Cancel Behavior

1. User sends `/stop`.
2. Gateway dispatches to `POST /v1/runs/{run_id}/stop`.
3. Executor cancels the agent execution.
4. Run transitions to `cancelled` (terminal).
5. `run.cancelled` event appended.
6. Platform receives cancellation confirmation.

---

## Cleanup / Revocation Steps

After smoke completion:

1. **Disconnect adapter**: Stop the gateway process (Ctrl-C).
2. **Remove bot from test channels** (or delete test channels).
3. **Revoke bot tokens** if they were created specifically for the test:
   - Telegram: `BotFather → /mybots → <bot> → API Token → Revoke`
   - Discord: Developer Portal → Bot → Reset Token
   - Slack: api.slack.com → Apps → <app> → OAuth & Permissions → Reinstall
4. **Verify tokens removed from `~/.hermes/.env`** (or commented out).

---

## Secret Redaction Requirements

- Never print token values in logs, smoke output, errors, or reports.
- Never include token values in this document or any committed file.
- Never include raw token values in run events or approval payloads.
- The `redact_secrets()` utility in `gateway/runtime/run_manager.py`
  automatically redacts known secret patterns from event payloads.
- Verify redaction by checking event payloads for `[REDACTED]` instead
  of raw secrets.
