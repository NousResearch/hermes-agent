# How to: Discord Approval Notification (Mention User)

## Problem
When Hermes needs user approval for dangerous commands, clarify questions, or slash command confirmations on Discord — it sends the prompt into a thread silently. The user doesn't get notified and may miss the request entirely.

## Solution
Hermes now automatically mentions (`<@USER_ID>`) the user in all approval/clarify prompts sent via Discord. This triggers a Discord notification so you see the request immediately.

## What Gets Mentioned

| Feature | Trigger | Notification |
|---------|---------|--------------|
| **Exec Approval** | Dangerous command execution | ✅ Yes — `<@USER_ID>` prepended to embed |
| **Slash Confirm** | Expensive slash commands (`/browser`, `/terminal`, etc.) | ✅ Yes — `<@USER_ID>` prepended to embed |
| **Clarify Prompt** | Agent needs clarification (multi-choice or open-ended) | ✅ Yes — `<@USER_ID>` prepended to embed |
| **Update Prompt** | `hermes update --gateway` needs input | ✅ Yes — `<@USER_ID>` prepended to embed |

## How It Works

### Architecture
```
Agent Thread → Gateway (run.py) → Discord Adapter (adapter.py) → Discord API
     ↓              ↓                    ↓
  approval/    Build <@user_id>   Prepend mention to
  clarify      for Discord        embed description
  callback     only               with ping
```

### Code Flow

**1. Gateway (`gateway/run.py`)** — Detects platform and builds user mention:
```python
# In _approval_notify_sync(), _clarify_callback_sync(), _request_slash_confirm()
_user_mention = None
if source.platform.value == "discord" and source.user_id:
    _user_mention = f"<@{source.user_id}>"
```

**2. Discord Adapter (`plugins/platforms/discord/adapter.py`)** — Prepends mention to embed:
```python
# In send_exec_approval(), send_slash_confirm(), send_clarify(), send_update_prompt()
if user_mention:
    body = f"{user_mention}\n\n{body}"  # Prepend for notification ping
```

### What the User Sees
When a prompt is sent, the embed description looks like:
```
<@123456789012345678>

⚠️ Command Approval Required
[command details...]
```

The `<@...>` syntax triggers Discord's native notification ping.

## Platform Scope

This feature is **Discord-only**. Other platforms (Telegram, Slack, etc.) are unaffected because the mention logic checks `source.platform.value == "discord"` before building the user mention.

## No Configuration Needed

This is a code-level fix — no config changes required. It works automatically for all Discord users once the gateway is restarted after applying these changes.

## Files Changed

| File | Changes |
|------|---------|
| `gateway/run.py` | Added `_user_mention` building in 3 callback functions |
| `plugins/platforms/discord/adapter.py` | Added `user_mention` parameter to 4 send methods |

## Restart Required

After applying these changes, restart the gateway:
```bash
launchctl kickstart gui/$(id -u)/ai.hermes.gateway
```

Or use:
```bash
hermes gateway restart
```

---

*Created for Hermes Agent community sharing.*
