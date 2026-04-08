---
name: slack-api
description: Interact with Slack workspaces via the Bot Token and Web API — read channels, send messages, scan for mentions, resolve users, manage DMs. Use when the user asks to read/send Slack messages, search channels, or audit mentions.
tags: [slack, messaging, api, workspace]
---

# Slack API via Bot Token

## When to Use
- User asks to read Slack channel messages
- User asks to send messages/DMs via Slack
- User asks to scan channels for mentions or keywords
- User asks to check bot permissions or channel access

## Setup
Bot token is in `~/.hermes/.env` as `SLACK_BOT_TOKEN`. Load it:
```bash
TOKEN=$(grep SLACK_BOT_TOKEN ~/.hermes/.env | cut -d= -f2)
```

## Key Patterns

### Check Current Bot Scopes
The `x-oauth-scopes` response header reveals all granted scopes — much more reliable than trial-and-error:
```bash
curl -sI -H "Authorization: Bearer $TOKEN" "https://slack.com/api/auth.test" | grep -i x-oauth-scopes
```

### List Public Channels
```bash
curl -s -H "Authorization: Bearer $TOKEN" "https://slack.com/api/conversations.list?types=public_channel&limit=200&exclude_archived=true"
```
Paginate with `cursor` from `response_metadata.next_cursor`.

### Join a Channel (requires `channels:join`)
```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"channel":"CHANNEL_ID"}' "https://slack.com/api/conversations.join"
```
Bot must join before reading history. Joining via API may not show the bot in the member list visibly.

### Read Channel History (requires `channels:history` + bot in channel)
```bash
curl -s -H "Authorization: Bearer $TOKEN" "https://slack.com/api/conversations.history?channel=CHANNEL_ID&limit=100"
```

### Send a Message (requires `chat:write`)
```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"channel":"CHANNEL_ID","text":"Hello!"}' "https://slack.com/api/chat.postMessage"
```

### Open a DM with a User (requires `im:write`)
```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"users":"USER_ID"}' "https://slack.com/api/conversations.open"
```
Returns a DM channel ID. For group DMs, comma-separate user IDs.

### Resolve User IDs to Names (requires `users:read`)
```bash
curl -s -H "Authorization: Bearer $TOKEN" "https://slack.com/api/users.info?user=USER_ID"
```
Access `user.real_name` or `user.name` from response.

### List All Users
```bash
curl -s -H "Authorization: Bearer $TOKEN" "https://slack.com/api/users.list?limit=200"
```

### Search Messages
`search.messages` requires a **User token** (not bot token) — it returns `not_allowed_token_type` with bot tokens. Instead, iterate channels and grep history.

## Scanning All Channels for Mentions (Pattern)
Use `execute_code` for reliability (avoids terminal timeout on large workspaces):
1. `conversations.list` with pagination to get all channels
2. `conversations.join` each channel
3. `conversations.history` with limit=200 per channel
4. Filter messages containing `<@USER_ID>`
5. `users.info` to resolve mentioned user IDs to names
6. Rate limit: `time.sleep(0.2-0.3)` between API calls

## Recommended Bot Scopes
Comprehensive set for a fully capable bot:
- `app_mentions:read` — respond to @mentions
- `channels:history` — read public channel messages
- `channels:join` — join public channels
- `channels:read` — list public channels
- `chat:write` — send messages
- `chat:write.public` — post to channels without joining
- `connections:write` — Socket Mode
- `files:read` / `files:write` — view and upload files
- `groups:history` / `groups:read` — private channel access (when invited)
- `im:history` / `im:read` / `im:write` — DM access
- `mpim:history` / `mpim:read` — group DM access
- `pins:read` / `pins:write` — pinned messages
- `reactions:read` / `reactions:write` — emoji reactions
- `users:read` / `users:read.email` — user lookups

## Daily Mentions Briefing (Cron Pattern)
A reusable script exists at `~/.hermes/scripts/slack_mentions.py` that:
1. Scans all public channels for `<@USER_ID>` mentions in the last 24 hours
2. Checks DMs and group DMs for new messages (skips own messages and subtypes)
3. Resolves user IDs to real names
4. Outputs structured text for agent summarization

Set up as a cron job with `script: slack_mentions.py` and a prompt that categorizes mentions into questions, action items, FYIs, and kudos. Deliver to `slack` for Slack DM delivery.

### Scanning DMs and Group DMs
Use `conversations.list?types=im` and `conversations.list?types=mpim` to enumerate DM/group DM channels, then `conversations.history` on each. Filter out the bot's own user ID and subtype messages (joins, etc.).

### Read Thread Replies (requires `channels:history`)
```bash
curl -s -H "Authorization: Bearer $TOKEN" "https://slack.com/api/conversations.replies?channel=CHANNEL_ID&ts=THREAD_TS&limit=100"
```
Returns all messages in a thread given the parent message's `ts` (thread_ts).

**IMPORTANT — Gateway thread context:** When running as a Slack gateway bot and a user @mentions you in a thread, the gateway only passes the single mention message — it does NOT include the rest of the thread. If the user's request references "the message above", "this thread", or seems to lack context, **use your tools to call `conversations.replies`** to fetch the full thread. Do not say you can't see thread messages — you can always fetch them via the API.

The thread_ts is typically available from the message metadata. If not, use `conversations.history` on the channel and look for the parent message.

### Delete a Bot Message (requires `chat:write` — bot can only delete its own)
```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"channel":"CHANNEL_ID","ts":"MESSAGE_TIMESTAMP"}' "https://slack.com/api/chat.delete"
```
To find a message to delete, use `conversations.history` and match on text content or bot_id.

## Diagnosing Socket Mode Hijacking (OpenClaw / Other Bots)
If the gateway stops responding on Slack but `hermes gateway status` shows it running:
1. Check `gateway.log` — if no recent Slack entries, the socket was stolen
2. On this server: `ps aux | grep -i "openclaw\|claw\|node.*slack" | grep -v grep`
3. Check macOS: `launchctl list | grep -i claw` — OpenClaw installs as `ai.openclaw.gateway` launchd service
4. To remove on macOS:
   ```bash
   kill <PID>                                          # kill the process
   launchctl remove ai.openclaw.gateway                # remove the service (more reliable than bootout)
   rm ~/Library/LaunchAgents/ai.openclaw.gateway.plist  # prevent restart on reboot
   launchctl list | grep claw                           # verify it's gone
   ```
   Note: `launchctl bootout` often fails with I/O errors — use `launchctl remove` instead.
5. Then restart the Hermes gateway to reclaim the socket connection
6. OpenClaw is installed via Homebrew (`which openclaw` → `/opt/homebrew/bin/openclaw`), config in `~/.openclaw/`
7. **`ps | grep claw` false positive** — `grep` matches itself in the process list. If the only match is `grep --color=auto claw`, there is no OpenClaw process running.

## Downloading and Forwarding Slack Files (Images, Documents)

Slack file URLs (`files.slack.com/files-pri/...`) are **private** — they require the bot token to access. They cannot be embedded directly in external services (Linear, GitHub, etc.).

### Pattern: Slack Files → Linear Issues
When creating Linear issues from Slack threads that contain images:

1. **Fetch thread** via `conversations.replies` to get all messages with `files` arrays
2. **Download each file** using the bot token for auth:
```python
req = urllib.request.Request(file["url_private"], 
    headers={"Authorization": f"Bearer {TOKEN}"})
data = urllib.request.urlopen(req).read()
with open(local_path, "wb") as f:
    f.write(data)
```
3. **Rename files** to remove spaces (breaks CLI tools): `fname.replace(" ", "_")`
4. **Upload to Linear** via CLI: `linear issue attach ISSUE-ID /path/to/file -t "Title"`
5. **Get attachment URLs** from `linear issue view ISSUE-ID --json` → `attachments[].url`
6. **Embed in description** as markdown images: `![description](attachment_url)`
7. **Update the issue** with `linear issue update ISSUE-ID --description "..."`

### Key Details
- Slack file objects have: `url_private` (needs auth), `name`, `mimetype`, `filetype`
- Each Slack message can have multiple files in its `files` array
- Linear attachments get public URLs (`public.linear.app/...`) after upload
- `linear issue attach` fails with spaces in filenames — always sanitize first
- Use `execute_code` for bulk operations to avoid terminal timeouts

## Pitfalls
1. **`search.messages` is user-token only** — bot tokens get `not_allowed_token_type`. Must iterate channels manually.
2. **Bot must join channel before reading history** — otherwise get `not_in_channel` error (not a scope issue).
3. **`conversations.open` for DMs requires `im:write`** — easy to miss, not included in basic bot templates.
4. **Bot profile (name/image) can't be set via API** — must be configured in Slack App settings under Display Information.
5. **Rate limits** — Slack Tier 3 methods allow ~50 req/min. Add 200-300ms delays when iterating many channels.
6. **Large workspaces timeout in terminal** — use `execute_code` instead for bulk operations across 100+ channels.
7. **Scope changes require app reinstall** — after adding scopes in OAuth & Permissions, must reinstall from "Install App" page (not OAuth page, which may fail with `redirect_uri` error for Socket Mode apps).
8. **Socket Mode only allows ONE active connection per app token** — if another process (e.g., an old OpenClaw bot) connects with the same `SLACK_APP_TOKEN`, it silently steals the connection. The gateway log will stop showing Slack messages but the process stays alive. Diagnose by checking if `gateway.log` has recent Slack entries. Fix: find and kill the other process, or create a new Slack app with fresh tokens.
9. **Slack scope picker is virtualized** — the dropdown only renders visible items. Scopes like `channels:history` won't appear until you scroll up or type in the search filter. Always use the search box.
10. **Check bot scopes via response header** — `curl -sI ... auth.test | grep x-oauth-scopes` is the fastest way to see what scopes are active, rather than checking the Slack app settings page.
