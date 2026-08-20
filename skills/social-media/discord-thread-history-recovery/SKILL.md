---
name: discord-thread-history-recovery
description: "Use when recovering prior Discord channel or thread messages that are missing from the current Hermes prompt context. Fetches history through the Hermes Discord tool or Discord REST, with required bot permissions and safe token handling."
version: 1.0.0
author: Hermes Agent
license: MIT
prerequisites:
  env_vars: [DISCORD_BOT_TOKEN]
metadata:
  hermes:
    tags: [Discord, History, REST API, Gateway, Troubleshooting]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [hermes-agent]
---

# Discord Thread History Recovery

## Overview

Hermes gateway prompts do not always include the full prior Discord channel or thread history. That does **not** necessarily mean the history is unavailable: if the Discord bot can see the channel or thread, Hermes can usually recover messages through the built-in `discord` tool or directly through Discord's REST API.

Use this skill to recover recent Discord messages, attachment metadata, message IDs, timestamps, and author names from the current channel/thread without leaking the bot token or dumping private server history into unnecessary places.

## When to Use

Use this when:

- A Discord user asks you to use earlier messages from the same channel or thread.
- The current prompt context only includes the latest message, but the user expects thread context.
- You need message IDs, timestamps, author names, attachment metadata, reactions, or pinned-state from a Discord conversation.
- A prior assistant message mentioned an artifact/path and you need the surrounding Discord message to continue safely.

Do **not** use this when:

- The request is about a past Hermes session outside the current Discord channel/thread; use `session_search` instead.
- The bot is not in the private thread or cannot view the channel; ask the user to add/mention the bot or grant access.
- The user asks for server administration beyond history recovery; load a Discord/admin skill or use the `discord_admin` tool if available.

## Required Discord Setup and Permissions

### Required for normal Hermes Discord gateway operation

1. `DISCORD_BOT_TOKEN` must be configured in the Hermes environment or `~/.hermes/.env`.
2. The bot must be invited to the server and able to receive events in the channel/thread.
3. **Message Content Intent** must be enabled in the Discord Developer Portal under **Bot → Privileged Gateway Intents** for full text content in normal server channels. Without it, Discord may return message metadata while `content` is empty except for DMs, direct mentions, and messages otherwise exempted by Discord.
4. To use the built-in `discord` tool rather than the REST fallback, the `discord` toolset must be enabled for Discord sessions in Hermes (`hermes tools` or the equivalent platform tools config). If it is not enabled, use the REST fallback below.

### Minimum channel permissions for history recovery

For `GET /channels/<channel_or_thread_id>/messages` and the Hermes `discord(action="fetch_messages")` action, the bot needs, in the target channel/thread:

- **View Channel** — to see the channel/thread.
- **Read Message History** — to fetch existing messages.

For private threads, the bot must also be a participant/member of the private thread, or otherwise have server permissions that let it access that thread. If REST returns `403` or `Unknown Channel`, assume the bot is not in the private thread or lacks channel visibility until proven otherwise.

### Additional permissions only for adjacent actions

These are **not required** just to recover history, but are commonly needed when the user expects the bot to keep working in Discord:

| Capability | Permission / setting |
| --- | --- |
| Reply in a text channel | Send Messages |
| Reply in a thread | Send Messages in Threads |
| Create public threads / Hermes auto-threading | Create Public Threads, plus Send Messages in Threads |
| Upload images/documents/audio | Attach Files |
| Add reactions / typing niceties | Add Reactions / Send Messages as applicable |
| Pin/unpin messages | Manage Messages |
| Manage roles | Manage Roles, with the bot role above target roles |
| Search/list guild members by name or role | Server Members Intent in Developer Portal; channel permissions alone are not enough |

Do **not** recommend Administrator as the default fix. The narrow minimum for history recovery is **View Channel + Read Message History** plus Message Content Intent when full message text is needed.

## Recovery Workflow

### 1. Identify the Discord channel or thread ID

Prefer IDs already present in the current session context. Discord threads are queried through the same messages endpoint as channels, using the thread ID as `channel_id`.

Common sources:

- Current session context line such as `thread: 1501314808090464297`.
- A Hermes Discord IDs block with `Channel`, `Thread`, or `Parent channel`.
- Gateway logs containing inbound events, for example:
  ```text
  gateway.run: inbound message: platform=discord ... chat=<thread_or_channel_id> ...
  ```

Use the thread ID if the conversation is inside a thread. Do not substitute the guild/server ID or parent channel ID unless the conversation is actually in the parent channel.

### 2. Prefer the built-in Discord tool when available

If the live session exposes the `discord` tool, use it first. It already uses the configured bot token and returns structured JSON.

```text
discord(action="fetch_messages", channel_id="<channel_or_thread_id>", limit=50)
```

Notes:

- Discord returns newest-first at the API layer; reverse the messages before human summarization if needed.
- Use `before=<oldest_message_id_seen>` to page backward.
- If the tool says `content` is empty and metadata is present, check Message Content Intent.
- If it returns `403`, check View Channel / Read Message History and private-thread membership.

### 3. REST fallback when the Discord tool is unavailable

Use direct REST only when the `discord` tool is not available in the session. Load the token silently and never print it.

```python
import json
import os
import pathlib
import urllib.request

channel_id = "<channel_or_thread_id>"
limit = 50

hermes_home = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
for env_path in [hermes_home / ".env", pathlib.Path.home() / ".hermes/.env"]:
    if not env_path.exists():
        continue
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

token = os.getenv("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_TOKEN")
if not token:
    raise SystemExit("No Discord bot token found in environment or Hermes .env")

req = urllib.request.Request(
    f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={limit}",
    headers={
        "Authorization": f"Bot {token}",
        "User-Agent": "Hermes-Agent Discord history recovery",
    },
)
messages = json.load(urllib.request.urlopen(req, timeout=20))
messages = list(reversed(messages))

outdir = hermes_home / "workspace" / "discord-history-recovery"
outdir.mkdir(parents=True, exist_ok=True)

clean = []
for msg in messages:
    author = msg.get("author", {})
    clean.append({
        "id": msg.get("id"),
        "timestamp": msg.get("timestamp"),
        "author": author.get("global_name") or author.get("username") or author.get("id"),
        "author_id": author.get("id"),
        "content": msg.get("content", ""),
        "attachments": [
            {
                "filename": attachment.get("filename"),
                "content_type": attachment.get("content_type"),
                "size": attachment.get("size"),
                "url": attachment.get("url"),
            }
            for attachment in msg.get("attachments", [])
        ],
        "embed_count": len(msg.get("embeds", [])),
        "pinned": msg.get("pinned", False),
    })

json_path = outdir / f"{channel_id}-messages.json"
md_path = outdir / f"{channel_id}-messages.md"
json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

parts = []
for msg in clean:
    parts.append(f"## {msg['timestamp']} — {msg['author']} — {msg['id']}\n")
    if msg["content"]:
        parts.append(msg["content"] + "\n")
    if msg["attachments"]:
        parts.append(
            "Attachments: "
            + ", ".join(
                f"{a['filename']} ({a['content_type']}, {a['size']} bytes)"
                for a in msg["attachments"]
            )
            + "\n"
        )
    if msg["embed_count"]:
        parts.append(f"Embeds: {msg['embed_count']}\n")
    if msg["pinned"]:
        parts.append("Pinned: true\n")
    parts.append("\n")
md_path.write_text("\n".join(parts), encoding="utf-8")

print(json.dumps({
    "ok": True,
    "channel_id": channel_id,
    "messages": len(clean),
    "json_path": str(json_path),
    "markdown_path": str(md_path),
}, indent=2))
```

### 4. Summarize only what is needed

After recovery, report back with:

- Whether recovery worked.
- The channel/thread ID queried.
- Number of messages fetched.
- Where JSON/Markdown history was saved if files were written.
- The specific prior context needed to answer the user.

Do not paste large channel-history dumps into the conversation unless the user explicitly asks for raw history.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `401 Unauthorized` | Bad/missing bot token | Verify `DISCORD_BOT_TOKEN`; never print it |
| `403 Forbidden` | Missing View Channel / Read Message History, or bot not in private thread | Grant channel permission or add bot to thread |
| `404 Unknown Channel` | Wrong ID, bot cannot see channel/thread, or private thread membership issue | Re-check thread/channel ID and bot access |
| Messages returned but `content` is empty | Message Content Intent missing or Discord content restrictions | Enable Message Content Intent; DMs/mentions may still show content |
| Only old parent-channel messages show up | Used parent channel ID instead of thread ID | Query the thread ID directly |
| No recent inbound logs for the expected thread | Bot never received the Discord event | Mention/add the actual bot user, verify allowed channels, or restart gateway after config changes |

## Safety Rules

- Never print, paste, or log the Discord bot token.
- Treat recovered Discord history as private workspace context.
- Do not expose attachment URLs unless the user needs them; Discord CDN URLs can reveal private content to anyone with access while valid.
- Confirm before taking external/public actions based on recovered messages.
- Keep summaries tight; recover history to answer the task, not to create a permanent transcript dump.

## Verification Checklist

- [ ] Correct Discord thread/channel ID identified.
- [ ] Bot token loaded silently, without printing `.env` or auth headers.
- [ ] Bot has View Channel and Read Message History for the target channel/thread.
- [ ] Message Content Intent status considered if `content` is empty.
- [ ] REST/tool fetch succeeded with expected message count.
- [ ] Messages normalized in chronological order before summarization.
- [ ] JSON/Markdown output saved only when useful, under Hermes workspace.
- [ ] Relevant prior context summarized accurately without over-sharing private history.
