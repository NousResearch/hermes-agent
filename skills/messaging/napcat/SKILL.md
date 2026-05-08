---
name: napcat
description: NapCat / QQ (OneBot 11) operating manual. Load this whenever you are responding through a NapCat-backed QQ chat OR planning to use the napcat_call tool. Forces NapCat output style (plain text, no Markdown, concise) and documents how to send images/voice/files, fetch group history, manage members, and modify the QQ profile via napcat_call.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [napcat, qq, onebot, messaging, gateway]
    homepage: https://napcat.napneko.icu/
    related_skills: []
prerequisites:
  env_vars: [NAPCAT_TOKEN]
---

# NapCat / QQ (OneBot 11)

You are talking through **NapCat**, a OneBot 11 implementation that drives a real
QQ account. Your replies are delivered as ordinary QQ messages and your tools
operate on a real friend list / group list. Read this whole skill before doing
anything.

---

## 1. Output style — read this first

QQ does **not** render Markdown. Whatever you type is delivered verbatim. So:

- **No `**bold**`, no `*italic*`, no `__underline__`**.
- **No `#` headings**, no `>` quotes, no `---` separators.
- **No `- ` / `* ` bullets**, no `1.` numbered lists styled like Markdown — use
  plain-text numbering (`1) `, `① `, or just sentences) when truly needed.
- **No fenced code blocks** (` ``` `) and no inline backticks. If you must show
  code or a command, paste it as-is on its own line.
- **No table syntax** (`|---|`).
- **No `[label](url)` links** — paste the bare URL.
- **No HTML / emoji shortcodes** like `:smile:` (real emoji characters are fine).
- **Do not paste this skill's examples into QQ**. Example paths like
  `MEDIA:https://example.com/...`, `MEDIA:/absolute/path`, and
  `/path/to/...` are documentation only; only output `MEDIA:` when you are
  actually sending a real reachable file/URL.

Length rules:

- Default to one or two short sentences. QQ is a chat client, not a wiki.
- Only expand when the user explicitly asks for detail (“详细说说”, “展开”,
  “一步步”). Even then, prefer compact prose over long bullet lists.
- Don’t restate the question. Don’t pad with “希望对你有帮助” style filler.
- Don’t prefix with “好的，”/“当然可以！”/“As an AI…” boilerplate.
- If you don’t know, say so in one short sentence.

When the user asks for something a list naturally fits (e.g. group members,
schedule), use plain numbered prose: `1) 张三  2) 李四  3) 王五`.

---

## 2. When this skill applies

- Hermes session env `HERMES_SESSION_PLATFORM` is `napcat`, **or**
- you are about to invoke the `napcat_call` tool, **or**
- you are using `send_message` with a `napcat:` target.

If you're replying as the agent on a different platform (Telegram, Discord,
Slack, CLI) — Markdown is fine there; ignore this skill.

### 2.1 Group context rule — read history before answering group questions

When the incoming NapCat message is from a QQ group and the user is asking a
question, making a request, or referring to previous discussion, **first call**
`napcat_call("get_group_msg_history", {"group_id": <current_group_id>, "count": 30})`
before composing the final answer.

Use the current group id from the message source / raw NapCat event. If the
chat id is prefixed like `group:977185513`, strip `group:` and pass the numeric
group id. Do not ask the user for the group id when it is available in context.

Use the returned history only as private context to understand pronouns,
recent topics, attachments, and who said what. Do not dump raw history back to
the group unless the user explicitly asks for history or a summary.

If `get_group_msg_history` fails or `napcat_call` is unavailable, continue with
the visible message context and, only when it affects the answer, say briefly
that group history could not be read.

---

## 3. Reply paths at a glance

| What you want to do                  | How to do it                                                                 |
|--------------------------------------|------------------------------------------------------------------------------|
| Send a plain text reply              | Just produce text in your final answer. The gateway sends it via `send_msg`. |
| Send an image URL                    | Embed `MEDIA:https://example.com/image.png` in your reply. Gateway calls `send_image_file`. |
| Send voice / video URL               | `MEDIA:https://example.com/audio.ogg` → `send_voice`; `.mp4` → `send_video`. |
| Upload a non-media file              | Prefer final-answer file attachment tags if available; otherwise use `napcat_call("upload_group_file", {...})` or `upload_private_file`. |
| Anything else (recall, kick, like…)  | `napcat_call(action, params)`.                                               |

> Prefer `MEDIA:` tags over raw `napcat_call("send_group_msg", …)` for media —
> the gateway handles chat-id routing, chunking, captions, and reply threading
> for you.

### 3.1 Deployment/path rule — Hermes and NapCat are not the same filesystem

Assume Hermes and NapCat may be running on different machines. A local path that
exists on the Hermes host is not automatically readable by the NapCat/QQ host.
For outgoing media and file attachments, the Hermes gateway can stream-upload
an existing Hermes-local file to NapCat first when NapCat supports
`upload_file_stream` (NapCat v4.8.115+), then send the returned NapCat-local
temp path through `send_*_msg` or `upload_*_file`. Remote URLs, `base64://...`,
and `data:` are still passed through directly. Do not invent or echo placeholder
local paths like `/path/to/...`, `/abs/path/to/...`, or `/home/user/cache/...`
in final replies.

If you need to send a Hermes-side local file:

1. If the file really exists on Hermes, you may use `MEDIA:/absolute/path` and
   let the gateway stream-upload it to NapCat before sending.
2. Prefer creating a public or LAN-reachable HTTP(S) URL for the file, then use
   `MEDIA:https://...`.
3. For small media, encode the file as base64 and use NapCat's supported
   `base64://...` form if size limits allow it.
4. If both hosts share storage, copy the file into the shared mount and use the
   path exactly as NapCat sees it on the NapCat machine.
5. If none of those are available, do not send a `MEDIA:` tag. Reply in text
   that the file is only local to Hermes and needs to be exposed or copied to
   the NapCat host first.

If a previous media send failed and the conversation contains a
`[Hermes gateway delivery feedback]` message, treat it as authoritative. Fix
the path/URL strategy before trying to send the same media again. If the
feedback says `upload_file_stream` is unsupported or failed, switch to an
HTTP(S) URL, `base64://...`, or a NapCat-readable shared path.

---

## 4. The `napcat_call` tool

```
napcat_call(action: str, params: object) -> JSON
```

- `action` — any OneBot 11 endpoint name, e.g. `delete_msg`, `get_group_msg_history`, `set_qq_profile`.
- `params` — JSON object whose shape depends on the action. See the catalog in §5.

Successful response shape:

```
{ "success": true, "action": "...", "retcode": 0, "data": { ... } }
```

Failure shape:

```
{ "error": "NapCat <action> failed (retcode=<n>): <msg>", "retcode": <n>, "raw": {...} }
```

The tool is only available when the gateway is running and a NapCat client is
connected. If it returns an error about no connection, ask the user to start
`hermes gateway run` and connect NapCat — don't keep retrying.

### Useful invariants

- QQ user IDs (`user_id`, `target_id`) and group IDs (`group_id`) are integers.
  The tool will coerce numeric strings, but pass ints when you can.
- `message_id` values are strings — pass them through as-is.
- Don't pass `self_id`. The adapter manages that.
- Group operations require the bot account to actually be a member (and an
  admin/owner for moderation actions). `set_group_kick`, `set_group_ban`,
  `set_group_admin`, `set_group_name` all fail silently with `retcode != 0`
  for non-admins.

---

## 5. OneBot 11 action catalog

Pick the right `action` and pass the listed params. Anything not listed is
still callable through `napcat_call` if the user asks — refer to the NapCat
docs (https://napcat.napneko.icu/) for exotic endpoints.

### 5.1 Messages

| Action                       | Params                                                                                                | Returns                                |
|------------------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------|
| `send_private_msg`           | `user_id`, `message` (string OR segment array), `auto_escape?`                                        | `{message_id}`                         |
| `send_group_msg`             | `group_id`, `message`, `auto_escape?`                                                                 | `{message_id}`                         |
| `send_msg`                   | `message_type` (`private`/`group`), `user_id?`, `group_id?`, `message`                                | `{message_id}`                         |
| `delete_msg`                 | `message_id`                                                                                          | `{}`                                   |
| `get_msg`                    | `message_id`                                                                                          | full message dict (sender, segments…)  |
| `get_forward_msg`            | `id`                                                                                                  | `{messages: [...]}`                    |
| `get_group_msg_history`      | `group_id`, `count?` (default 20), `message_seq?` (paginate from this seq)                            | `{messages: [...]}`                    |
| `send_group_forward_msg`     | `group_id`, `messages` (array of `{type:"node", data:{name, uin, content}}`)                          | `{message_id}`                         |
| `send_private_forward_msg`   | `user_id`, `messages`                                                                                 | `{message_id}`                         |
| `mark_msg_as_read` *(NapCat)*| `message_id`                                                                                          | `{}`                                   |
| `set_msg_emoji_like` *(NapCat)*| `message_id`, `emoji_id` (string, e.g. `"76"` for thumbs-up)                                        | `{}`                                   |

Recall pattern: `delete_msg` only works on messages the bot itself sent
(within ~2 minutes for group messages, longer for private).

### 5.2 Group management

| Action                     | Params                                                          | Notes                                  |
|----------------------------|-----------------------------------------------------------------|----------------------------------------|
| `get_group_info`           | `group_id`, `no_cache?`                                         | name, member count, max members        |
| `get_group_list`           | (none)                                                          | array of groups the bot is in          |
| `get_group_member_info`    | `group_id`, `user_id`, `no_cache?`                              | nickname, card, role, join time        |
| `get_group_member_list`    | `group_id`                                                      | array of member dicts                  |
| `set_group_card`           | `group_id`, `user_id`, `card`                                   | empty `card` clears nickname           |
| `set_group_kick`           | `group_id`, `user_id`, `reject_add_request?`                    | requires admin                         |
| `set_group_ban`            | `group_id`, `user_id`, `duration` (seconds, 0=unmute)           | requires admin                         |
| `set_group_whole_ban`      | `group_id`, `enable` (bool)                                     | requires admin                         |
| `set_group_admin`          | `group_id`, `user_id`, `enable` (bool)                          | requires owner                         |
| `set_group_name`           | `group_id`, `group_name`                                        | requires admin                         |
| `set_group_leave`          | `group_id`, `is_dismiss?` (owner only)                          | bot leaves the group                   |
| `set_group_special_title`  | `group_id`, `user_id`, `special_title`, `duration?`             | owner only                             |
| `set_group_add_request`    | `flag`, `sub_type` (`add`/`invite`), `approve` (bool), `reason?`| respond to join requests               |

### 5.3 Friends

| Action                  | Params                                              |
|-------------------------|-----------------------------------------------------|
| `get_friend_list`       | (none)                                              |
| `send_like`             | `user_id`, `times?` (1–20)                          |
| `set_friend_add_request`| `flag`, `approve` (bool), `remark?`                 |
| `delete_friend`         | `user_id`                                           |

### 5.4 Files & media (low level)

Prefer `MEDIA:` tags over manual `napcat_call("send_*_msg", …)` for images,
voice, and video. Use these only when you specifically need NapCat actions:

| Action                  | Params                                                  | Use case                            |
|-------------------------|---------------------------------------------------------|-------------------------------------|
| `get_image`             | `file` (file_id or URL)                                 | resolve an inbound `[图片:…]` to a local cached path |
| `get_record`            | `file`, `out_format?` (`mp3`/`amr`/`wav`)               | resolve an inbound `[语音:…]`        |
| `get_file`              | `file_id`                                               | resolve `[文件:…]` to a local path   |
| `download_file`         | `url`, `headers?`, `thread_count?`                      | pre-cache a remote file              |
| `upload_file_stream` *(NapCat v4.8.115+)* | chunked stream params; gateway normally handles this | transfer Hermes-local files to NapCat |
| `upload_group_file`     | `group_id`, `file` (NapCat-readable path), `name`, `folder?` | post a file to a group; gateway stream-uploads Hermes-local paths first |
| `upload_private_file`   | `user_id`, `file` (NapCat-readable path), `name`        | DM a file to a user; gateway stream-uploads Hermes-local paths first |
| `can_send_image`        | (none)                                                  | capability probe                     |
| `can_send_record`       | (none)                                                  | capability probe                     |

### 5.5 Profile

| Action               | Params                                                                                                  |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `set_qq_profile`     | `nickname?`, `company?`, `email?`, `college?`, `personal_note?`                                         |
| `get_login_info`     | (none) — returns the bot's own `{user_id, nickname}`                                                    |
| `get_stranger_info`  | `user_id`, `no_cache?`                                                                                  |

### 5.6 System / housekeeping

| Action               | Params         | Notes                                  |
|----------------------|----------------|----------------------------------------|
| `get_status`         | (none)         | online state, app version              |
| `get_version_info`   | (none)         | NapCat / OneBot version                |
| `clean_cache`        | (none)         | clears NapCat's local cache            |
| `ocr_image`          | `image`        | NapCat extension; OCRs a file_id       |
| `send_group_ai_record` *(NapCat)* | `group_id`, `character`, `text` | AI TTS into a group as voice |

---

## 6. Sending images — best practice

Embed a `MEDIA:` tag in your final reply. The gateway extracts it and calls
`send_image_file` automatically:

```
MEDIA:https://example.com/meme.png
```

Multiple media in one reply is fine; each one becomes a separate QQ message.
A short caption written before the tag is sent first as text. Don't wrap
`MEDIA:` paths in backticks or quotes — the extractor handles bare paths.

If the image was created or downloaded by Hermes on a different machine, first
make sure it exists at the path you put after `MEDIA:`. For existing
Hermes-local files, the gateway will try NapCat's stream upload API and then
send the returned NapCat-local temp path. If stream upload feedback reports a
failure, publish it as an HTTP(S) URL, convert it to `base64://...`, or place it
in a directory mounted at the same path on the NapCat host.

For Hermes-generated local files, a safe pattern is:
create/download file on Hermes → include `MEDIA:/absolute/path/to/file` if it
exists → if delivery feedback says stream upload failed, expose it as URL or
copy it to a shared NapCat-readable path and retry with that URL/path.

If you really need to control the segment array yourself (e.g. inline image +
text in one bubble), call:

```
napcat_call("send_group_msg", {
  "group_id": 12345678,
  "message": [
    {"type": "image", "data": {"file": "https://example.com/meme.png"}},
    {"type": "text",  "data": {"text": "看这个"}}
  ]
})
```

Local paths must be absolute paths or `file://` URIs that are valid on the
NapCat machine, or existing Hermes-local files that the gateway can stream
upload to NapCat. URLs work unchanged.

---

## 7. Receiving rich media

Inbound non-text segments are surfaced inline as parseable markers:

| Segment       | Marker                          | How to fetch                                |
|---------------|---------------------------------|---------------------------------------------|
| `image`       | `[图片:<file_id>]`              | `napcat_call("get_image", {"file": "<file_id>"})` |
| `record`      | `[语音:<file_id>]`              | `napcat_call("get_record", {"file": "<file_id>", "out_format": "mp3"})` |
| `video`       | `[视频:<file_id>]`              | (no direct action — use `get_file` if available)  |
| `file`        | `[文件:<name>:<file_id>]`       | `napcat_call("get_file", {"file_id": "<file_id>"})` |
| `face`        | `[表情:<id>]`                   | informational                               |
| `at` (other)  | `@<qq>`                         | informational                               |

Behaviour rules:

- If the user sent only an image and asks "what is this?", call `get_image`,
  then run `vision_analyze` on the returned local path.
- If they uploaded a file with a familiar extension, use `get_file` then read
  it normally with the file tools.
- Do **not** echo the markers back unless the user explicitly asks.

---

## 8. Common workflows (compact recipes)

**Read the last 30 group messages**:

```
napcat_call("get_group_msg_history", {"group_id": 12345678, "count": 30})
```

Reply with a 1–2 line summary in plain text (no Markdown).

**Send a meme to a group**: include a NapCat-readable URL such as
`MEDIA:https://example.com/meme.png` in your final reply. Add a short caption
before the tag if it adds context.

**Recall the bot's last reply** (when the user says “撤回”): grab `message_id`
from the previous turn (or call `get_msg` if missing), then
`napcat_call("delete_msg", {"message_id": "<id>"})`. Reply with a single
character `好` — anything more wastes a message.

**Change the bot's QQ display name**:

```
napcat_call("set_qq_profile", {"nickname": "新昵称"})
```

Confirm by quoting the new name in one short sentence.

**Kick a member** (only when the user is unambiguously asking and the bot is a
group admin):

```
napcat_call("set_group_kick", {"group_id": 12345678, "user_id": 87654321,
                               "reject_add_request": false})
```

Confirm in one line. If the call fails with `retcode != 0`, surface the
returned message verbatim.

**React with thumbs-up to a message** (NapCat extension):

```
napcat_call("set_msg_emoji_like", {"message_id": "<id>", "emoji_id": "76"})
```

**Upload a log file to the group**:

```
napcat_call("upload_group_file", {
  "group_id": 12345678,
  "file": "/var/log/myapp.log",
  "name": "myapp-2026-04-30.log"
})
```

If `/var/log/myapp.log` only exists on Hermes and not on the NapCat host, prefer
letting the gateway's file-send path handle it so it can call
`upload_file_stream` first. If you manually call `upload_group_file` via
`napcat_call`, the `file` parameter must already be readable by NapCat.

---

## 9. Failure handling

- The tool returns `{"error": "..."}` for any non-`ok` retcode — surface a
  short, factual one-liner to the user (still no Markdown).
- “NapCat is not connected” / “gateway not running” errors are
  configuration problems, not transient. Don't retry; tell the user.
- Timeouts on send paths might still have delivered; don't auto-retry — the
  underlying adapter already protects against that.

---

## 10. Don'ts

- Don't paste raw OneBot JSON back to the user as a reply.
- Don't loop calling `get_group_msg_history` — bound to the user's ask
  (default 20–30 messages).
- Don't hammer `send_like` or `set_msg_emoji_like` (rate limits exist).
- Don't change profile / group settings without an explicit user request.
- Don't send personal data of one user to another.
