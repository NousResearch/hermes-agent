# QQ (OneBot)

> 中文版见 [`README.zh-CN.md`](README.zh-CN.md) · Chinese readme: [`README.zh-CN.md`](README.zh-CN.md)

The OneBot adapter connects Hermes to QQ through the **OneBot 11 protocol**, compatible with [NapCat](https://napneko.github.io/), [Lagrange](https://github.com/LagrangeDev/Lagrange.Core), LLOneBot, and go-cqhttp. Instead of the official QQ Bot platform (which requires a Tencent-approved app), it drives a regular QQ account through a local bridge. Good for personal bots and groups the official platform can't reach.

```
User (QQ) ←→ NapCat ←→ Hermes onebot adapter ←→ Hermes agent
                      ├─ reverse WS server / forward WS client (auto-reconnect)
                      ├─ inbound: CQ parsing, image download, voice STT, quote / forward expand
                      └─ outbound: split sends, markdown strip, [[qq_forward]], image/voice/video/file
```

> Run `hermes gateway setup` and pick **QQ (OneBot)** for a guided walk-through.

## Feature overview

| Area | Capability |
|---|---|
| Connection | reverse WS (NapCat ws-reverse dials in, default port 8643) or forward WS (dial out, default `ws://127.0.0.1:3001`); auto-reconnect |
| Inbound | private/group chats, segment-array parsing first (CQ-string fallback), CQ unescaping, @/reply trigger detection (fail-closed), **image resolution (url/base64/file/hash via `get_image`)** + downscale, file/voice/video/face/json/poke segment types, quote original message (`get_msg`), **file dual-channel receive (CDN direct `get_private_file_url` + `get_file` base64/url fallback)**, poke-notice reply when the bot itself is poked (opt-in, `poke_reply`) |
| Voice | ffmpeg → 16 kHz mono WAV → Hermes STT pipeline; voice without a URL fetched via `get_record` (base64); failure degrades to `[语音]` |
| Text image | AstrBot-style t2i card renderer: headings / bold / italic / strikethrough / quote / list / code block / **table** / inline-code pill / color emoji / CJK punctuation rules; 800 px wide |
| Outbound | sentence-boundary split (default ≤100 chars), **>150 chars rendered as a t2i card**, markdown stripped to plain text, `[[qq_forward]]` merged forwarding, **loop interim merge + recall** (turn-end **"本轮进展" summary card**, 60 ms recall spacing), typing indicator (private chats) |
| Commands | admin-only local slash commands: `/ocr` / `/mode` / `/id` / `/ver` / `/approve` / `/reject`; other slashes keep flowing to the gateway core |
| Tools | `qq_send_image` (≤9, path or URL), `qq_send_voice` / `qq_send_video` / `qq_send_file` / `qq_send_forward`, `qq_napcat_api` (15-action whitelist), `qq_group_history`; HTTP `/api/napcat` + `/api/send_media`; available in CLI/TUI too via `provides_tools` |
| Permissions | admin allowlist (`ONEBOT_ALLOWED_USERS`), dm/group policy (open/allowlist/disabled), group mention gating, restricted-member `[受限用户:仅问答]` soft limits, outbound sensitive-intent audit |
| Ops | hot reload for `onebot_utils.py` / `t2i_render.py` (`extra.hot_reload`, no gateway restart), temp-media TTL cleanup, startup t2i ink check (font chain self-test) |

## Compatibility

| Item | Requirement |
|---|---|
| Hermes | gateway with the onebot platform enabled |
| OneBot 11 bridge | NapCat / Lagrange / LLOneBot / go-cqhttp (reverse or forward WebSocket) |
| Optional deps | `ffmpeg` on the Hermes host for voice transcription (the adapter probes it once at startup and logs a WARNING — voice STT stays off until you install it); an STT backend configured under `stt:` (local faster-whisper, auto-downloaded on first use, or an OpenAI-compatible API); CJK fonts for text-image cards (see below) |

## Configure Hermes

Add the platform to the `gateway` block in `~/.hermes/gateway-config.yaml`:

```yaml
gateway:
  platforms:
    onebot:
      enabled: true
      extra:
        mode: reverse              # reverse | forward
        host: "127.0.0.1"          # listen address (reverse: WS+API; forward: /api-only server); 0.0.0.0 requires access_token
        port: 8643                 # listen port (reverse: WS+API; forward: /api-only server)
        # url: "ws://127.0.0.1:3001"   # forward: bridge ws endpoint
        # access_token: ""         # must match the bridge's token; REQUIRED when host is 0.0.0.0
        # bot_qq: ""               # optional; auto-learned from meta events
        require_mention: true      # groups: only reply when @'d
        dm_policy: open            # open | allowlist | disabled
        allow_from: []             # user ids when dm_policy=allowlist
        group_policy: open         # open | allowlist | disabled
        group_allow_from: []       # group ids when group_policy=allowlist
        admin_users: []            # admins; falls back to ONEBOT_ALLOWED_USERS
        split_length: 100          # long replies split at this many chars
        text_image_threshold: 150  # longer replies render as a text image
        image_max_size: 2048       # downscale inbound images (0 = keep as-is)
        poke_reply: false          # light reply when the bot itself gets poked
```

### Extra keys (all optional)

| Key | Default | Meaning |
|---|---|---|
| `mode` | `reverse` | `reverse` (bridge dials in) / `forward` (adapter dials out) |
| `host` / `port` | `127.0.0.1` / `8643` | reverse: WS + `/api/*` listener; forward: `/api/*`-only server (no `/ws`). Non-loopback hosts like `0.0.0.0` require `access_token` |
| `url` | `ws://127.0.0.1:3001` | forward target |
| `access_token` | empty | OneBot token, must match the bridge |
| `bot_qq` | empty | bot's QQ (empty = learned from meta events) |
| `require_mention` | `true` | groups: only respond when @'d or replying to the bot (undeterminable replies fall back to responding) |
| `dm_policy` | `open` | `open` (admins only) / `allowlist` / `disabled` |
| `allow_from` | `[]` | allowed user ids when `dm_policy=allowlist` |
| `group_policy` | `open` | `open` / `allowlist` / `disabled` |
| `group_allow_from` | `[]` | allowed group ids when `group_policy=allowlist` |
| `admin_users` | `[]` | admin QQ ids; falls back to `ONEBOT_ALLOWED_USERS` |
| `split_length` | `100` | long-reply split threshold |
| `text_image_threshold` | `150` | t2i card threshold; `<=0` disables the card path |
| `image_max_size` | `2048` | inbound image long-edge cap (px); `0` keeps originals |
| `max_inbound_file_bytes` | `20971520` (20 MB) | inbound file size cap; larger files degrade to a `[文件:name]` marker |
| `interim_recall_seconds` | `90` | auto-recall timeout for unsettled interim messages (`0` disables) |
| `hot_reload` | `false` | dev only: reload `onebot_utils.py` / `t2i_render.py` on mtime change (saves a gateway restart while iterating styles; keep off in production, an in-place write during upgrade can reload a half-written module) |
| `poke_reply` | `false` | reply with a light hint (`@我` / `/help`) when the bot itself is poked (`notify/poke` notice); member-to-member pokes are ignored; per-chat 60s cooldown |

Environment variables: `ONEBOT_ALLOWED_USERS` (comma-separated admin ids), `ONEBOT_ALLOW_ALL_USERS=true` (dev only), and the global `GATEWAY_ALLOW_ALL_USERS`.

> **`dm_policy: open` + allow-all env**: `ONEBOT_ALLOW_ALL_USERS=true` (or
> `GATEWAY_ALLOW_ALL_USERS=true`) is the explicit opt-in that makes `open`
> actually open to non-admins. Without it, `open` means **admins only**:
> the adapter rejects non-admin DMs at intake, before the gateway's
> allow-all check would ever run.

### Connection modes

| Mode | Description |
|------|-------------|
| `reverse` (default) | Hermes hosts a WebSocket server; the bridge's **ws-reverse** client dials in (`ws://<hermes-host>:8643/ws`). One connection carries both events and actions. |
| `forward` | Hermes dials the bridge's WebSocket server (`ws://<bridge-host>:3001` for NapCat's default). Also serves the qq_* tools via a local `/api/*` server (see below). |

If the bridge uses an access token, set the same value in `access_token` (Hermes sends it as `Authorization: Bearer <token>` on the reverse connection; forward mode includes it in the handshake headers).

#### Tool availability in forward mode

Forward mode also serves the qq_* tools: once the forward WS connection is
up, the adapter starts a local HTTP server on `host:port` that registers only
the `/api/*` helper endpoints (`/api/group_history`, `/api/napcat`,
`/api/send_media`) — the reverse-only `/ws` route is **not** registered. The
server reuses the same `host` / `port` keys as reverse mode (code default
`127.0.0.1:8643`, matching the qq_* tools' default base URL
`http://127.0.0.1:8643`), and its endpoints sit behind the same auth gate
described below. If you customize `host` / `port`, set `ONEBOT_TOOL_BASE`
accordingly (e.g. `http://127.0.0.1:9000`) — the qq_* tools derive their base
URL from that environment variable only, not from the plugin config. Reverse
and forward are mutually exclusive: each adapter
instance runs exactly one of the two servers, never both.

### Token & network security

The local helper endpoints `/api/group_history`, `/api/napcat` and
`/api/send_media` (used by the qq_* model tools) hang off the same
`host:port` — alongside `/ws` on the reverse listener, on their own in
forward mode — and all of them sit behind one shared auth gate:

- **`access_token` set**: every request to `/ws` and `/api/*` must carry
  `Authorization: Bearer <token>`; anything else gets `401`.
- **`access_token` empty**: only loopback clients (`127.0.0.1` / `::1` — the
  whole `127.0.0.0/8` range and IPv4-mapped loopback forms included) are
  served; requests from other addresses get `401` and a WARNING is logged per
  request. Default `127.0.0.1` deployments are unaffected.

**If `host` is set to `0.0.0.0` (or any other non-loopback address), you MUST
set `access_token`.** Without a token, any host that can reach the port can
connect to `/ws` as if it were your bridge, while the local helper endpoints
stay locked to loopback — leaving a `0.0.0.0` deployment unauthenticated for
the very protocol it was bound for.

The bundled qq_* model tools authenticate automatically: once `access_token`
is set, their `/api/*` requests carry `Authorization: Bearer <token>` — the
token is read from the same config source as the adapter (or the
`ONEBOT_ACCESS_TOKEN` environment override), so no extra setup is needed.
Without a token they send no Authorization header and token-less loopback
deployments keep working unchanged.

**WS frame limit of 4 MiB** (declared explicitly, matching the aiohttp default —
zero behavior change): NapCat frames larger than 4 MiB (e.g. base64-inline
media) are rejected. Prefer URL-based image delivery (NapCat's file-to-URL
switch) in production to stay under the limit.

### NapCat-side setup (required)

After enabling the plugin you **must also configure the bridge**. The plugin cannot connect by itself:

- **reverse mode (recommended, NapCat dials in)**: in NapCat's network settings add a **WebSocket client**:
  - report URL: `ws://<hermes-host-ip>:<port>/ws` (e.g. `ws://192.168.1.100:8643/ws`; use the LAN IP, not `127.0.0.1`, when Hermes and NapCat are on different machines)
  - token: same value as `access_token` on the Hermes side (leave both empty if no token)
  - message post format: **array** is recommended (segment-array parsing first, CQ string only as fallback)
- **forward mode (Hermes dials out)**: in NapCat's network settings enable the **WebSocket server** (default `0.0.0.0:3001`), then set the plugin's `url` to `ws://<napcat-host-ip>:3001` (`ws://127.0.0.1:3001` when same host) and keep tokens identical.

The bridge must be on a network Hermes can reach (same LAN / routable); WS connection, image downloads and file resolution all depend on that path. This adapter is **LAN-only**. Cross-internet deployments are not supported.

> ⚠️ When NapCat runs on a **different machine than Hermes**, enable the **「文件转 URL」/ file-to-URL** switch in NapCat's network settings (`enableLocalFile2Url`). Without it `get_file` returns container-local paths that Hermes cannot access, and file messages degrade to a `[文件:name]` marker instead of being downloadable.

## dm / group access policy (choose at setup)

Both policies **must be chosen when you first configure the plugin**. The
defaults only make sense after you pick one of the three options for each:

| Value | dm_policy (private) | group_policy (group) |
|---|---|---|
| `open` | **admins only** (non-admins silently rejected, no pairing flow) | **all groups** can chat; replies gated by `require_mention` |
| `allowlist` | only `allow_from` ids can DM (admin not required) | only `group_allow_from` groups can chat |
| `disabled` | all DMs rejected | all group messages ignored |

> ⚠️ **At least one admin must be set at setup** (`admin_users` or the
> `ONEBOT_ALLOWED_USERS` env var). With `dm_policy: open` (the default) only
> admins can DM, and slash commands are admin-only. With no admin configured
> nobody can talk to the bot. For quick dev testing use
> `ONEBOT_ALLOW_ALL_USERS=true`.

Which option is right for you:

- **personal bot** → `dm_policy: open` + `admin_users: [<your QQ>]`. Only you can DM
- **a few friends** → `dm_policy: allowlist` + `allow_from: [<QQ1>, <QQ2>]`. Non-admins in the list can DM too
- **group bot** → `group_policy: open` (default `require_mention: true` keeps it quiet: members must @ or reply to trigger)
- **specific groups only** → `group_policy: allowlist` + `group_allow_from: [<group_id>]`

### Access tiers (admin / restricted member)

Groups can be opened to all members of allowlisted groups while privileged
operations stay admin-only. The adapter enforces its own access policy
(`enforces_own_access_policy`), so the gateway trusts its allowlist decisions.

| Role | Who | Group @ | DM | Capabilities |
|---|---|---|---|---|
| admin | `extra.admin_users` (falls back to `ONEBOT_ALLOWED_USERS`) | full | allowed | everything incl. slash commands |
| member | any other user in an allowlisted group | restricted | rejected | quick Q&A, image analysis, group summaries only |
| unauthorized | outside allowlisted groups / DM allowlist | blocked | rejected (pairing disabled) | — |

Enforcement points:

- member group messages get a `[受限用户:仅问答]` text prefix so the agent applies the soft restriction (quick Q&A only, no file/terminal/config/HA/cross-platform/cron actions; declared in the platform hint)
- member slash commands (`/new`, `/model`, `/help`, …) are dropped before a `MessageEvent` is constructed; path-like text (`/tmp/x`) is not affected
- member DMs are rejected
- outbound replies to member chats are scanned against sensitive-intent keywords and logged with a WARNING (audit, not hard blocking)

### Admin local slash commands

Admins get a few local slash commands handled inside the adapter (anything else keeps flowing to the gateway core):

| Command | Action |
|---|---|
| `/ocr` | OCR the most recently received inbound image via NapCat's `ocr_image` |
| `/mode interim\|instant` | per-chat loop-merge mode override (`interim` = merge commentary into forwards, `instant` = send as-is); in-memory only, resets on restart |
| `/id` | print the current chat id |
| `/ver` | print the plugin version |
| `/approve <seq\|flag>` | accept a pending friend request / group invite (OneBot `set_friend_add_request` / `set_group_add_request`); references use the `#seq` from the admin DM notification or the full flag. Bare `/approve` (no arg) is **not** handled here — it passes through to the gateway core's pending dangerous-command / data-training-tier confirmation flow |
| `/reject <seq\|flag>` | decline a pending friend request / group invite (same APIs with `approve=false`); bare `/reject` passes through to the gateway core the same way |

`/ocr` needs the image to still exist in the temp media dir (6-hour TTL cleanup applies).

### Friend request / group invite approval

Incoming OneBot `request` events (`request_type=friend`, or `request_type=group` with
`sub_type=add` / `invite`) are parsed, recorded to `<HERMES_HOME>/onebot_requests.json`
(0600 — contains applicant QQ ids and verification messages, treat as sensitive), and
every configured admin gets a private DM with the sequence number, applicant, verification
message summary and flag. Reply `/approve 1` / `/reject 1` (or use the full flag) in the
admin DM to decide. Approvals are idempotent (an already-processed flag never triggers a
second API call), survive restarts via the persisted ledger, and unknown references are
reported explicitly. Receiving requests and the admin notification are isolated from the
main message flow.

## Group mentions

With `require_mention: true` (default), the bot only responds in groups when it is explicitly @'d or when the message replies to a message the bot itself sent. Replying to someone else stays silent, so busy groups are no longer woken by unrelated reply chains. The replied-to message is resolved with `get_msg` — the same fetch used for quote text, so there is no extra API call — and when the sender cannot be determined (fetch failed or timed out, the original message was recalled and deleted, or the bot's own id is still unknown) the bot falls back to treating the reply as a mention, so replies to recalled messages still trigger. Set it to `false` to respond to every group message (noisy; not recommended for large groups). When no `bot_qq` is configured the bot learns its own id from OneBot meta events, so mention detection works out of the box.

## Long replies (three tiers, fully configurable)

Reply length is handled in three tiers. Both thresholds are user-configurable
(`split_length` and `text_image_threshold` in the platform `extra` block):

- **≤ `split_length`** (default 100) characters: sent as a single text message.
- **`split_length` to `text_image_threshold`** (default 150): split into multiple messages, breaking at sentence boundaries (`。！？!?；;\\n`) so sentences are never cut in half.
- **> `text_image_threshold`**: rendered as a black-on-white text image (800 px wide, CJK-aware font fallback chain) and sent as a single image message. The card is capped at a total height of 8000 px — taller content aborts the render and falls back to plain text chunks (the same fallback as a failed render). Falls back to text chunks if rendering fails.

Set `text_image_threshold: 0` to disable the image path (everything splits as text);
raise/lower either value to tune the trade-off between message count and card rendering.

The text-image renderer is an AstrBot-style **element-based Markdown renderer**: bold / italic / strikethrough / inline code / code blocks / headers / quotes / lists and **tables** (AstrBot itself has no table element) are all drawn natively. Chinese typography rules are honored. Punctuation never starts a line (行首禁则), inline styles wrap as a whole line, literal `\\n` in plain text becomes a real line break (inside inline code it becomes a space; `\\\\n` is kept), and inline code uses a light-blue pill with a monospace font for Latin/digits and glyph-level fallback for CJK. When the reply target's nickname is known, the card gets an AstrBot-style blue top bar (`To <nickname>`, Klein blue #002FA7, white text at **twice the body font size**, ~68 px tall), matching AstrBot's card header proportions.

### Font dependencies (auto-install one-liner)

The card renderer needs three font families (CJK / monospace / color emoji),
auto-registered from the system; missing glyphs render as tofu boxes.

| Dependency | Provides (auto-registered path) | Used for |
|---|---|---|
| `fonts-noto-cjk` | `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` | Chinese body/headers (ttc SC face auto-selected, JP/Mono fallback) |
| `fonts-dejavu-core` | `/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf` | code blocks / inline code monospace |
| `fonts-noto-color-emoji` | `/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf` | color emoji |
| optional `fonts-wqy-zenhei` / `fonts-wqy-microhei` | `/usr/share/fonts/truetype/wqy/*.ttc` | CJK fallback if Noto is missing |
| optional `fonts-unifont` | `/usr/share/fonts/opentype/unifont/*.otf` | last-resort fallback |

Linux (Debian/Ubuntu): one command covers all required fonts:

```sh
sudo apt install fonts-noto-cjk fonts-dejavu-core fonts-noto-color-emoji
```

macOS needs nothing (system Hiragino Sans GB / Songti SC, Menlo and Apple Color Emoji are picked up automatically). The renderer runs an ink self-check: font families without usable glyphs are dropped and fall back silently instead of producing tofu cards.

## Markdown & voice

- **Markdown is stripped** before delivery. QQ does not render it, so `**bold**`, headings, lists, and tables are converted to readable plain text (headings → `【…】`, lists → `•`, tables → spaced cells, fenced code blocks → bordered boxes). This runs before splitting and text-image rendering, so images are clean too.
- **Inbound voice messages are transcribed**: the adapter downloads the clip and converts it with `ffmpeg` to 16 kHz mono WAV, then hands it to Hermes' STT pipeline. NapCat private voice messages often carry only a file hash (no URL): the adapter calls the OneBot `get_record` action to fetch the base64 audio before the download→ffmpeg→STT pipeline.

  STT itself uses the global `stt:` config (same pipeline as other platforms):
  `provider: local` runs faster-whisper on the Hermes host (model = `stt.local.model`, default `small`, downloaded automatically on first use); `provider: openai` calls an OpenAI-compatible endpoint (configure `stt.openai.*` + API key). Requirements: `ffmpeg` must be installed; without it (or without a working STT backend), voice clips degrade to a `[语音]` marker. The adapter probes `ffmpeg` once at startup (`shutil.which`) and logs a WARNING with install instructions when it is missing — voice STT stays unavailable until ffmpeg is on PATH.

## Images

- Inbound messages are parsed as the **OneBot segment array** (`message` field) when available. Image/voice/at/face/video/file segments are handled structurally; text-format clients fall back to CQ-code string parsing. CQ entity escaping (`&amp;` → `&`, `&#91;` → `[`, …) is reversed before any URL is used, so CDN links with `&` parameters download correctly.
- Images are downloaded to a temp directory and exposed to the vision tool via `media_urls`; undownloadable images degrade to a `[图片]` placeholder. If the image segment only carries a `file` hash (no URL), the adapter calls the OneBot `get_image` action to resolve the real URL; `base64://` and `file://` forms are handled directly.
- Images larger than `image_max_size` (default **2048** px on the long edge) are downscaled with Pillow before the LLM sees them. High-resolution QQ photos otherwise make vision calls slow or time out. RGBA stays PNG, everything else becomes JPEG (q85); animated GIFs collapse to their first frame. Set `image_max_size: 0` to keep originals untouched.

## Inbound files

Inbound `file` segments are resolved through **two channels** so container-local paths never leak to the agent:

1. **CDN direct link first**: for private chats the adapter calls `get_private_file_url` and downloads the file straight from the QQ CDN.
2. **`get_file` fallback**: when no direct link is available (group files, or NapCat without the file-to-URL switch), it falls back to `get_file`, accepting base64 or an http URL.

The downloaded file is stored under the temp media dir, annotated `[文件:本地路径]` in the message text so the agent can read it locally, and capped by `max_inbound_file_bytes` (default 20 MB). Files over the cap degrade to a plain `[文件:name]` marker. The annotation is skipped for user-level plugins that must stay path-neutral.

## Outbound media

The adapter implements the gateway's native media senders as OneBot segments, so the agent can send media through the standard `MEDIA:` / markdown-image mechanism:

| Capability | OneBot segment | Notes |
|---|---|---|
| Image URL (direct) | `image` with `url` | Markdown image URLs from the agent are sent as-is; the bridge downloads them (no local file needed). |
| Local image | `image` with `base64://` | Up to 8 MB. |
| Batch images | multiple `image` segments | One message, max 9 images per message; URL + local mixed. |
| Voice | `record` with `base64://` | Up to 20 MB; the bridge transcodes to silk. |
| Video | `video` with `base64://` | Up to 20 MB. |
| File | `file` with `base64://` + `name` | Up to 20 MB. |
| Forwarded messages | `send_forward_msg` with `node`s | Group chats only. |

**Merged forwarding** is triggered by an agent-side block:

```
[[qq_forward]]
<display name>
<message text>
---
<display name>
<message text>
[[/qq_forward]]
```

Each `---`-separated block becomes one forwarded node (name + text, 500-char cap per node). In private chats the marker is ignored and the block degrades to plain text.

## Replying to a message (quote)

When the user replies to (quotes) a previous message, the adapter calls the OneBot `get_msg` API to fetch the original message and:

- prefixes the original text with `[引用]` so the agent sees what was quoted
- attaches any image / voice / video from the original message as media (voice goes through the STT pipeline, video is downloaded for frame extraction)

This works for both segment-array and CQ-string payloads. If `get_msg` fails the current message is delivered unchanged.

## Loop-message merge (interim commentary folding)

During a multi-tool turn the gateway sends interim commentary messages ("Using tool X…") followed by the final response. To save chat space the adapter buffers interim text messages per chat and, when the final message arrives, **merges them into a single QQ forwarded-message first, sends the final content (including any text-image card), then retracts the originals**:

- group chats use `send_forward_msg`, private chats use `send_private_forward_msg`
- merge happens only when ≥2 interim messages are buffered
- retraction (`delete_msg`) runs only after the merge forward succeeded; on failure the originals are kept
- the buffer and pending-retraction list are cleared on any new inbound user message

This relies on the gateway marking commentary sends with `interim: True` in the stream-consumer metadata (see `gateway/stream_consumer.py`).

### Per-interim auto-recall

Each buffered interim message gets its own independent timer (`interim_recall_seconds`, default **90**). If the turn never settles (no final message arrives), the interim is recalled on its own instead of piling up forever. Once the buffer is settled by a final message the pending tasks self-cancel, so a settled turn is never double-retracted. Set `interim_recall_seconds: 0` to disable.

### Turn-end summary card

When a turn settles with **≥2** buffered interims, they are rendered into a single **"本轮进展" text-image card** summarizing the partial progress, the originals are retracted, and the final reply continues normally. If rendering or sending the card fails, the adapter falls back to the plain merge-forward path.

### Recall spacing

`delete_msg` calls are spaced **60 ms apart** to stay under NapCat's rate limit when retracting a batch of messages.

## Agent model tools

The plugin registers model-facing tools so the agent can push media and query NapCat directly (available in CLI/TUI sessions too via `provides_tools`):

| Tool | Purpose |
|---|---|
| `qq_send_image` / `qq_send_voice` / `qq_send_video` / `qq_send_file` | send media out of band; chat resolves from the argument or `HERMES_SESSION_CHAT_ID` |
| `qq_send_forward` | merged forward of text nodes |
| `qq_napcat_api` | whitelisted NapCat actions: `get_group_member_list`, `get_group_member_info`, `get_stranger_info`, `get_forward_msg`, `get_record`, `get_file`, `upload_group_file`, `upload_private_file`, `get_group_root_files`, `get_group_files_by_folder`, `get_group_file_url`, `ocr_image`, `get_ai_characters`, `send_group_ai_record`, `get_group_msg_history`; anything outside the whitelist returns 403 |
| `qq_group_history` | fetch recent group history, paged via `message_seq` (≤50 per page) |

HTTP equivalents: `GET /api/napcat` (action proxying) and `POST /api/send_media` on the adapter's local API.

## Hot reload

Opt-in via `extra.hot_reload: true` (default **off**, dev only). When enabled, `onebot_utils.py` (pure helpers: CQ parsing, splitting, markdown stripping, emoji map) and `t2i_render.py` (text-image renderer) are reloaded on every use: the adapter stats the file mtime and calls `importlib.reload` when it changed, so style/rule tweaks apply without a gateway restart. Changes to `adapter.py` itself still require a restart.

## Privacy & data

- **Network**: one WebSocket connection to the OneBot 11 bridge (reverse listener or forward dial-out); inbound images/files are downloaded from QQ CDN.
- **Files**: inbound media is cached under a temp dir with a 6-hour TTL cleanup; the nickname cache used by the card banner is persisted to `nicknames.json` next to the plugin.
- **System calls**: voice transcription invokes local `ffmpeg` (and the configured STT backend); no other local tooling is required.
- **Sensitive info**: `access_token` and the admin allowlist are read from config only and never written to logs; outbound replies to restricted-member chats are audit-logged against sensitive-intent keywords.
- **No telemetry**: the adapter makes no third-party calls beyond the configured OneBot bridge and QQ CDN.

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Group chat not responding | `require_mention: true` needs an @ or a reply to the bot; reply triggering falls back to mention when the replied-to sender is undeterminable; confirm `bot_qq` was learned from meta events or set it explicitly |
| Image download 403 | NapCat escapes `&` in URLs to `&amp;` (parsing unescapes automatically); check the media-download log lines if it still fails |
| Voice shows `[语音]` placeholder | `ffmpeg` unavailable, or `get_record` failed; install ffmpeg and retry — a missing ffmpeg also triggers a startup WARNING |
| File message arrives empty | CQ-string bridges may omit the `file` segment name. The adapter marks it `[文件:<name>]` (name falls back to the `file=` attribute); NapCat private files carry only a hash + container path, so the name comes from the `file=` attribute |
| Chinese tofu boxes in text-image cards | CJK fonts missing: `apt install fonts-noto-cjk` |
| Loop interim messages not merged | gateway must send `interim: True` in commentary metadata (patched `_send_commentary` in `gateway/stream_consumer.py`); adapter-side merge is only a fallback consumer |

## Notes

- Outbound messages use the OneBot segment-array format (not CQ-code strings). Required for NapCat's message handling.
- Replies are sent as plain text without quoting the triggering message.
- QQ faces map to common emoji; unknown faces collapse to `[表情]`. Voice without a downloadable link degrades to `[语音]`; inbound video/file segments degrade to `[视频]` / `[文件:name]` placeholders; unknown segment types (json cards, poke, forwarded-message CQ codes) degrade to `[卡片]` / `[戳一戳]` / `[合并转发:id]` placeholders.
- In **private chats** the bot shows QQ's native "typing…" bubble while the agent generates (via NapCat's `set_input_status` extension). Group chats have no typing indicator on QQ.
- Long replies may take a few seconds to render; the gateway shows a typing indicator where supported.
- Cron / scheduled deliveries cannot attach media to OneBot yet (the core `send_message_tool` media whitelist covers telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack only). Interactive replies are unaffected.
