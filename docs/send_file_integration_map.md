# send_file Integration Map — Hermes Agent Codebase Deep Dive

## 1. environments/tool_context.py — Base64 File Transfer Implementation

### upload_file() (lines 153-205)
- Reads local file as raw bytes, base64-encodes to ASCII string
- Creates parent dirs in sandbox via `self.terminal(f"mkdir -p {parent}")`
- **Chunk size:** 60,000 chars (~60KB per shell command)
- **Small files (<=60KB b64):** Single `printf '%s' '{b64}' | base64 -d > {remote_path}`
- **Large files:** Writes chunks to `/tmp/_hermes_upload.b64` via `printf >> append`, then `base64 -d` to target
- **Error handling:** Checks local file exists; returns `{exit_code, output}`
- **Size limits:** No explicit limit, but shell arg limit ~2MB means chunking is necessary for files >~45KB raw
- **No theoretical max** — but very large files would be slow (many terminal round trips)

### download_file() (lines 234-278)
- Runs `base64 {remote_path}` inside sandbox, captures stdout
- Strips output, base64-decodes to raw bytes
- Writes to host filesystem with parent dir creation
- **Error handling:** Checks exit code, empty output, decode errors
- Returns `{success: bool, bytes: int}` or `{success: false, error: str}`
- **Size limit:** Bounded by terminal output buffer (practical limit ~few MB via base64 terminal output)

### Promotion potential:
- These methods work via `self.terminal()` — they're environment-agnostic
- Could be directly lifted into a new tool that operates on the agent's current sandbox
- For send_file, this `download_file()` pattern is the key: it extracts files from sandbox → host

## 2. tools/environments/base.py — BaseEnvironment Interface

### Current methods:
- `execute(command, cwd, timeout, stdin_data)` → `{output, returncode}`
- `cleanup()` — release resources
- `stop()` — alias for cleanup
- `_prepare_command()` — sudo transformation
- `_build_run_kwargs()` — subprocess kwargs
- `_timeout_result()` — standard timeout dict

### What would need to be added for file transfer:
- **Nothing required at this level.** File transfer can be implemented via `execute()` (base64 over terminal, like ToolContext does) or via environment-specific methods.
- Optional: `upload_file(local_path, remote_path)` and `download_file(remote_path, local_path)` methods could be added to BaseEnvironment for optimized per-backend transfers, but the base64-over-terminal approach already works universally.

## 3. tools/environments/docker.py — Docker Container Details

### Container ID tracking:
- `self._container_id` stored at init from `self._inner.container_id`
- Inner is `minisweagent.environments.docker.DockerEnvironment`
- Container ID is a standard Docker container hash

### docker cp feasibility:
- **YES**, `docker cp` could be used for optimized file transfer:
  - `docker cp {container_id}:{remote_path} {local_path}` (download)
  - `docker cp {local_path} {container_id}:{remote_path}` (upload)
- Much faster than base64-over-terminal for large files
- Container ID is directly accessible via `env._container_id` or `env._inner.container_id`

### Volumes mounted:
- **Persistent mode:** Bind mounts at `~/.hermes/sandboxes/docker/{task_id}/workspace` → `/workspace` and `.../home` → `/root`
- **Ephemeral mode:** tmpfs at `/workspace` (10GB), `/home` (1GB), `/root` (1GB)
- **User volumes:** From `config.yaml docker_volumes` (arbitrary `-v` mounts)
- **Security tmpfs:** `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)

### Direct host access for persistent mode:
- If persistent, files at `/workspace/foo.txt` are just `~/.hermes/sandboxes/docker/{task_id}/workspace/foo.txt` on host — no transfer needed!

## 4. tools/environments/ssh.py — SSH Connection Management

### Connection management:
- Uses SSH ControlMaster for persistent connection
- Control socket at `/tmp/hermes-ssh/{user}@{host}:{port}.sock`
- ControlPersist=300 (5 min keepalive)
- BatchMode=yes (non-interactive)
- Stores: `self.host`, `self.user`, `self.port`, `self.key_path`

### SCP/SFTP feasibility:
- **YES**, SCP can piggyback on the ControlMaster socket:
  - `scp -o ControlPath={socket} {user}@{host}:{remote} {local}` (download)
  - `scp -o ControlPath={socket} {local} {user}@{host}:{remote}` (upload)
- Same SSH key and connection reuse — zero additional auth
- Would be much faster than base64-over-terminal for large files

## 5. tools/environments/modal.py — Modal Sandbox Filesystem

### Filesystem API exposure:
- **Not directly.** The inner `SwerexModalEnvironment` wraps Modal's sandbox
- The sandbox object is accessible at: `env._inner.deployment._sandbox`
- Modal's Python SDK exposes `sandbox.open()` for file I/O — but only via async API
- Currently only used for `snapshot_filesystem()` during cleanup
- **Could use:** `sandbox.open(path, "rb")` to read files or `sandbox.open(path, "wb")` to write
- **Alternative:** Base64-over-terminal already works via `execute()` — simpler, no SDK dependency

## 6. gateway/platforms/base.py — MEDIA: Tag Flow (Complete)

### extract_media() (lines 587-620):
- **Pattern:** `MEDIA:\S+` — extracts file paths after MEDIA: prefix
- **Voice flag:** `[[audio_as_voice]]` global directive sets `is_voice=True` for all media in message
- Returns `List[Tuple[str, bool]]` (path, is_voice) and cleaned content

### _process_message_background() media routing (lines 752-786):
- After extracting MEDIA tags, routes by file extension:
  - `.ogg .opus .mp3 .wav .m4a` → `send_voice()`
  - `.mp4 .mov .avi .mkv .3gp` → `send_video()`
  - `.jpg .jpeg .png .webp .gif` → `send_image_file()`
  - **Everything else** → `send_document()`
- This routing already supports arbitrary files!

### send_* method inventory (base class):
- `send(chat_id, content, reply_to, metadata)` — ABSTRACT, text
- `send_image(chat_id, image_url, caption, reply_to)` — URL-based images
- `send_animation(chat_id, animation_url, caption, reply_to)` — GIF animations
- `send_voice(chat_id, audio_path, caption, reply_to)` — voice messages
- `send_video(chat_id, video_path, caption, reply_to)` — video files
- `send_document(chat_id, file_path, caption, file_name, reply_to)` — generic files
- `send_image_file(chat_id, image_path, caption, reply_to)` — local image files
- `send_typing(chat_id)` — typing indicator
- `edit_message(chat_id, message_id, content)` — edit sent messages

### What's missing:
- **Telegram:** No override for `send_document` or `send_image_file` — falls back to text!
- **Discord:** No override for `send_document` — falls back to text!
- **WhatsApp:** Has `send_document` and `send_image_file` via bridge — COMPLETE.
- The base class defaults just send "📎 File: /path" as text — useless for actual file delivery.

## 7. gateway/platforms/telegram.py — Send Method Analysis

### Implemented send methods:
- `send()` — MarkdownV2 text with fallback to plain
- `send_voice()` — `.ogg`/`.opus` as `send_voice()`, others as `send_audio()`
- `send_image()` — URL-based via `send_photo()`
- `send_animation()` — GIF via `send_animation()`
- `send_typing()` — "typing" chat action
- `edit_message()` — edit text messages

### MISSING:
- **`send_document()` NOT overridden** — Need to add `self._bot.send_document(chat_id, document=open(file_path, 'rb'), ...)`
- **`send_image_file()` NOT overridden** — Need to add `self._bot.send_photo(chat_id, photo=open(path, 'rb'), ...)`
- **`send_video()` NOT overridden** — Need to add `self._bot.send_video(...)`

## 8. gateway/platforms/discord.py — Send Method Analysis

### Implemented send methods:
- `send()` — text messages with chunking
- `send_voice()` — discord.File attachment
- `send_image()` — downloads URL, creates discord.File attachment
- `send_typing()` — channel.typing()
- `edit_message()` — edit text messages

### MISSING:
- **`send_document()` NOT overridden** — Need to add discord.File attachment
- **`send_image_file()` NOT overridden** — Need to add discord.File from local path
- **`send_video()` NOT overridden** — Need to add discord.File attachment

## 9. gateway/run.py — User File Attachment Handling

### Current attachment flow:
1. **Telegram photos** (line 509-529): Download via `photo.get_file()` → `cache_image_from_bytes()` → vision auto-analysis
2. **Telegram voice** (line 532-541): Download → `cache_audio_from_bytes()` → STT transcription
3. **Telegram audio** (line 542-551): Same pattern
4. **Telegram documents** (line 553-617): Extension validation against `SUPPORTED_DOCUMENT_TYPES`, 20MB limit, content injection for text files
5. **Discord attachments** (line 717-751): Content-type detection, image/audio caching, URL fallback for other types
6. **Gateway run.py** (lines 818-883): Auto-analyzes images with vision, transcribes audio, enriches document messages with context notes

### Key insight: Files are always cached to host filesystem first, then processed. The agent sees local file paths.

## 10. tools/terminal_tool.py — Terminal Tool & Environment Interaction

### How it manages environments:
- Global dict `_active_environments: Dict[str, Any]` keyed by task_id
- Per-task creation locks prevent duplicate sandbox creation
- Auto-cleanup thread kills idle environments after `TERMINAL_LIFETIME_SECONDS`
- `_get_env_config()` reads all TERMINAL_* env vars for backend selection
- `_create_environment()` factory creates the right backend type

### Could send_file piggyback?
- **YES.** send_file needs access to the same environment to extract files from sandboxes.
- It can reuse `_active_environments[task_id]` to get the environment, then:
  - Docker: Use `docker cp` via `env._container_id`
  - SSH: Use `scp` via `env.control_socket`
  - Local: Just read the file directly
  - Modal: Use base64-over-terminal via `env.execute()`
- The file_tools.py module already does this with `ShellFileOperations` — read_file/write_file/search/patch all share the same env instance.

## 11. tools/tts_tool.py — Working Example of File Delivery

### Flow:
1. Generate audio file to `~/.hermes/audio_cache/tts_TIMESTAMP.{ogg,mp3}`
2. Return JSON with `media_tag: "MEDIA:/path/to/file"`
3. For Telegram voice: prepend `[[audio_as_voice]]` directive
4. The LLM includes the MEDIA tag in its response text
5. `BasePlatformAdapter._process_message_background()` calls `extract_media()` to find the tag
6. Routes by extension → `send_voice()` for audio files
7. Platform adapter sends the file natively

### Key pattern: Tool saves file to host → returns MEDIA: path → LLM echoes it → gateway extracts → platform delivers

## 12. tools/image_generation_tool.py — Working Example of Image Delivery

### Flow:
1. Call FAL.ai API → get image URL
2. Return JSON with `image: "https://fal.media/..."` URL
3. The LLM includes the URL in markdown: `![description](URL)`
4. `BasePlatformAdapter.extract_images()` finds `![alt](url)` patterns
5. Routes through `send_image()` (URL) or `send_animation()` (GIF)
6. Platform downloads and sends natively

### Key difference from TTS: Images are URL-based, not local files. The gateway downloads at send time.

---

# INTEGRATION MAP: Where send_file Hooks In

## Architecture Decision: MEDIA: Tag Protocol vs. New Tool

The MEDIA: tag protocol is already the established pattern for file delivery. Two options:

### Option A: Pure MEDIA: Tag (Minimal Change)
- No new tool needed
- Agent downloads file from sandbox to host using terminal (base64)
- Saves to known location (e.g., `~/.hermes/file_cache/`)
- Includes `MEDIA:/path` in response text
- Existing routing in `_process_message_background()` handles delivery
- **Problem:** Agent has to manually do base64 dance + know about MEDIA: convention

### Option B: Dedicated send_file Tool (Recommended)
- New tool that the agent calls with `(file_path, caption?)`
- Tool handles the sandbox → host extraction automatically
- Returns MEDIA: tag that gets routed through existing pipeline
- Much cleaner agent experience

## Implementation Plan for Option B

### Files to CREATE:

1. **`tools/send_file_tool.py`** — The new tool
   - Accepts: `file_path` (path in sandbox), `caption` (optional)
   - Detects environment backend from `_active_environments`
   - Extracts file from sandbox:
     - **local:** `shutil.copy()` or direct path
     - **docker:** `docker cp {container_id}:{path} {local_cache}/` 
     - **ssh:** `scp -o ControlPath=... {user}@{host}:{path} {local_cache}/`
     - **modal:** base64-over-terminal via `env.execute("base64 {path}")`
   - Saves to `~/.hermes/file_cache/{uuid}_{filename}`
   - Returns: `MEDIA:/cached/path` in response for gateway to pick up
   - Register with `registry.register(name="send_file", toolset="file", ...)`

### Files to MODIFY:

2. **`gateway/platforms/telegram.py`** — Add missing send methods:
   ```python
   async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None):
       with open(file_path, "rb") as f:
           msg = await self._bot.send_document(
               chat_id=int(chat_id), document=f,
               caption=caption, filename=file_name or os.path.basename(file_path))
       return SendResult(success=True, message_id=str(msg.message_id))
   
   async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None):
       with open(image_path, "rb") as f:
           msg = await self._bot.send_photo(chat_id=int(chat_id), photo=f, caption=caption)
       return SendResult(success=True, message_id=str(msg.message_id))
   
   async def send_video(self, chat_id, video_path, caption=None, reply_to=None):
       with open(video_path, "rb") as f:
           msg = await self._bot.send_video(chat_id=int(chat_id), video=f, caption=caption)
       return SendResult(success=True, message_id=str(msg.message_id))
   ```

3. **`gateway/platforms/discord.py`** — Add missing send methods:
   ```python
   async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None):
       channel = self._client.get_channel(int(chat_id)) or await self._client.fetch_channel(int(chat_id))
       with open(file_path, "rb") as f:
           file = discord.File(io.BytesIO(f.read()), filename=file_name or os.path.basename(file_path))
           msg = await channel.send(content=caption, file=file)
       return SendResult(success=True, message_id=str(msg.id))
   
   async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None):
       # Same pattern as send_document with image filename
   
   async def send_video(self, chat_id, video_path, caption=None, reply_to=None):
       # Same pattern, discord renders video attachments inline
   ```

4. **`toolsets.py`** — Add `"send_file"` to `_HERMES_CORE_TOOLS` list

5. **`agent/prompt_builder.py`** — Update platform hints to mention send_file tool

### Code that can be REUSED (zero rewrite):

- `BasePlatformAdapter.extract_media()` — Already extracts MEDIA: tags
- `BasePlatformAdapter._process_message_background()` — Already routes by extension
- `ToolContext.download_file()` — Base64-over-terminal extraction pattern
- `tools/terminal_tool.py` _active_environments dict — Environment access
- `tools/registry.py` — Tool registration infrastructure
- `gateway/platforms/base.py` send_document/send_image_file/send_video signatures — Already defined

### Code that needs to be WRITTEN from scratch:

1. `tools/send_file_tool.py` (~150 lines):
   - File extraction from each environment backend type
   - Local file cache management
   - Registry registration
   
2. Telegram `send_document` + `send_image_file` + `send_video` overrides (~40 lines)
3. Discord `send_document` + `send_image_file` + `send_video` overrides (~50 lines)

### Total effort: ~240 lines of new code, ~5 lines of config changes

## Key Environment-Specific Extract Strategies

| Backend    | Extract Method                 | Speed    | Complexity |
|------------|-------------------------------|----------|------------|
| local      | shutil.copy / direct path     | Instant  | None       |
| docker     | `docker cp container:path .`  | Fast     | Low        |
| docker+vol | Direct host path access       | Instant  | None       |
| ssh        | `scp -o ControlPath=...`      | Fast     | Low        |
| modal      | base64-over-terminal          | Moderate | Medium     |
| singularity| Direct path (overlay mount)   | Fast     | Low        |

## Data Flow Summary

```
Agent calls send_file(file_path="/workspace/output.pdf", caption="Here's the report")
    │
    ▼
send_file_tool.py:
    1. Get environment from _active_environments[task_id]
    2. Detect backend type (docker/ssh/modal/local)
    3. Extract file to ~/.hermes/file_cache/{uuid}_{filename}
    4. Return: '{"success": true, "media_tag": "MEDIA:/home/user/.hermes/file_cache/abc123_output.pdf"}'
    │
    ▼
LLM includes MEDIA: tag in its response text
    │
    ▼
BasePlatformAdapter._process_message_background():
    1. extract_media(response) → finds MEDIA:/path
    2. Checks extension: .pdf → send_document()
    3. Calls platform-specific send_document(chat_id, file_path, caption)
    │
    ▼
TelegramAdapter.send_document() / DiscordAdapter.send_document():
    Opens file, sends via platform API as native document attachment
    User receives downloadable file in chat
```
