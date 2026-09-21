# Voice Auto-Join Pattern (Misa-Misa Workflow)

Documentation of the auto-join voice workflow used by Misa-Misa to automatically enter a voice channel when Sahil joins, transcribe speech, and route text replies back into the voice channel via TTS.

## When auto-join is needed

When a bot serves as a voice-first intake layer (like Misa-Misa), requiring the user to manually type `/voice join` every time creates friction. Auto-join removes the manual step by detecting the user's voice channel entry and joining automatically.

## Config fields (discord.extra)

All auto-join configuration lives under the `extra:` block inside `discord:` in the bot's profile `config.yaml` — NEVER in the root config.yaml, because the feature is per-bot:

```yaml
discord:
  require_mention: true
  free_response_channels: '1506022800190607370'  # #misa-misa
  extra:
    auto_join_user_id: '797682085224513547'      # Sahil's Discord user ID (REQUIRED)
    auto_join_text_channel_id: '1506022800190607370'  # Text channel for transcripts + replies
    auto_join_greeting_text: "Hey, it's Misa-Misa. What's up? How can I help?"
    auto_leave_on_user_exit: true                   # Auto-leave when target user leaves voice
    voice_timeout_seconds: 900                      # Max seconds to stay alone in VC (15min)
```

### Field reference

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `auto_join_user_id` | string | — | **Yes** | The Discord user ID (snowflake) to trigger auto-join. Only this user entering a voice channel causes the bot to join. Multiple users are NOT supported — must be a single snowflake. |
| `auto_join_text_channel_id` | string | — | Yes | Text channel where transcripts are posted and where agent replies are read from. This channel becomes the "voice text channel" — the bridge between voice input and text output. |
| `auto_join_greeting_text` | string | `""` | No | Text sent to `auto_join_text_channel_id` when auto-join triggers. Also used to generate the TTS greeting played in the voice channel. Empty = no greeting. |
| `auto_leave_on_user_exit` | bool | `true` | No | If `true`, the bot automatically disconnects when `auto_join_user_id` leaves the voice channel (detected via `on_voice_state_update`). |
| `voice_timeout_seconds` | int | `900` | No | How many seconds the bot stays in the voice channel after being alone. Applies both to auto-join and manual `/voice join` contexts. Configurable — when unset, falls back to `DiscordAdapter.VOICE_TIMEOUT` (300s). |

## Pitfall: auto-join code lost during venv migration

This exact failure occurred on **2026-05-27** when the auto-join feature was lost during a systemd service venv migration.

**What happened:**
1. Auto-join feature was developed (2026-05-21) as a live patch inside the old pipx venv: `~/.local/share/pipx/venvs/hermes-agent/lib/python3.12/site-packages/gateway/platforms/discord.py`
2. Feature was tested and approved by Sahil.
3. On 2026-05-26, the systemd service was migrated from `~/.hermes/hermes-agent/venv` → `~/repos/KenseiAgent/.venv`.
4. The new venv had clean upstream code — the live-patch changes from the old venv were **never ported**.
5. Service restarts used the new venv → old code → feature "disappeared".

**Detection in logs:**
```
grep -n "auto_join\|_handle_auto_join_voice_state" adapter.py
# → 0 hits = feature code is absent
```

**Root cause verification:**
```bash
# Check if the feature keyword exists in the service runtime
for p in /home/kensei/.hermes/hermes-agent /home/kensei/repos/KenseiAgent; do
  echo "=== $p ==="
  grep -c "auto_join" "$p/plugins/platforms/discord/adapter.py"
done
# Divergent counts = code was lost in migration
```

**Prevention:**
- After every live-patch or hot-fix, immediately commit to the repo AND run `pip install -e .`
- Before any venv migration, extract a diff: `diff -ur $OLD_VENV $NEW_VENV > /tmp/migration-diff.txt`
- Verify the feature keyword count matches across repo and service after migration restart
- See `references/service-repo-coherence.md` for the full sync pattern

**Recovery (if already lost):**
1. Use `session_search` to find the session where the feature was originally built:
   ```
   session_search(query="auto_join voice_state_update", sort="newest")
   ```
2. Extract the implementation from the session compact/history — include full code blocks
3. Re-apply to the new codebase: `patch` on `adapter.py`, `write_file` on `config.yaml`
4. Run `pip install -e .` from the repo checkout (force-reinstall may be needed to refresh the `hermes_plugins` namespace package):
   ```bash
   /path/to/venv/bin/pip install -e /path/to/repo --force-reinstall --no-deps
   ```
5. Verify the editable mapping now points to the repo by checking the `.pth` file and by doing an in-process module import test (see `systematic-debugging/references/linux-system-process-debugging.md`, section "In-process module path verification")
6. Restart the gateway
7. Verify with keyword grep across both repo and service

**Post-recovery stale-process check:**
After any gateway restart, always confirm the new process loads the expected module:
```bash
# Ask the new process where it loaded adapter.py
python3 -c "
import sys, os
os.chdir('/home/kensei/repos/KenseiAgent')
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
from hermes_cli.plugins import get_plugin_manager
get_plugin_manager().discover_and_load(force=True)
import hermes_plugins.discord_platform.adapter as adapter
print('File:', adapter.__file__)
print('Has _handle_auto_join_voice_state:', hasattr(adapter.DiscordAdapter, '_handle_auto_join_voice_state'))
"
```
If `__file__` points to a path other than the repo, the editable mapping is stale.

## Detection and join sequence

```
1. on_voice_state_update(DISCONNECT→CONNECT) for target user
2. Is the target user in a voice channel? (voice_state.channel_id is set)
3. Is the bot already in a voice channel? (avoids double-join)
4. Is there a configured voice channel OR can we use the user's current channel?
5. join_voice_channel(channel_id=channel_id)                           [adapter join]
6. Wire text channel: _voice_text_channels[channel_id] = text_channel_id   [get text transcriptions here]
7. Wire source tracking: _voice_sources[channel_id] = ...user info       [needed by listen loop]
8. Send text greeting to text channel
9. Generate TTS audio for greeting_text via tts_tool
10. Play greeting in voice channel via play_audio()
```

**Critical step 6-7:** The `_voice_text_channels` and `_voice_sources` MUST be populated before the `VoiceReceiver` starts. These dictionaries tell the `_voice_listen_loop` where to post transcriptions and which user is the source. Without this wiring, the transcript is generated but never sent anywhere.

## How voice timeout interacts

If `auto_leave_on_user_exit: true`, the timeout life looks like this:

```
User joins VC  → bot joins → greet → listen loop starts
  |
User leaves VC → on_voice_state_update detects → bot immediately disconnects
  |
If user stays, but disconnects (network hiccup):
  Bot alone → voice_timeout_seconds countdown → leaves if alone at timeout
  User rejoins before timeout → bot stays
```

The timeout uses `_voice_timeout_task` which is per-VC. When the user leaves, the bot cancels this task and disconnects. When the user rejoins, a new task is scheduled.

## Diagnostic checklist for auto-join failure

When the user reports "voice chat isn't working" **and the bot has auto_join configured**, check in this order:

| # | Check | How | Expected |
|---|---|---|---|
| 1 | `auto_join_user_id` matches the joining user | Compare the VMU of the user in `on_voice_state_update` against config | Exact match. User IDs are snowflakes; string comparison. |
| 2 | `auto_join_text_channel_id` configured | `grep auto_join_text_channel_id config.yaml` | Present and valid channel ID |
| 3 | Adaftertext channel exists and bot can see it | REST API GET /channels/{id} with bot token | Returns channel object, not 404 or 403 |
| 4 | Text channel permissions | Check bot role overwrites | SEND_MESSAGES allowed |
| 5 | Voice channel permissions | `CONNECT` + `SPEAK` + `USE_VAD` on guild role or bot role | All three granted |
| 6 | Bot has VoiceReceiver | `ldconfig -p \| grep opus` + `__import__('discord').opus.is_loaded()` | opus loaded, PyNaCl present |
| 7 | Text-to-speech (TTS) enabled | `grep "tts_tool" adapter.py` | At least one TTS tool registration exists |
| 8 | STT/faster-whisper | `import faster_whisper` | No ImportError |
| 9 | The target user is actually joining a VC, not just a text channel | Voice state `channel_id` is set | Non-null `channel_id` |
| 10 | Auto-join code present in adapter.py | `grep -n "auto_join" adapter.py` | Lines for `_handle_auto_join_voice_state` and callers |

If checks 1-9 pass but #10 fails, the auto-join code was **notported** from the feature branch to the runtime service. See `references/service-repo-coherence.md`.

### Diagnostic checklist additions (items 11–13)

| # | Check | How | Expected |
|---|---|---|---|
| 11 | `_auto_join_user_id` truthy at runtime | Add a startup log immediately after the field is parsed in `__init__`: `logger.info(f"[Discord] _auto_join_user_id={self._auto_join_user_id}")` | Correct user ID (int or string), not `None` or `0` |
| 12 | `VC EVENT` log emitted when target user joins VC | `grep "VC EVENT" gateway.log` immediately after target user joins voice channel | Should appear within seconds of the `Voice state: ... joined` line |
| 13 | `AUTO-JOIN HANDLER called:` log emitted | `grep "AUTO-JOIN HANDLER called:" gateway.log` | Should appear if item 12 passed |

### "Smoking gun" log-line hierarchy

When voice state events are dispatched by Discord, `adapter.py` emits log lines at different depths:

| Log line | Typical trigger | Condition | Meaning |
|---|---|---|---|
| `Voice state: {user} ({id}) joined {channel}` | After `bot_guild_ids` validation | Unconditional once the event reaches the adapter | Discord dispatched the event and the adapter received it |
| `VC EVENT: ...` | Early in `on_voice_state_update` | Requires `adapter_self._auto_join_user_id` to be truthy | The auto-join gate has opened |
| `AUTO-JOIN HANDLER called:` | Entry of `_handle_auto_join_voice_state` | Requires `_auto_join_user_id` truthy inside the handler | The handler itself is entered |

**If you see `Voice state:` but never `VC EVENT`** → `_auto_join_user_id` is falsy at runtime despite being present in `config.yaml`. Discord sent the event; the adapter dropped it because the auto-join user is not configured on the instance. Add a startup log (item 11) to confirm the parsed value.

### Pitfall: Auto-join config present in YAML but falsy at runtime

The adapter parses `_auto_join_user_id` during `__init__`. Even when the value exists in `config.yaml`, the runtime attribute can end up as `None` or `0` because:

- A `_coerce_int` helper returns `None` for non-integer or empty values.
- The adapter reads from the wrong config path (e.g., top-level `extra:` instead of `discord.extra:`). Hermes bridges platform-specific configs into `PlatformConfig`, but adapter implementations vary in how they access nested `extra` values.

**Key symptom:** `Voice state: {name} ({id}) joined {channel}` appears in `gateway.log`, but neither `VC EVENT` nor `AUTO-JOIN HANDLER called:` ever appears. The event arrives, but the auto-join gate rejects it before any handler code runs.

**Fix:** Add a startup log immediately after `_auto_join_user_id` is set in `__init__`. If it logs as `None`, trace backwards through the config object to find where the value lives in the parsed dict and update the adapter's read path. Verify both locations:
```python
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("/home/kensei/.hermes/profiles/<profile>/config.yaml").read_text())
print("discord.extra:", cfg.get("discord", {}).get("extra", {}).get("auto_join_user_id"))
print("top-level extra:", cfg.get("extra", {}).get("auto_join_user_id"))
```

### Pitfall: `auto_join_*` keys placed outside `discord.extra` in config.yaml

The adapter reads `extra.get("auto_join_user_id")` (and related keys) from `PlatformConfig.extra`. However, the shared-key bridge in `gateway/config.py` (lines 809-881) only copies a **fixed whitelist** of top-level `discord:` keys into `PlatformConfig.extra`. `auto_join_user_id` is **not** on that whitelist.

**Wrong placement (top-level under `discord:`):**
```yaml
discord:
  require_mention: true
  free_response_channels: '1506022800190607370'
  auto_join_user_id: '797682085224513547'          # ← outside extra — INVISIBLE to adapter
  auto_join_text_channel_id: '1506022800190607370'
```

**Correct placement (inside `discord.extra`):**
```yaml
discord:
  require_mention: true
  free_response_channels: '1506022800190607370'
  extra:
    auto_join_user_id: '797682085224513547'        # ← inside extra — READ by adapter
    auto_join_text_channel_id: '1506022800190607370'
    auto_join_greeting_text: "Hey, it's Misa-Misa..."
    auto_leave_on_user_exit: true
    voice_timeout_seconds: 900
```

**Why this happens:** Hermes's `gateway/config.py` bridges `discord.require_mention`, `discord.free_response_channels`, `discord.allowed_channels`, `discord.server_actions`, and a few others — but any custom key added by a feature (like `auto_join_user_id`) must live in `extra:` to be visible. `extra:` is an opaque passthrough dict; top-level keys are explicitly enumerated.

**Signature in logs:** The debug log in `_handle_auto_join_voice_state` prints `auto_join=None` even though `config.yaml` contains a value. This is a near-certain sign the key is at the wrong nesting level.

### Pitfall: `/voice channel` command works but auto-join does not

When debug logs show `auto_join=None` (see pitfall above), the `/voice channel <channel>` slash command **still works** because it follows a completely different code path:

| Path | Trigger | Checks `auto_join_user_id`? | Code |
|---|---|---|---|
| Auto-join | `on_voice_state_update` event | Yes | `_handle_auto_join_voice_state` → `join_voice_channel` |
| Manual join | `/voice channel` slash command | No | `_handle_voice_channel_join` → `join_voice_channel` |

The manual-join handler (`_handle_voice_channel_join`) receives the channel object directly from the slash-command interaction and passes it straight to `join_voice_channel`. It never references `_auto_join_user_id`. This means a manual `/voice channel` test passing **does not** prove auto-join config is correct — it only proves voice infrastructure (Opus, permissions, etc.) is functional.

**Diagnostic implication:** If the user says "manual join works but auto-join doesn't", immediately suspect one of two things:
1. `auto_join_user_id` is falsy at runtime (config placement or bridge issue), OR
2. `intents.members = False` causing silent event discard (see `references/discord-voice-members-intent-silent-discard.md`)

The two failures produce identical user-visible symptoms (manual join OK, auto-join silent) but have different log signatures.

## The post-auto-join voice loop

After successful auto-join, the standard voice workflow takes over:

```
User speaks → RTP packets → VoiceReceiver buffers → Opus decode → PCM → WAV
Silence detected → synthesise MessageEvent with text transcript
Agent processes text → text reply generated
TTS converts text reply → audio file
Bot plays audio via VC play_audio()
Audio played → temporary file cleaned up (if in /tmp)
```

The only difference from manual `/voice join` is step 1: auto-detect instead of command trigger.

## Related

- `references/voice-channel-diagnostic-pattern.md` — the companion reference for manual-join diagnostics
- `references/discord-voice-config-bridge-three-layer-fix.md` — three-layer fix (config placement, config bridge, intents.members) where fixing any one in isolation does not restore auto-join; contains the definitive one-line log verification pattern
- `references/service-repo-coherence.md` — when code is present in a repo checkout but missing from the service runtime