# Voice Channel Diagnostic Pattern

Reference for investigating "voice-to-voice chat not working" reports on Discord voice channels (e.g. Misa-Misa).

## Default assumption: infrastructure is present

Hermes bundles all voice infrastructure. When a user reports voice-to-voice failure, **do not assume missing dependencies**. Check activation state first:

1. Is there a voice channel for the bot to join?
2. Does the bot have voice permissions?
3. Has `/voice join` ever been invoked?

Infrastructure checks are secondary — they are almost always present.

## Infrastructure verification checklist

| Component | Check | Expected |
|---|---|---|
| Opus codec | `python -c "import discord; print(discord.opus.is_loaded())"` | `True` |
| libopus.so | `ldconfig -p | grep opus` | `libopus.so.0` present |
| PyNaCl | `python -c "import nacl; print(nacl.__version__)"` | `1.5.0`+ |
| DAVE E2EE | `python -c "import davey"` | No error |
| faster-whisper | `python -c "import faster_whisper; print(faster_whisper.__version__)"` | `1.2.1`+ |
| STT config | `grep -A2 "^stt:" ~/.hermes/config.yaml` | `enabled: true`, `provider: local` |
| Voice intents | `grep "intents.voice_states" plugins/platforms/discord/adapter.py` | Present in init |
| `/voice` handler | `grep "voice join" gateway/run.py` | Present |
| `join_voice_channel` | `grep "def join_voice_channel" plugins/platforms/discord/adapter.py` | Present |

If any check fails, that's a real infrastructure gap. If all pass, the problem is environmental/permissional.

## Environmental checks (the real failure mode)

### 1. Does the voice channel exist?

The `DISCORD_HOME_CHANNEL` in the profile `.env` may be a text channel, not a voice channel. The bot needs a voice channel to join.

```bash
# Check if the configured home channel is a voice channel
GUILD_ID="<guild_id>"
TOKEN=$(grep -oP 'DISCORD_BOT_TOKEN=\K.*' ~/.hermes/profiles/<profile>/.env)
CHANNEL_ID=$(grep -oP 'DISCORD_HOME_CHANNEL=\K\d+' ~/.hermes/profiles/<profile>/.env)

curl -s -H "Authorization: Bot $TOKEN" \
  "https://discord.com/api/v10/channels/$CHANNEL_ID" | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"id={d['id']} name={d['name']} type={d['type']}\")"
```

Channel types: `0` = text, `2` = voice. If the home channel is text (`0`), the bot needs a separate voice channel.

### 2. Does the bot have voice permissions?

Check the bot's invite permissions. Voice requires `CONNECT` (1048576) + `SPEAK` (2097152) + `USE_VAD` (4194304).

```bash
# Decode permissions from invite URL or check guild member roles
# Text-only: 117760
# Text + Voice: 7457792
```

### 3. Has the bot ever joined a voice channel?

```bash
# Search gateway log for voice join/leave events
grep -n -i "voice\|joined\|joined voice\|failed to join\|Left voice" \
  ~/.hermes/profiles/<profile>/logs/gateway.log
```

Zero hits = the `/voice join` command was never invoked. The voice workflow is not broken — it was never started.

### 4. Can the bot see the voice channel?

If permission overwrites deny @everyone VIEW_CHANNEL (1024) and the bot's role is not in the allow list, the channel is invisible to the bot. The `join_voice_channel` method will fail with a permissions error.

```bash
# Check channel overwrites
curl -s -H "Authorization: Bot $TOKEN" \
  "https://discord.com/api/v10/channels/<voice_channel_id>" | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps(d.get('permission_overwrites', []), indent=2))"
```

## Corrective actions

| Problem | Fix |
|---|---|
| No voice channel exists | Create one via REST API (`type: 2`) or Developer Portal |
| Bot lacks voice permissions | Regenerate invite URL with `permissions=7457792`, re-invite |
| Bot can't see voice channel | Add bot role to channel permission overwrites with VIEW + CONNECT + SPEAK + VAD |
| `/voice join` never invoked | User must run `/voice join` from the linked text channel |
| Home channel is text, no voice channel defined | Create voice channel, update `DISCORD_HOME_CHANNEL` if it should be the voice channel, or link text→voice via channel config |

## Voice permission integer

| Permission | Bit | Value |
|---|---|---|
| VIEW_CHANNEL | 10 | 1024 |
| CONNECT | 20 | 1048576 |
| SPEAK | 21 | 2097152 |
| USE_VAD | 22 | 4194304 |
| **Voice total** | | **7340032** |
| **Text + Voice total** | | **7457792** (includes text perms) |

## How the voice workflow activates (manual join path)

```
1. User joins Discord voice channel
2. User types "/voice join" in linked text channel
3. Gateway calls _handle_voice_channel_join() → adapter.join_voice_channel()
4. Bot connects, starts VoiceReceiver + _voice_listen_loop
5. User speaks → RTP → Opus decode → PCM buffer
6. Silence detected → _process_voice_input() → WAV → transcribe_audio()
7. faster-whisper transcribes → synthetic MessageEvent → agent → TTS reply
8. Bot speaks response back in voice channel
```

Without step 2 (`/voice join`), steps 3-8 never execute. The infrastructure is idle but ready.

## How the voice workflow activates (auto-join path)

Some bots (e.g. Misa-Misa) support **auto-join**: when a specific user joins a monitored voice channel, the bot joins automatically without requiring `/voice join`. The auto-join path differs from manual join:

```
1. User joins voice channel (same guild, channel matches config)
2. on_voice_state_update fires in Discord adapter
3. _handle_auto_join_voice_state checks {user_id, channel_id} match
4. If match: adapter.join_voice_channel() called automatically
5. _voice_sources dict wired so voice transcripts route to configured text channel
6. Text greeting sent via _send_auto_join_text_greeting() or
   Voice greeting played via _play_auto_join_greeting()
7. Steps 5-8 of manual join path proceed normally
```

### Auto-join configuration (adapter.py + config.yaml)

The adapter reads these keys from `config.yaml` under `discord.extra`:

```yaml
discord:
  extra:
    auto_join_user_id: "<discord_user_id>"          # user to watch for
    auto_join_text_channel_id: "<text_channel_id>"  # where transcripts go
    auto_join_greeting_text: "Hey, it's Misa-Misa. What's up?"
    auto_leave_on_user_exit: true                     # leave when user leaves
    voice_timeout_seconds: 900                        # timeout if idle
```

### Auto-join troubleshooting checklist

When auto-join "used to work" but now doesn't, follow this sequence:

1. **Is the bot in the voice channel at all?** — Check Discord UI. If not, auto-join logic was never triggered or the adapter was never wired.

2. **Is `on_voice_state_update` receiving events?**
   ```bash
   tail -f ~/.hermes/profiles/<profile>/logs/gateway.log | grep -i "voice_state\|auto_join\|joined voice"
   ```
   Zero hits = the gateway's Discord client may need `intents.voice_states = True`, or the event handler is missing entirely (e.g. the feature was lost during a migration or venv change).

3. **Has the feature code been lost?** — Verify the adapter has the handler methods:
   ```bash
   # Check inside the repo (source of truth for editable installs)
   grep "_handle_auto_join_voice_state\|_send_auto_join\|_play_auto_join" \
     /home/kensei/repos/KenseiAgent/plugins/platforms/discord/adapter.py
   ```

4. **Is `auto_join_user_id` actually reaching the runtime?**
   Zero hits of `VC EVENT` but `Voice state: ... joined` appears → `_auto_join_user_id` is falsy at runtime despite being in `config.yaml`. Usually means `auto_join_*` keys are placed at the wrong nesting level in `config.yaml` (must be inside `discord.extra:`, not top-level under `discord:`). See `references/voice-auto-join-pattern.md` for full details.

5. **Is the running process using the updated source?** — See `systematic-debugging` skill reference `references/linux-system-process-debugging.md`, section "In-process module path verification". A common failure mode: the adapter.py is patched in the repo, but the gateway process was started from a different venv or older install that loads a different `hermes_plugins.discord_platform.adapter` module. The bot appears online but the auto-join code never executes.

6. **Are the config values correctly typed?** — Discord.py passes `int` IDs. If the config stored them as strings, coercion is required or the equality check fails silently.

7. **Does the bot have voice permissions?** — Even if auto-join fires, `join_voice_channel()` silently fails if permissions are missing. See manual join section above for permission values.

### When auto-join was historically present and disappeared

Symptom: "After venv migration the bot stopped auto-joining."

Likely cause: The auto-join code was present in a previous environment and was never forward-ported. Hermes does NOT persist adapter customisations across venv reinstalls.

**Also check:** If the code was ported but auto-join still fails after restart, the issue may be the **three-layer fix** pattern — config placement, config bridge, and `intents.members` must ALL be correct simultaneously. Fixing any one in isolation does not restore auto-join. See `references/discord-voice-config-bridge-three-layer-fix.md` for the definitive single-log-line verification.

Recovery steps:
1. Check `session_search` or session history for previous sessions on the old venv where auto-join worked
2. Diff the `adapter.py` from the old environment against the current repo copy
3. Forward-port the missing methods and wiring into the current `adapter.py`
4. Verify the cole method signatures match the current Discord.py version
5. Re-install the editable link if the `hermes_plugins` namespace is not mapped:  
   `/path/to/venv/bin/pip install -e /path/to/repo --force-reinstall --no-deps`
6. Restart the gateway and verify via in-process import test (see systematic-debugging reference)

## Related

- `references/discord-rest-api-channel-management.md` — channel creation, permission overwrites, REST API patterns
- SKILL.md "Bot creation workflow" — permissions values and invite URL construction
- `systematic-debugging/references/linux-system-process-debugging.md` — stale-process misdiagnosis and in-process module verification