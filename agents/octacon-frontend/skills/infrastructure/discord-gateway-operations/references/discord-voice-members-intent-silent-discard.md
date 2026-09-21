# Discord Voice `intents.members` Silent Discard Pattern

Discovered: 2026-05-27
Meets: Misa-Misa voice auto-join investigation

## The symptom

Bot is configured for `auto_join_user_id`, `voice_only` mode enabled, `/voice on` set, and the user joins a voice channel. The bot **never joins**. Zero log entries related to voice state updates, auto-join, or VoiceReceiver. Manual `/voice join` works fine. The auto-join handler code is present in `adapter.py`.

No error in `gateway.log`. No `PrivilegedIntentsRequired` exception. Discord connection is healthy. The bot appears fully functional except this one feature.

## Root cause

`intents.members` is `False` for this gateway. When `intents.members` is disabled, `discord.py` does not populate `guild._members`. When a `VOICE_STATE_UPDATE` event fires for a non-self member, `discord.py` calls `guild._update_voice_state(data, channel_id)` which returns `None` because the member is not in cache. discord.py then silently **discards** the event with a DEBUG log that never appears in production logs.

The `on_voice_state_update` handler in Hermes's adapter is **never called** — not because the handler is missing, but because `discord.py` never dispatches the event.

## Why this happens specifically with auto-join

In `adapter.py` line 748-751:

```python
intents.members = (
    any(not entry.isdigit() for entry in self._allowed_user_ids)
    or bool(self._allowed_role_ids)
)
```

- `allowed_user_ids` entries are pure numeric snowflakes → `all(d.isdigit() for d in ...)` → first term is `False`
- `allowed_role_ids` is empty/None → second term is `False`
- Result: `intents.members = False`

The code is **oblivious to `auto_join_user_id`**. Auto-join fundamentally requires tracking the member's voice state — which requires the `members` intent.

## Detection checklist

| # | Check | Command | Expected for this failure |
|---|---|---|---|
| 1 | Gateway connected | `grep "Connected as" gateway.log` | Yes, bot is online |
| 2 | Auto-join config present | `grep auto_join_user_id config.yaml` | Correct user ID configured |
| 3 | Voice mode enabled | `cat gateway_voice_mode.json` | `"voice_only"` |
| 4 | `on_voice_state_update` in code | `grep "on_voice_state_update" adapter.py` | Present (handler exists) |
| 5 | VC EVENT in logs | `grep "VC EVENT" gateway.log` | **Zero hits** (this is the tell) |
| 6 | `intents.members` truthiness | Read adapter.py line 748-751 | Evaluates to `False` |
| 7 | PrivilegedIntentsRequired | `grep -i "privileged" gateway.log` | None (this is a silent drop, not a hard error) |

Key signature: **zero voice-state log entries despite a live handler**. If the handler were missing or broken, you'd see VC EVENT logs but no join action. If the intent were enabled but permissions missing, you'd see VC EVENT logs and a permission error.

## The two-part fix

### Part 1: Adapter code (Hermes side)

```python
# adapter.py, inside connect() where intents are set
intents.members = (
    any(not entry.isdigit() for entry in self._allowed_user_ids)
    or bool(self._allowed_role_ids)
    or bool(self._auto_join_user_id)  # <--- REQUIRED for auto-join
)
```

This ensures `intents.members = True` whenever `auto_join_user_id` is configured.

### Part 2: Discord Developer Portal (Discord side)

The "Server Members Intent" must be toggled ON in the bot's application:

1. Go to https://discord.com/developers/applications
2. Select the Misa-Misa application
3. Bot tab → Privileged Gateway Intents
4. Toggle **Server Members Intent** → ON
5. Save Changes
6. Restart the gateway

Without the portal toggle, even `intents.members = True` in code will produce a `PrivilegedIntentsRequired` error on connection. With the toggle on and the code fix, `discord.py` caches the member object and `on_voice_state_update` fires normally.

## Verification after fix

```bash
# 1. Check the patched adapter.py
grep -A5 "intents.members =" /home/kensei/repos/KenseiAgent/plugins/platforms/discord/adapter.py
# Should show "or bool(self._auto_join_user_id)"

# 2. Verify discord.py will now cache members
python3 -c "
import discord
i = discord.Intents.default()
i.voice_states = True
i.members = True
print('members intent:', i.members)
print('voice_states intent:', i.voice_states)
"

# 3. After gateway restart, join a voice channel
# 4. grep gateway.log for "VC EVENT" — should appear within seconds
# 5. grep gateway.log for "Auto-join" — should show join action
```

## How this differs from similar failures

| Scenario | Log evidence | Root cause |
|---|---|---|
| Code lost in migration | `grep auto_join adapter.py` = 0 hits | Feature not present in runtime |
| `intents.members = False` | Handler exists, zero VC EVENT logs | Event silently discarded by discord.py |
| **`auto_join_user_id` in wrong config nest** | `AUTO-JOIN HANDLER called: auto_join=None` | Top-level `discord:` key not bridged into `PlatformConfig.extra` — must live inside `discord.extra:` |
| Missing voice permissions | VC EVENT logged, then permission error | Bot role lacks CONNECT/SPEAK |
| Opus not loaded | VC EVENT logged, then opus warning | Voice channel joined but audio fails |
| `auto_join_user_id` mismatch | VC EVENT logged, "not configured user" | Wrong user ID in config |

**Distinguishing `intents.members = False` from config-placement fault:**
- `intents.members = False`: Zero voice-state logs of any kind. `Voice state: ... joined` never appears because discord.py discards the event before it reaches the adapter.
- Config-placement fault: `Voice state: ... joined` **does** appear, plus `AUTO-JOIN HANDLER called: auto_join=None` — discord.py dispatched the event, handler ran, but `_auto_join_user_id` was `None` because config key was not bridged to `extra`.

## Related

- `references/voice-auto-join-pattern.md` — auto-join config, detection sequence, post-join workflow
- `references/voice-channel-diagnostic-pattern.md` — manual-join diagnostics, infrastructure verification
- `references/service-repo-coherence.md` — when code is present but the running process loads a different copy