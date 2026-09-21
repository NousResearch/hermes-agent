# Discord Voice Config Bridge: Three-Layer Fix Pattern

Discovered: 2026-05-27 (Misa-Misa voice auto-join re-investigation)

## Context

After a previously-working voice auto-join workflow broke, three separate issues were found in a single diagnostic session. Fixing any one in isolation did not restore auto-join. This reference documents the three layers, how they interact, and the single log line that confirms all three are correct.

## The three layers

| Layer | Component | Failure mode | User-visible symptom |
|---|---|---|---|
| 1 | Config placement | `auto_join_user_id` placed at top-level `discord:` instead of inside `discord.extra:` | `_auto_join_user_id` is `None` at runtime; config key invisible to adapter |
| 2 | Config bridge | `_apply_yaml_config` hook returns `None` instead of a dict bridging custom keys | Same as layer 1 (`auto_join=None`) but config is correctly nested; hook is the culprit |
| 3 | Intent flag | `intents.members` not forced to `True` when `auto_join_user_id` is configured | Zero voice-state logs of any kind; `discord.py` silently discards `VOICE_STATE_UPDATE` events |

All three must be correct for auto-join to work. Fixing layers 1+2 without layer 3 means the event never fires. Fixing layer 3 without 1+2 means the event fires but the handler sees `auto_join=None` and exits early.

## Layer-by-layer diagnostics

### Layer 1: Config placement

Keys must live inside `discord.extra:` in the profile's `config.yaml`:

```yaml
discord:
  extra:
    auto_join_user_id: '797682085224513547'
```

Top-level placement under `discord:` (outside `extra:`) is invisible because `gateway/config.py` only copies a fixed whitelist into `PlatformConfig.extra`.

**Detection:** `grep "auto_join_user_id" ~/.hermes/profiles/<profile>/config.yaml` — if the line is NOT indented under `extra:`, it's wrong.

### Layer 2: Config bridge via `_apply_yaml_config`

Even correctly nested keys may not reach `PlatformConfig.extra` if the adapter's `_apply_yaml_config(raw_config)` hook returns `None`.

**How the bridge works:**

1. Hermes parses `config.yaml` → creates `PlatformConfig`.
2. Fixed whitelist keys from `discord:` are copied into `PlatformConfig.extra`.
3. The adapter's `_apply_yaml_config(raw_config)` is called.
4. **If it returns a `dict`**, Hermes merges it into `PlatformConfig.extra`.
5. **If it returns `None`**, nothing extra is merged.

**Detection:**
```bash
grep -A10 "_apply_yaml_config" /path/to/adapter.py | grep "return None"
```
Any `return None` in that method = layer 2 fault.

**Fix:** Change the hook to return a dict of voice keys:

```python
async def _apply_yaml_config(self, raw_config: dict) -> dict | None:
    discord_cfg = raw_config.get("discord", {})
    extras = discord_cfg.get("extra", {})
    bridged = {}
    for key in ("auto_join_user_id", "auto_join_text_channel_id",
                "auto_join_greeting_text", "auto_leave_on_user_exit",
                "voice_timeout_seconds"):
        # Prefer extra, fall back to top-level
        value = extras.get(key, discord_cfg.get(key))
        if value is not None:
            bridged[key] = value
    return bridged if bridged else None
```

### Layer 3: `intents.members` forced ON

`discord.py` 2.7.1 silently discards `VOICE_STATE_UPDATE` events for members not in cache when `intents.members = False`. The adapter's `on_voice_state_update` handler is never called.

**In the adapter code (e.g. `connect()` method):**

```python
intents.members = (
    any(not entry.isdigit() for entry in self._allowed_user_ids)
    or bool(self._allowed_role_ids)
    or bool(self._auto_join_user_id)  # ← REQUIRED layer
)
```

**Detection:** After gateway restart, grep for the composite intents line:

```bash
tail -5 /home/kensei/.hermes/profiles/<profile>/logs/gateway.log | grep "Intents configured"
```

If it shows `members=False`, layer 3 is broken.

## The definitive verification log line

After restart, the adapter should emit exactly one line summarising all voice-relevant config:

```
[Discord] Intents configured: members=True voice_states=True auto_join_user_id=797682085224513547
```

| Token | Meaning |
|---|---|
| `members=True` | Layer 3 fixed — `discord.py` will cache members and dispatch `VOICE_STATE_UPDATE` |
| `voice_states=True` | Baseline — voice state intent enabled (required for all voice functionality) |
| `auto_join_user_id=<value>` | Layers 1+2 fixed — config key is correctly placed AND bridged by `_apply_yaml_config` |

If ANY component is wrong, the line changes accordingly:

```
members=False voice_states=True auto_join_user_id=None
```
→ Layer 3 (intents.members not forced) AND/OR layers 1+2 (config not visible).

```
members=True voice_states=True auto_join_user_id=None
```
→ Layer 1 or 2: config placement or bridge fault.

```
members=False voice_states=True auto_join_user_id=<value>
```
→ Layer 3 only: config is visible but intents.members not forced.

**Always check for this line immediately after restart. It saves iterative user-testing round-trips.**

## Why manual `/voice channel` can still work when auto-join is broken

The `/voice channel` slash command uses `_handle_voice_channel_join()`, which directly calls `join_voice_channel()` with the channel from the interaction. It never checks `_auto_join_user_id`. This means manual join passing **does NOT** prove layers 1-2 are correct — it only proves voice infrastructure (Opus, permissions, PyNaCl) is functional.

If a user reports "manual join works but auto-join doesn't", immediately check all three layers above. Do not assume the config is correct just because manual join succeeds.

## Relation to Opus loading

This three-layer fix is distinct from Opus loading issues. A bot with all three layers correct but Opus not loaded will:
- Auto-join the voice channel successfully
- Never hear or transcribe speech
- Show no transcript activity

See `references/discord-voice-opus-linux-fallback.md` for the Opus-specific diagnostic.

## Related

- `references/voice-auto-join-pattern.md` — full auto-join config reference, detection sequence, and post-join workflow
- `references/discord-voice-members-intent-silent-discard.md` — detailed analysis of `intents.members = False` causing silent event discard
- `references/discord-voice-opus-linux-fallback.md` — Opus loading failure in systemd environments
- `references/service-repo-coherence.md` — when patched code exists in the repo but not the service runtime
