# Discord Voice Opus Linux Fallback Loading Pattern

Discovered: 2026-05-27 (Misa-Misa voice auto-join investigation)

## Symptom

Bot auto-joins the voice channel successfully (or is manually joined via `/voice join`), but never hears or responds to speech. No transcription appears in the text channel. Log shows no `VoiceReceiver` or transcript activity after join.

All infrastructure checks pass (Opus package installed, `libopus.so.0` present on the system), yet `discord.opus.is_loaded()` returns `False` inside the bot process.

## Root cause

`ctypes.util.find_library("opus")` fails in restricted or systemd-launched environments even when Opus is installed and `ldconfig -p` lists it. `find_library` depends on the `LD_LIBRARY_PATH` and shell tool access, both of which may be stripped or unavailable in systemd services.

Without Opus loaded, `discord.py` cannot decode RTP audio packets. The voice receiver buffers packets but never produces PCM → no silence detection → no transcription → no response.

## Detection

```bash
# System-level Opus is installed
ldconfig -p | grep opus
# → libopus.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libopus.so.0

# But inside the bot process or venv:
python3 -c "import discord; print(discord.opus.is_loaded())"
# → False ← the tell
```

## Fix: add Linux-specific fallback paths

In `plugins/platforms/discord/adapter.py`, inside the `connect()` method or voice-initialisation block where Opus is loaded:

```python
# After standard load attempt
if not discord.opus.is_loaded():
    import ctypes.util
    # find_library often fails in systemd/restricted envs
    _linux_paths = [
        "/usr/lib/x86_64-linux-gnu/libopus.so.0",
        "/usr/lib/x86_64-linux-gnu/libopus.so",
        "/lib/x86_64-linux-gnu/libopus.so.0",
        "/usr/lib64/libopus.so.0",
        "/usr/lib/libopus.so.0",
    ]
    for opus_path in _linux_paths:
        try:
            discord.opus.load_opus(opus_path)
            if discord.opus.is_loaded():
                break
        except Exception:
            continue
```

Key design points:
- The `_linux_paths` list is declared **before** the loop to satisfy Python static-analysis tools (avoid "possibly unbound" warnings).
- The `.so.0` suffix is used on Debian/Ubuntu because `libopus.so` (unversioned symlink) is typically in the `-dev` package, whereas `.so.0` is in the runtime package (`libopus0`).
- If `is_loaded()` becomes `True` at any point, the loop exits early.

## Verification after fix

```bash
# Restart the gateway, then check in-process state
python3 -c "
import discord, sys, os
os.chdir('/home/kensei/repos/KenseiAgent')
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
from hermes_cli.plugins import get_plugin_manager
get_plugin_manager().discover_and_load(force=True)
import hermes_plugins.discord_platform.adapter as adapter
print('opus loaded:', discord.opus.is_loaded())
"
```

## When this applies

- Any systemd-managed Hermes Discord bot that uses voice features
- After any venv migration where the new environment inherits the upstream code but not live-patched fallback paths
- When `find_library` is known to be unreliable (containerised, sandboxed, or systemd-launched Python)

## Related

- `references/discord-voice-members-intent-silent-discard.md` — the other half of the auto-join silent-failure pattern: events never fire because `intents.members = False`
- `references/voice-channel-diagnostic-pattern.md` — full infrastructure and permission checklist for voice workflows