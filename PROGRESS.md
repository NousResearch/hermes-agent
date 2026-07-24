# PROGRESS — feat/discord-single-reply-channel

## Investigation
- Branch created: feat/discord-single-reply-channel.
- Config precedent found: `_discord_free_response_channels()` in plugins/platforms/discord/adapter.py (~L5750) reads `self.config.extra.get("free_response_channels")` falling back to env `DISCORD_FREE_RESPONSE_CHANNELS`. Will mimic with `reply_channel` / `DISCORD_REPLY_CHANNEL`.
- Config bridge: gateway/config.py ~L1462 copies `free_response_channels` from config.yaml discord: section into adapter extra dict. Add `reply_channel` there.
- Send path: `async def send(...)` at adapter.py L2858 resolves thread/channel then posts chunks. Redirect hook goes right after channel resolution (~L2897), before forum handling.
- Defaults: hermes_cli/config.py ~L2523/2540/2640 have discord default key blocks — add `reply_channel: ""` alongside.
- Test model: tests/gateway/test_discord_allowed_channels.py exists.

## Implementation
- adapter.py: added `_discord_reply_channel()` + `_reply_redirect_target(channel)` helpers (near free_response helper) and a redirect hook in `send()` after channel resolution: resolves reply channel, prefixes "[re: #origin]", drops cross-channel reply_to. DMs (no guild), the reply channel itself, and threads under it are exempt; threads under other channels redirect.
- gateway/config.py: bridge `reply_channel` from discord: config section into adapter extra (Discord only).
- hermes_cli/config.py: default `discord.reply_channel: ""` documented in defaults block.

## Tests
- New: tests/gateway/test_discord_reply_channel.py (11 tests) using load_plugin_adapter("discord") — covers config parsing (unset/config/numeric-YAML/env fallback) and redirect rules (off, other channel, DM, reply channel itself, thread under reply channel, thread under other channel, name fallback to ID).
- Repo venv lacked pytest (python3.14 homebrew too); created .testvenv (not committed).
- Run: .testvenv/bin/python -m pytest tests/gateway/test_discord_allowed_channels.py tests/gateway/test_discord_reply_channel.py -x -q
  Output tail:
  ..........................                                               [100%]
  26 passed in 0.67s

## Docs
- website/docs/user-guide/messaging/discord.md: added `discord.reply_channel` section after free_response_channels (rules: DMs exempt, reply-in-place in reply channel + its threads, other threads redirect, empty = off).
