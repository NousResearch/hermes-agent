# PROGRESS — feat/discord-single-reply-channel

## Investigation
- Branch created: feat/discord-single-reply-channel.
- Config precedent found: `_discord_free_response_channels()` in plugins/platforms/discord/adapter.py (~L5750) reads `self.config.extra.get("free_response_channels")` falling back to env `DISCORD_FREE_RESPONSE_CHANNELS`. Will mimic with `reply_channel` / `DISCORD_REPLY_CHANNEL`.
- Config bridge: gateway/config.py ~L1462 copies `free_response_channels` from config.yaml discord: section into adapter extra dict. Add `reply_channel` there.
- Send path: `async def send(...)` at adapter.py L2858 resolves thread/channel then posts chunks. Redirect hook goes right after channel resolution (~L2897), before forum handling.
- Defaults: hermes_cli/config.py ~L2523/2540/2640 have discord default key blocks — add `reply_channel: ""` alongside.
- Test model: tests/gateway/test_discord_allowed_channels.py exists.
