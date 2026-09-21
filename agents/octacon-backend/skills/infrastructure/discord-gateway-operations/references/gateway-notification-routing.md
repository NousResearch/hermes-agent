# Gateway Notification Routing — Internal Architecture

**When this matters:** You're being flooded by `⚠️ Gateway restarting/shutting down` or `♻️ Gateway online` messages across multiple Discord channels, and setting `gateway_restart_notification: false` isn't the right trade-off because you still want notifications in one place.

## The Dual-Loop Problem

The method `_notify_active_sessions_of_shutdown()` in `gateway/run.py` (line 3358 in KenseiAgent) has **two independent notification loops**, both running on every shutdown/restart:

### Loop 1: Active Sessions (lines 3377-3453)

```python
for session_key in active:   # ALL running agents
    dedup_key = (platform_str, chat_id, thread_id)  # per-channel!
    adapter.send(chat_id, msg)  # sends directly to each session's source channel
```

Every Discord channel, forum thread, or DM with an active agent gets its own copy of "⚠️ Gateway restarting — Your current task will be interrupted."

### Loop 2: Home Channels (lines 3459-3503)

```python
for platform, adapter in list(self.adapters.items()):
    home = self.config.get_home_channel(platform)
    adapter.send(home.chat_id, msg)
```

Each platform's home channel gets a second copy. The `notified` set deduplicates only on `(platform, chat_id, thread_id)`, so Loop 2 skips channels already notified by Loop 1, but Loop 1 has already flooded N channels.

### Net result

| Gateway state | Active sessions | Home channel | Total notifications |
|---|---|---|---|
| 5 Discord channels with active agents + home channel set | 5 | 1 (dedup'd) | 5 |
| 3 Discord channels + 2 Telegram chats active | 5 | 2 (1 per platform) | 6 |

## Why this exists

The original design assumed single-channel usage — if you only chat in one Discord channel, sending the shutdown notification to the active session IS sending it to the right place. The home channel loop acts as a broadcast for idle platforms. In a multi-channel/multi-bot setup, this assumption breaks.

## The dedup key

```python
dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
```

This deduplicates *identical delivery targets*, not *platforms*. Different Discord channels have different `chat_id`s, so they each get their own notification. There is currently no per-platform dedup — one notification per active session per channel.

## Startup side

`_send_home_channel_startup_notifications()` (line 14387) is **home-channel-only** — it iterates platform adapters and sends `♻️ Gateway online — Hermes is back and ready.` to each platform's home channel. No per-active-session loop exists on startup. This means:

- Shutdown: floods N active channels + homes → N+ total
- Startup: sends to home channels only → 1 per platform

The asymmetry is intentional — startup doesn't know which sessions to resume yet — but it means shutdown is always the noisier side.

## What `gateway_restart_notification` actually controls

The flag is checked in **three** places:

| Check | Effect when false |
|---|---|
| `_notify_active_sessions_of_shutdown` active session loop (line 3422) | Skips Loop 1 entirely for that platform |
| `_notify_active_sessions_of_shutdown` home channel loop (line 3465) | Skips Loop 2 for that platform |
| `_send_home_channel_startup_notifications` (line 14408) | Suppresses `♻️ Gateway online` for that platform |

A single `gateway_restart_notification: false` in the Discord platform config silences ALL three for Discord. There is no granular "send to home channel but not to active sessions" option.

## Verification pattern

To confirm which channels would receive a shutdown notification right now:

```bash
# 1. Check home channel config
grep 'DISCORD_HOME_CHANNEL' /home/kensei/.hermes/.env

# 2. Check which sessions are active (agent state)
#    Look for running gateway processes with active conversations

# 3. Check gateway_restart_notification flag across all profiles
grep -rn 'gateway_restart_notification' \
  /home/kensei/.hermes/config.yaml \
  /home/kensei/.hermes/profiles/*/config.yaml

# 4. For a specific profile, confirm the flag is wired correctly
grep -A3 'extra:' /home/kensei/.hermes/profiles/<profile>/config.yaml
```

## Related

- `Gateway lifecycle notification suppression` in the main SKILL.md — operational guidance for setting the flag
- The `gateway_restart_notification` field in `gateway/config.py` PlatformConfig dataclass (line 299) — default is `True`
- Config parser bridging in `gateway/config.py` (lines 329-331) — checks both top-level and `extra:` nesting
