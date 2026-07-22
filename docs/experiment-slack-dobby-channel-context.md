# Slack Dobby channel context and noise control

## What changed

Slack now supports `channel_session_scope_channels`, a per-channel option that scopes top-level channel mentions to the shared channel session while still replying in the original Slack thread. Slack's default display profile also disables long-running heartbeat posts and busy-ack details, so workspace channels do not get permanent messages like `Working — 9 min — iteration 12/90, terminal`.

## Why

The `#carby-dobby` operator channel needs two behaviors at once: replies should stay threaded for Slack readability, and follow-up top-level mentions should keep context from the channel. Before this change, `reply_in_thread=true` converted every top-level mention into its own synthetic thread session, so a follow-up like "make the PRs" had no memory of the previous "come up with experiments" request.

The same Slack display defaults also inherited medium-tier progress notifications. That made long-running agent work leave noisy operational breadcrumbs in the channel, including iteration counts and terminal status, even when the final useful answer or blocker was the only message the user needed.

## Root cause

Slack top-level messages were using the message timestamp as a synthetic `thread_ts` whenever `reply_in_thread=true`. That was correct for isolated threaded support conversations, but wrong for operator channels where Peter often sends related top-level follow-ups. There was no narrower option between "one session per Slack thread" and "flat channel replies."

## What was tried

The first implementation scoped configured channel messages to the shared channel session by clearing `thread_ts`, but that also cleared `reply_to_message_id`. The test caught that regression because Hermes would have lost the Slack reply anchor and answered flat in the channel. The final implementation keeps `source.thread_id=None` for session scope while setting `reply_to_message_id` to the top-level message timestamp for threaded Slack delivery.

## Key files

- `plugins/platforms/slack/adapter.py` handles Slack event session scoping and reply anchoring.
- `gateway/config.py` bridges top-level `slack.channel_session_scope_channels` into the Slack platform config.
- `gateway/display_config.py` defines Slack's default chatter level.
- `tests/gateway/test_slack_channel_session_scope.py` proves shared session scope can coexist with threaded Slack replies.
- `tests/gateway/test_slack_mention.py` proves the config key is bridged.
- `tests/gateway/test_display_config.py` proves Slack's default heartbeat/busy-ack behavior stays quiet.

## Gotchas for future work

Do not use `reply_in_thread=false` just to preserve context in operator channels. That changes Slack delivery and makes replies flat. Use `channel_session_scope_channels` for channels like `#carby-dobby` where the channel itself is the working context but individual answers should still land as thread replies.

Do not re-enable `long_running_notifications` or `busy_ack_detail` globally for Slack unless the channel explicitly wants permanent operational noise. Slack posts are durable workspace messages, not an ephemeral terminal status area.

## Verification

Run the focused regression suite:

```bash
scripts/run_tests.sh tests/gateway/test_display_config.py tests/gateway/test_slack_channel_session_scope.py tests/gateway/test_slack_mention.py -q
```

Expected result: all tests pass, including `test_configured_channel_can_share_session_while_replying_in_thread`, `test_config_bridges_slack_channel_session_scope_channels`, and `test_slack_workspace_chatter_defaults`.

Read back the live Dobby config:

```bash
HERMES_HOME=/Users/peter/.hermes venv/bin/python - <<'PY'
from pathlib import Path
import yaml
from gateway.config import Platform, load_gateway_config
from gateway.display_config import resolve_display_setting

raw = yaml.safe_load(Path('/Users/peter/.hermes/config.yaml').read_text()) or {}
cfg = load_gateway_config()
slack = cfg.platforms[Platform.SLACK]
override = slack.channel_overrides.get('C0BJU6Q2890')
print("restart_notification", slack.gateway_restart_notification)
print("channel_scope", slack.extra.get("channel_session_scope_channels"))
print("long_running", resolve_display_setting(raw, "slack", "long_running_notifications"))
print("busy_ack", resolve_display_setting(raw, "slack", "busy_ack_detail"))
print("channel_prompt_prefix", (override.system_prompt if override else "")[:80])
PY
```

Expected result: `restart_notification False`, `channel_scope ['C0BJU6Q2890']`, `long_running False`, `busy_ack False`, and a `channel_prompt_prefix` starting with `You are Dobby in #carby-dobby`.

Check the gateway after restart:

```bash
hermes gateway status
rg 'Socket Mode connected|Gateway shutting down|Working — .*iteration|inbound message|response ready|ERROR|Traceback' ~/.hermes/logs/gateway.log | tail -n 50
```

Expected result: the gateway is supervised and connected; new channel runs do not produce `Working — ... iteration` posts or shutdown interruption posts in Slack.

## Verified live surface

- Slack route: BUILDFAST `#carby-dobby`, channel ID `C0BJU6Q2890`.
- Code path: Slack Socket Mode event enters `SlackAdapter._handle_slack_message`, becomes a `MessageEvent`, then the gateway session key is derived from `SessionSource`.
- Surface files intentionally touched: Slack adapter session/reply logic, gateway config bridging, Slack display defaults, and focused tests.
- Stale surfaces intentionally not touched: launchd scheduling and raw cron job definitions, because the issue was gateway Slack behavior rather than job timing.
