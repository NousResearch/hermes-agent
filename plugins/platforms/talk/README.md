# Nextcloud Talk platform plugin

Adds **Nextcloud Talk** as a first-class Hermes messaging platform, following the
plugin path in `gateway/platforms/ADDING_A_PLATFORM.md` (zero core changes).

Hermes agents can chat in Nextcloud Talk rooms exactly like Telegram/Slack —
native sessions, tools, cron, memory.

## Install

Drop this directory in `~/.hermes/plugins/talk/` (or `plugins/platforms/talk/`
for a bundled plugin), then enable it:

```yaml
plugins:
  enabled: [talk]

platforms:
  talk:
    enabled: true
    extra:
      base_url: "https://your-nextcloud.example.com"
      secret: "<40-128 char shared secret, matches the registered Talk bot>"
      port: 8646          # container-internal webhook port
      webhook_path: /talk
      allow_all: true     # or use TALK_ALLOWED_USERS
```

Register the bot in Nextcloud (as a Talk admin):

```
occ talk:bot:install "MyAgent" "<secret>" "http://<host>:8646/talk" "..." -f webhook -f response
occ talk:bot:setup <botId> <roomToken>
```

Nextcloud must be able to reach `http://<host>:<port>/talk`.

## Protocol (verified against the spreed source)

- **Inbound** (Talk → bot): headers `X-Nextcloud-Talk-Random` (≥32 chars) +
  `X-Nextcloud-Talk-Signature` = `hex(HMAC-SHA256(secret, random + raw_body))`.
  Body is an ActivityStreams event (`type:"Create"`, `object.name:"message"`).
- **Outbound** (bot → Talk): `POST {base}/ocs/v2.php/apps/spreed/api/v1/bot/
  {roomToken}/message`, headers `X-Nextcloud-Talk-Bot-Random` +
  `X-Nextcloud-Talk-Bot-Signature` = `hex(HMAC-SHA256(secret, random + message_text))`
  (signs the message text, not the JSON body).
- **Loop guard:** ignores bot-authored messages (`actor.type == "bots"`).

## Notes

- Ack the webhook `200` immediately; the agent turn runs in the background so
  Nextcloud doesn't time out and disable the bot.
- `connect(self, *, is_reconnect=False, **kwargs)` — accepts newer Hermes'
  `is_reconnect` kwarg and older no-arg calls.
- `edit_message` is unsupported (Talk bots can't edit); set
  `display.platforms.talk.streaming: false` for a single clean reply.

Verified in production driving 6 agents across single-gateway and profile-based
containers on Hermes 0.17.0 and 0.18.2.
