# Discord Gateway

The Discord platform plugin connects Hermes Agent to Discord through the
[`discord.py`](https://github.com/Rapptz/discord.py) gateway. It supports
guild channels, threads, direct messages, slash commands, reactions, media,
voice mode, and channel-specific skills and prompts.

## Requirements

- A Discord application and bot
- The bot token
- Python Discord support (`discord.py` is installed automatically when lazy
  dependency installation is enabled)

The bot must have permission to view channels, read message history, send
messages, attach files, and use application commands. Enable these privileged
intents in the Discord Developer Portal under **Bot → Privileged Gateway
Intents**:

- **Message Content Intent** — required for reading message text
- **Server Members Intent** — required when using username or role allowlists

## Setup

Run the standard Hermes setup flow:

```bash
hermes setup
```

Select **Discord**, enter the bot token, and optionally configure allowed users
and a home channel. You can also configure the token directly in
`~/.hermes/.env`:

```dotenv
DISCORD_BOT_TOKEN=your-bot-token
```

Start the gateway with:

```bash
hermes gateway start
```

Never commit the token or place it in `config.yaml`.

## Configuration

Discord settings can be configured under `discord` in
`~/.hermes/config.yaml`. Environment variables remain supported for backwards
compatibility and secrets.

```yaml
discord:
  require_mention: true
  thread_require_mention: false
  allow_from:
    - "123456789012345678"
  allowed_roles:
    - "987654321098765432"
  allowed_channels:
    - "123456789012345678"
  ignored_channels:
    - "234567890123456789"
  free_response_channels:
    - "345678901234567890"
  auto_thread: false
  reactions: true
  reply_to_mode: first
  extra:
    slash_commands: true
```

The most commonly used environment variables are:

| Variable | Purpose |
|---|---|
| `DISCORD_BOT_TOKEN` | Bot credential; required |
| `DISCORD_ALLOWED_USERS` | Comma-separated user IDs or usernames |
| `DISCORD_ALLOWED_ROLES` | Comma-separated role IDs |
| `DISCORD_ALLOWED_CHANNELS` | Restrict processing to these channel IDs |
| `DISCORD_IGNORED_CHANNELS` | Ignore these channel IDs |
| `DISCORD_ALLOW_ALL_USERS` | Disable user filtering; development only |
| `DISCORD_HOME_CHANNEL` | Default destination for cron and notifications |
| `DISCORD_REQUIRE_MENTION` | Require a mention in guild channels |
| `DISCORD_THREAD_REQUIRE_MENTION` | Require mentions in threads |
| `DISCORD_FREE_RESPONSE_CHANNELS` | Channels where messages need no mention |
| `DISCORD_AUTO_THREAD` | Automatically create response threads |
| `DISCORD_REACTIONS` | Enable processing and completion reactions |
| `DISCORD_REPLY_TO_MODE` | `off`, `first`, or `all` reply references |
| `DISCORD_PROXY` | Proxy used by the Discord client |

YAML configuration takes profile isolation into account when Hermes runs in
multiplex mode. Explicit environment variables take precedence over YAML
values.

## Security model

Discord access is fail-closed by default. Configure at least one of the
following before expecting user messages to be accepted:

- an allowed user list;
- an allowed role list;
- an allowed channel list; or
- `DISCORD_ALLOW_ALL_USERS=true` for a controlled development setup.

Component interactions, including approval buttons and model pickers, use the
same authorization policy. Keep bot permissions and allowlists as narrow as
possible. Role lookups require the Server Members intent.

## Features

### Messages and threads

Hermes handles guild messages, DMs, Discord threads, edits, deletes, replies,
and multi-message responses. Long responses are split safely at Discord's
message limit. Optional missed-message backfill can recover eligible messages
after a gateway outage when enabled in the Discord configuration.

### Slash commands and interactive views

The plugin registers Hermes slash commands, including session controls,
model selection, status, help, approvals, and voice controls. Interactive
buttons and select menus are authorized per user and expire safely when their
underlying request is no longer active.

### Media

Inbound images, documents, audio, and other supported attachments are cached
and exposed to the agent. Outbound images, files, video, voice, and multiple
images are uploaded through Discord's attachment API. Local `file://` image
URLs are supported on Windows and Unix-like systems.

### Voice mode

Voice commands can join or leave a voice channel, play text-to-speech output,
and optionally listen for speech. Voice support requires a working FFmpeg
installation and the optional voice dependencies. The adapter cleans up voice
clients, receivers, mixers, and timeout tasks on disconnect.

### Cron and standalone delivery

The plugin can deliver cron and notification messages to `DISCORD_HOME_CHANNEL`.
When no live adapter is available, Hermes uses the plugin's authenticated
standalone REST sender.

## Development

The adapter is intentionally a composition facade. Runtime responsibilities
are organized under:

- `commands/` — slash command implementations;
- `events/` — Discord event registration and normalization;
- `services/` — lifecycle, authorization, messaging, state, voice, recovery,
  and configuration services;
- `views/` — buttons, menus, and interactive approval views;
- `setup.py` — plugin setup, configuration bridging, and registration.

Run Discord tests with the repository test runner rather than bare `pytest`:

```bash
scripts/run_tests.sh tests/gateway/test_discord_*.py
```

See `plugin.yaml` for the plugin manifest and the gateway documentation for
platform-wide lifecycle, secret-scope, and adapter contracts.
