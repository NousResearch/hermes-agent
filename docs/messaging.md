# Messaging Platform Integrations (Gateway)

Hermes Agent can connect to messaging platforms like Telegram, Discord, and WhatsApp to serve as a conversational AI assistant.

## Quick Start

```bash
# 1. Set your bot token(s) in .env file
echo 'TELEGRAM_BOT_TOKEN="your_telegram_bot_token"' >> .env
echo 'DISCORD_BOT_TOKEN="your_discord_bot_token"' >> .env

# 2. Test the gateway (foreground)
./scripts/hermes-gateway run

# 3. Install as a system service (runs in background)
./scripts/hermes-gateway install

# 4. Manage the service
./scripts/hermes-gateway start
./scripts/hermes-gateway stop
./scripts/hermes-gateway restart
./scripts/hermes-gateway status
```

**Quick test (without service install):**
```bash
python cli.py --gateway  # Runs in foreground, useful for debugging
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Hermes Gateway                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Telegram │ │ Discord  │ │ WhatsApp │ │  Slack   │           │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │             │            │             │                │
│       └─────────────┼────────────┼─────────────┘                │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  Session Store  │                            │
│                  │  (per-chat)     │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │   AIAgent       │                            │
│                  │   (run_agent)   │                            │
│                  └─────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Session Management

### Session Persistence

Sessions persist across messages until they reset. The agent remembers your conversation context.

### Reset Policies

Sessions reset based on configurable policies:

| Policy | Default | Description |
|--------|---------|-------------|
| Daily | 4:00 AM | Reset at a specific hour each day |
| Idle | 120 min | Reset after N minutes of inactivity |
| Both | (combined) | Whichever triggers first |

### Manual Reset

Send `/new` or `/reset` as a message to start fresh.

### Per-Platform Overrides

Configure different reset policies per platform:

```json
{
  "reset_by_platform": {
    "telegram": { "mode": "idle", "idle_minutes": 240 },
    "discord": { "mode": "idle", "idle_minutes": 60 }
  }
}
```

## Platform Setup

### Telegram

1. **Create a bot** via [@BotFather](https://t.me/BotFather)
2. **Get your token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
3. **Set environment variable:**
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token_here"
   ```
4. **Optional: Set home channel** for cron job delivery:
   ```bash
   export TELEGRAM_HOME_CHANNEL="-1001234567890"
   export TELEGRAM_HOME_CHANNEL_NAME="My Notes"
   ```

**Requirements:**
```bash
pip install python-telegram-bot>=20.0
```

### Discord

1. **Create an application** at [Discord Developer Portal](https://discord.com/developers/applications)
2. **Create a bot** under your application
3. **Get the bot token**
4. **Enable required intents:**
   - Message Content Intent
   - Server Members Intent (optional)
5. **Invite to your server** using OAuth2 URL generator (scopes: `bot`, `applications.commands`)
6. **Set environment variable:**
   ```bash
   export DISCORD_BOT_TOKEN="your_token_here"
   ```
7. **Optional: Set home channel:**
   ```bash
   export DISCORD_HOME_CHANNEL="123456789012345678"
   export DISCORD_HOME_CHANNEL_NAME="#bot-updates"
   ```

**Requirements:**
```bash
pip install discord.py>=2.0
```

### WhatsApp

WhatsApp uses a built-in bridge powered by [Baileys](https://github.com/WhiskeySockets/Baileys) that connects via WhatsApp Web. The agent links to your WhatsApp account and responds to incoming messages.

**Setup:**

```bash
hermes whatsapp
```

This will:
- Enable WhatsApp in your `.env`
- Ask for your phone number (for the allowlist)
- Install bridge dependencies (Node.js required)
- Display a QR code — scan it with your phone (WhatsApp → Settings → Linked Devices → Link a Device)
- Exit automatically once paired

Then start the gateway:

```bash
hermes gateway
```

The gateway starts the WhatsApp bridge automatically using the saved session credentials in `~/.hermes/whatsapp/session/`.

**Environment variables:**

```bash
WHATSAPP_ENABLED=true
WHATSAPP_ALLOWED_USERS=15551234567    # Comma-separated phone numbers with country code
```

Agent responses are prefixed with "⚕ **Hermes Agent**" so you can distinguish them from your own messages when messaging yourself.

> **Re-pairing:** If WhatsApp Web sessions disconnect (protocol updates, phone reset), re-pair with `hermes whatsapp`.

## Configuration

There are **three ways** to configure the gateway (in order of precedence):

### 1. Environment Variables (`.env` file) - Recommended for Quick Setup

Add to your `~/.hermes/.env` file:

```bash
# =============================================================================
# MESSAGING PLATFORM TOKENS
# =============================================================================

# Telegram - get from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ALLOWED_USERS=123456789,987654321    # Security: restrict to these user IDs

# Optional: Default channel for cron job delivery
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"

# Discord - get from Discord Developer Portal
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_ALLOWED_USERS=123456789012345678      # Security: restrict to these user IDs

# Optional: Default channel for cron job delivery
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"

# Slack - get from Slack API (api.slack.com/apps)
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token      # Required for Socket Mode
SLACK_ALLOWED_USERS=U01234ABCDE                # Security: restrict to these user IDs

# Optional: Default channel for cron job delivery
# SLACK_HOME_CHANNEL=C01234567890

# WhatsApp - pair via: hermes whatsapp
WHATSAPP_ENABLED=true
WHATSAPP_ALLOWED_USERS=15551234567             # Phone numbers with country code

# =============================================================================
# AGENT SETTINGS
# =============================================================================

# Max tool-calling iterations per conversation (default: 60)
HERMES_MAX_ITERATIONS=60

# Working directory for terminal commands (default: home ~)
MESSAGING_CWD=/home/myuser

# =============================================================================
# TOOL PROGRESS NOTIFICATIONS
# =============================================================================

# Show progress messages as agent uses tools
HERMES_TOOL_PROGRESS=true

# Mode: "new" (only when tool changes) or "all" (every tool call)
HERMES_TOOL_PROGRESS_MODE=new

# =============================================================================
# SESSION SETTINGS
# =============================================================================

# Reset sessions after N minutes of inactivity (default: 120)
SESSION_IDLE_MINUTES=120

# Daily reset hour in 24h format (default: 4 = 4am)
SESSION_RESET_HOUR=4
```

### 2. Gateway Config File (`~/.hermes/gateway.json`) - Full Control

For advanced configuration, create `~/.hermes/gateway.json`:

```json
{
  "platforms": {
    "telegram": {
      "enabled": true,
      "token": "your_telegram_token",
      "home_channel": {
        "platform": "telegram",
        "chat_id": "-1001234567890",
        "name": "My Notes"
      }
    },
    "discord": {
      "enabled": true,
      "token": "your_discord_token",
      "home_channel": {
        "platform": "discord",
        "chat_id": "123456789012345678",
        "name": "#bot-updates"
      }
    }
  },
  "default_reset_policy": {
    "mode": "both",
    "at_hour": 4,
    "idle_minutes": 120
  },
  "reset_by_platform": {
    "discord": {
      "mode": "idle",
      "idle_minutes": 60
    }
  },
  "always_log_local": true
}
```

## Platform-Specific Toolsets

Each platform has its own toolset for security:

| Platform | Toolset | Capabilities |
|----------|---------|--------------|
| CLI | `hermes-cli` | Full access (terminal, browser, etc.) |
| Telegram | `hermes-telegram` | Full tools including terminal |
| Discord | `hermes-discord` | Full tools including terminal |
| WhatsApp | `hermes-whatsapp` | Full tools including terminal |
| Slack | `hermes-slack` | Full tools including terminal |

## User Experience Features

### Typing Indicator

The gateway keeps the "typing..." indicator active throughout processing, refreshing every 4 seconds. This lets users know the bot is working even during long tool-calling sequences.

### Tool Progress Notifications

When `HERMES_TOOL_PROGRESS=true`, the bot sends status messages as it works:

```
💻 `ls -la`...
🔍 web_search...
📄 web_extract...
🎨 image_generate...
```

Terminal commands show the actual command (truncated to 50 chars). Other tools just show the tool name.

**Modes:**
- `new`: Only sends message when switching to a different tool (less spam)
- `all`: Sends message for every single tool call

### Working Directory

- **CLI (`hermes` command)**: Uses current directory where you run the command
- **Messaging**: Uses `MESSAGING_CWD` (default: home directory `~`)

This is intentional: CLI users are in a terminal and expect the agent to work in their current directory, while messaging users need a consistent starting location.

### Max Iterations

If the agent hits the max iteration limit while working, instead of a generic error, it asks the model to summarize what it found so far. This gives you a useful response even when the task couldn't be fully completed.

## Voice Messages (TTS)

The `text_to_speech` tool generates audio that the gateway delivers as native voice messages on each platform:

| Platform | Delivery | Format |
|----------|----------|--------|
| Telegram | Voice bubble (plays inline) | Opus `.ogg` — native from OpenAI/ElevenLabs, converted via ffmpeg for Edge TTS |
| Discord | Audio file attachment | MP3 |
| WhatsApp | Audio file attachment | MP3 |
| CLI | Saved to `~/voice-memos/` | MP3 |

**Providers:**
- **Edge TTS** (default) — Free, no API key, 322 voices in 74 languages
- **ElevenLabs** — Premium quality, requires `ELEVENLABS_API_KEY`
- **OpenAI TTS** — Good quality, requires `OPENAI_API_KEY`

Voice and provider are configured by the user in `~/.hermes/config.yaml` under the `tts:` key. The model only sends text; it does not choose the voice.

The tool returns a `MEDIA:<path>` tag that the gateway send pipeline intercepts and delivers as a native audio message. If `[[audio_as_voice]]` is present (Opus format available), Telegram sends it as a voice bubble instead of an audio file.

**Telegram voice bubbles & ffmpeg:**

Telegram requires Opus/OGG format for native voice bubbles (the round, inline-playable kind). **OpenAI and ElevenLabs** produce Opus natively when on Telegram — no extra setup needed. **Edge TTS** (the default free provider) outputs MP3 and needs `ffmpeg` to convert:

```bash
sudo apt install ffmpeg    # Ubuntu/Debian
brew install ffmpeg         # macOS
sudo dnf install ffmpeg     # Fedora
```

Without ffmpeg, Edge TTS audio is sent as a regular audio file (still playable, but shows as a rectangular music player instead of a voice bubble).

## Cron Job Delivery

Cron jobs are executed automatically by the gateway daemon. When the gateway is running (via `hermes gateway` or `hermes gateway install`), it ticks the scheduler every 60 seconds and runs due jobs.

When scheduling cron jobs, you can specify where the output should be delivered:

```
User: "Remind me to check the server in 30 minutes"

Agent uses: schedule_cronjob(
  prompt="Check server status...",
  schedule="30m",
  deliver="origin"  # Back to this chat
)
```

### Delivery Options

| Option | Description |
|--------|-------------|
| `"origin"` | Back to where the job was created |
| `"local"` | Save to local files only |
| `"telegram"` | Telegram home channel |
| `"discord"` | Discord home channel |
| `"telegram:123456"` | Specific Telegram chat |

## Dynamic Context Injection

The agent knows where it is via injected context:

```
## Current Session Context

**Source:** Telegram (group: Dev Team, ID: -1001234567890)
**Connected Platforms:** local, telegram, discord

**Home Channels:**
  - telegram: My Notes (ID: -1001234567890)
  - discord: #bot-updates (ID: 123456789012345678)

**Delivery options for scheduled tasks:**
- "origin" → Back to this chat (Dev Team)
- "local" → Save to local files only
- "telegram" → Home channel (My Notes)
- "discord" → Home channel (#bot-updates)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/platforms` | Show gateway configuration and status |
| `--gateway` | Start the gateway (CLI flag) |

## Troubleshooting

### "python-telegram-bot not installed"

```bash
pip install python-telegram-bot>=20.0
```

### "discord.py not installed"

```bash
pip install discord.py>=2.0
```

### "No platforms connected"

1. Check your environment variables are set
2. Check your tokens are valid
3. Try `/platforms` to see configuration status

### Session not persisting

1. Check `~/.hermes/sessions/` exists
2. Check session policies aren't too aggressive
3. Verify no errors in gateway logs

## Adding a New Platform

To add a new messaging platform:

### 1. Create the adapter

Create `gateway/platforms/your_platform.py`:

```python
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.config import Platform, PlatformConfig

class YourPlatformAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.YOUR_PLATFORM)
    
    async def connect(self) -> bool:
        # Connect to the platform
        ...
    
    async def disconnect(self) -> None:
        # Disconnect
        ...
    
    async def send(self, chat_id: str, content: str, ...) -> SendResult:
        # Send a message
        ...
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Get chat information
        ...
```

### 2. Register the platform

Add to `gateway/config.py`:

```python
class Platform(Enum):
    # ... existing ...
    YOUR_PLATFORM = "your_platform"
```

### 3. Add to gateway runner

Update `gateway/run.py` `_create_adapter()`:

```python
elif platform == Platform.YOUR_PLATFORM:
    from gateway.platforms.your_platform import YourPlatformAdapter
    return YourPlatformAdapter(config)
```

### 4. Create a toolset (optional)

Add to `toolsets.py`:

```python
"hermes-your-platform": {
    "description": "Your platform toolset",
    "tools": [...],
    "includes": []
}
```

### 5. Configure

Add environment variables to `.env`:

```bash
YOUR_PLATFORM_TOKEN=...
YOUR_PLATFORM_HOME_CHANNEL=...
```

## Service Management

### Linux (systemd)

```bash
# Install as user service
./scripts/hermes-gateway install

# Manage
systemctl --user start hermes-gateway
systemctl --user stop hermes-gateway
systemctl --user restart hermes-gateway
systemctl --user status hermes-gateway

# View logs
journalctl --user -u hermes-gateway -f

# Enable lingering (keeps running after logout)
sudo loginctl enable-linger $USER
```

### macOS (launchd)

```bash
# Install
./scripts/hermes-gateway install

# Manage
launchctl start ai.hermes.gateway
launchctl stop ai.hermes.gateway

# View logs
tail -f ~/.hermes/logs/gateway.log
```

### Manual (any platform)

```bash
# Run in foreground (for testing/debugging)
./scripts/hermes-gateway run

# Or via CLI (also foreground)
python cli.py --gateway
```

## Interrupting the Agent

Send any message while the agent is working to interrupt it. The message becomes the next prompt after the agent stops. Key behaviors:

- **In-progress terminal commands are killed immediately** -- SIGTERM first, SIGKILL after 1 second if the process resists. Works on local, Docker, SSH, Singularity, and Modal backends.
- **Tool calls are cancelled** -- if the model generated multiple tool calls in one batch, only the currently-executing one runs. The rest are skipped.
- **Multiple messages are combined** -- if you send "Stop!" then "Do X instead" while the agent is stopping, both messages are joined into one prompt (separated by newline).
- **`/stop` command** -- interrupts without queuing a follow-up message.
- **Priority processing** -- interrupt signals bypass command parsing and session creation for minimal latency.

## Storage Locations

| Path | Purpose |
|------|---------|
| `~/.hermes/gateway.json` | Gateway configuration |
| `~/.hermes/sessions/sessions.json` | Session index |
| `~/.hermes/sessions/{id}.jsonl` | Conversation transcripts |
| `~/.hermes/cron/output/` | Cron job outputs |
| `~/.hermes/logs/gateway.log` | Gateway logs (macOS launchd) |
