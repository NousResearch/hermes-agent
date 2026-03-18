---
name: hermes-agent-setup
description: Help users configure Hermes Agent — enable tools, set up voice/STT/TTS, install dependencies, and troubleshoot. Use when someone asks to enable features, configure voice, or when the system detects missing config.
version: 1.0.0
author: Hermes Agent
tags: [setup, configuration, tools, stt, tts, voice, hermes]
---

# Hermes Agent Setup & Configuration

Use this skill when a user asks to enable features, configure voice messages, set up tools, or troubleshoot configuration.

## Key Paths

- Config: `~/.hermes/config.yaml`
- API keys: `~/.hermes/.env`
- Skills: `~/.hermes/skills/`
- Hermes install: `~/.hermes/hermes-agent/`

## Voice Messages (STT)

Voice messages from Telegram/Discord/WhatsApp/Slack/Signal are auto-transcribed when an STT provider is available.

### Provider priority (auto-detected):
1. **Local faster-whisper** — free, no API key, runs on CPU/GPU
2. **Groq Whisper** — free tier, needs GROQ_API_KEY
3. **OpenAI Whisper** — paid, needs VOICE_TOOLS_OPENAI_KEY

### Setup local STT (recommended):

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate  # or: source venv/bin/activate
pip install faster-whisper
```

Add to config.yaml under the `stt:` section:
```yaml
stt:
  enabled: true
  provider: local
  local:
    model: base  # Options: tiny, base, small, medium, large-v3
```

Model downloads automatically on first use (~150 MB for base).

### Setup Groq STT (free cloud):

1. Get free key from https://console.groq.com
2. Add GROQ_API_KEY to the env file
3. Set provider to groq in config.yaml stt section

### Verify STT:

After config changes, restart the gateway (send /restart in chat, or restart `hermes gateway run`). Then send a voice message.

## Voice Replies (TTS)

Hermes can reply with voice when users send voice messages.

### TTS providers (set API key in env file):

| Provider | Env var | Free? |
|----------|---------|-------|
| ElevenLabs | ELEVENLABS_API_KEY | Free tier |
| OpenAI | VOICE_TOOLS_OPENAI_KEY | Paid |
| Kokoro (local) | None needed | Free |
| Fish Audio | FISH_AUDIO_API_KEY | Free tier |

### Voice commands (in any chat):
- `/voice on` — voice reply to voice messages only
- `/voice tts` — voice reply to all messages
- `/voice off` — text only (default)

## Enabling/Disabling Tools

### Interactive tool config (requires terminal):

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
python -m hermes_cli.main tools
```

This opens a curses UI to enable/disable toolsets per platform.

### After changing tools:

Use `/reset` in the chat to start a fresh session with the new toolset. Tool changes do NOT take effect mid-conversation (this preserves prompt caching).

### Common toolsets:

| Toolset | What it provides |
|---------|-----------------|
| terminal | Shell command execution |
| file | File read/write/search/patch |
| web | Web search and extraction |
| browser | Browser automation |
| image_gen | AI image generation |
| mcp | MCP server connections |
| voice | Text-to-speech |
| cronjob | Scheduled tasks |

## Installing Dependencies

Some tools need extra packages:

```bash
cd ~/.hermes/hermes-agent && source .venv/bin/activate

pip install faster-whisper    # Local STT
pip install browserbase       # Browser automation
pip install mcp               # MCP servers
```

## Setup Wizard

For first-time setup or full reconfiguration:

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
python -m hermes_cli.main setup
```

## Gateway Commands

| Command | What it does |
|---------|-------------|
| /reset or /new | Fresh session (picks up new tool config) |
| /help | Show all commands |
| /model [name] | Show or change model |
| /compact | Compress conversation to save context |
| /voice [mode] | Configure voice replies |
| /reasoning [effort] | Set reasoning level |
| /sethome | Set home channel for cron/notifications |
| /restart | Restart the gateway (picks up config changes) |

## Troubleshooting

### Voice messages not working
1. Check stt.enabled is true in config.yaml
2. Check a provider is available (faster-whisper installed, or API key set)
3. Restart gateway after config changes

### Tool not available
1. Check if the toolset is enabled for your platform (run `hermes tools`)
2. Some tools need env vars — check the env file
3. Use /reset after enabling tools

### Changes not taking effect
- Gateway: /reset for tool changes, /restart for config changes
- CLI: start a new session
