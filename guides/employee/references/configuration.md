# Configuration, Toolsets & Voice

Administrators edit with the native dashboard, `hermes config edit` or `hermes config set section.key value`. Agent-requested service connections use the native MCP workflow in `references/native-mcp.md`.
Full reference: https://hermes-agent.nousresearch.com/docs/user-guide/configuration

### Config Sections (most-used keys)

| Section | Key options |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length`, `aliases` |
| `agent` | `max_turns`, `tool_use_enforcement`, `service_tier`, `verify_on_stop` |
| `terminal` | `backend` (local/docker/ssh/modal/daytona/singularity), `cwd`, `timeout` (180) |
| `compression` | `enabled`, `threshold`, `target_ratio` |
| `display` | `skin`, `interface` (cli/tui), `language`, `show_reasoning`, `show_cost`, `pet` |
| `approvals` | `mode` (smart/manual/off), `timeout`, `cron_mode` |
| `stt` | `enabled`, `provider` (local/groq/openai/mistral/elevenlabs/deepinfra) |
| `employee` | `name`, `instructions`, `owner`, `identity_links` |
| `memory` | Authored shared/per-person scopes and Hindsight are fixed; see `references/memory-and-learning.md` |
| `security` | `redact_secrets`, `tirith_enabled`, `website_blocklist` |
| `delegation` | `model`, `provider`, `max_concurrent_children`, `oneshot_max_children`, `max_iterations`, `max_spawn_depth` |
| `checkpoints` | `enabled`, `max_snapshots` (50) |

`hermes config check` reports sections missing from an older config.

### Toolsets

Enable/disable via `hermes tools` (interactive) or `hermes tools enable/disable NAME`.
The fixed employee tool policy filters configured toolsets. Configuration cannot restore excluded tools. The actual schemas are authoritative; configured MCP tools are supported.

| Toolset | What it provides |
|---------|-----------------|
| `web` / `search` | Web search + extraction / search-only subset |
| `browser` | `browser_exec` through Browser Use Cloud |
| `terminal` | Shell commands and process management |
| `file` | File read/write/search/patch |
| `code_execution` | Sandboxed Python execution |
| `vision` | Image analysis |
| `image_gen` | Image generation and image-to-image editing |
| `video` | Video analysis |
| `memory` | Authored shared and per-person memory |
| `messaging` | Send an additional message (`send_message`) |
| `session_search` | Search past conversations |
| `delegation` | Subagent task delegation |
| `todo` | In-session task planning |

Hindsight contributes the `recall` tool through the fixed memory-provider integration.

Tool changes take effect on `/reset` (new session) — never mid-conversation, to preserve prompt caching.

## Voice

### STT (Voice → Text)

Voice messages from messaging platforms are auto-transcribed.

```yaml
stt:
  enabled: true
  provider: local   # local (faster-whisper, free) | groq | openai | mistral | elevenlabs | deepinfra
  local:
    model: base     # tiny, base, small, medium, large-v3
```

This deployment selects local faster-whisper explicitly. Install/repair it through PM (`python -c "import pm; pm.sync_venv(['stt-whisper'], explicit=True)"`).

### TTS (Text → Voice)

| Provider | Env var | Free? |
|----------|---------|-------|
| Edge TTS (default) | None | Yes |
| ElevenLabs | `ELEVENLABS_API_KEY` | Free tier |
| OpenAI | `VOICE_TOOLS_OPENAI_KEY` | Paid |
| MiniMax | `MINIMAX_API_KEY` | Paid |
| Mistral | `MISTRAL_API_KEY` | Paid |
| Gemini | `GOOGLE_API_KEY` | Free tier |
| NeuTTS / Piper / KittenTTS (local) | None | Free |

Voice commands: `/voice on` (voice-to-voice), `/voice tts` (always voice), `/voice off`.
