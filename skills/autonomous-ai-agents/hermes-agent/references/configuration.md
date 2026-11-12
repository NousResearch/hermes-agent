# Configuration, Toolsets & Voice

Edit with `hermes config edit` or `hermes config set section.key value`.
Full reference: https://hermes-agent.nousresearch.com/docs/user-guide/configuration

### Config Sections (most-used keys)

| Section | Key options |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length`, `aliases` |
| `agent` | `max_turns` (90), `tool_use_enforcement`, `service_tier`, `verify_on_stop` |
| `terminal` | `backend` (local/docker/ssh/modal/daytona/vercel_sandbox, or a plugin backend like singularity), `cwd`, `timeout` (180) |
| `compression` | `enabled`, `threshold` (0.50), `target_ratio` (0.20) |
| `display` | `skin`, `interface` (cli/tui), `language`, `show_reasoning`, `show_cost`, `pet` |
| `approvals` | `mode` (smart/manual/off), `timeout`, `cron_mode` |
| `stt` | `enabled`, `provider` (local/groq/openai/mistral/elevenlabs/deepinfra) |
| `tts` | `provider` (edge/elevenlabs/openai/minimax/mistral/neutts/gemini/piper/kittentts/deepinfra/xai) |
| `memory` | `memory_enabled`, `user_profile_enabled`, `provider`, `write_approval` |
| `security` | `redact_secrets`, `website_blocklist` |
| `browser` | `headed` (false), `engine` (auto/lightpanda), `executable_path` ("" = auto-detect Chromium), `user_data_dir` ("" = fresh ephemeral profile per session), `chromium_args` ([]), `command_timeout` (Chromium launch timeout floor), `cloud_provider`, `cdp_url`, `use_real_profile`, `live_chrome` (false; drive your own running Chrome via chrome://inspect) |
| `delegation` | `model`, `provider`, `max_concurrent_children`, `max_iterations` (50), `max_spawn_depth` |
| `checkpoints` | `enabled`, `max_snapshots` (50) |
| `curator` | `enabled`, `consolidate` (false, opt-in aux-model consolidation), `interval_hours`, `stale_after_days` |

`hermes config check` reports sections missing from an older config.

### Setup optimization audit ("what could we improve in my Hermes setup?")

1. Audit read-only first, in parallel: `hermes --version`, `hermes doctor`, `hermes auth list`, `hermes fallback list`, `hermes config show`, size of `~/.hermes/config.yaml` and `~/.hermes/cache/*` (`du -sh`), and `~/.hermes/logs/agent.log` / `gateway.error.log` for recurring warnings. Count real model use with `grep -oE 'API call #[0-9]+: model=[^ ]+ provider=[^ ]+' ~/.hermes/logs/agent.log | awk '{print $4,$5}' | sort | uniq -c` before suggesting model changes.
2. Present candidates as ONE multi-select clarify, each with its mechanism and risk, biggest risk first. Keep every option inside the picker's limit (4 choices); fold or drop the rest into the reply text and say so, because an extra option is silently cut.
3. Apply only what was picked. Report what was checked and what is left for the user.

Candidates that have proven worth checking:

- **Shared Claude Code login.** `auth.adopt_external_logins: true` plus a `claude_code` pool entry means Hermes and Claude Code (including other agents that use the Claude Code login) share single-use OAuth refresh tokens and can log each other out; look for `credential pool: no available entries` in `agent.log`. Fix order matters: the user runs `hermes auth add anthropic --type oauth` (opens a browser), you confirm the new entry with `hermes auth list anthropic`, THEN `hermes config set auth.adopt_external_logins false`. Turning adoption off first leaves no subscription login.
- **Bloated config.yaml.** Copied template comments can grow it past 100 KB and bury real settings. Slim recipe (the one exception to "don't hand-edit", because `hermes config set` cannot drop comments): (1) back up to `~/.hermes/backups/config.yaml.pre-slim-<date>` with `cp -p`, mode 600; (2) read the install directory from the "Install directory:" line of `hermes --version` (if none is reported, stop and do NOT hand-edit config.yaml), and write a probe module there under a clearly temporary name (e.g. `_slim_probe_tmp.py`; the launcher runs Python with `-I` and drops `PYTHONPATH`, so a probe anywhere else won't import) that does `read_user_config_raw()` then `hermes_yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, default_flow_style=False, width=4096)` to a scratch file, reads it back and asserts equality with the original dict; (3) run it with `hermes --run-module <probe>` and delete the probe immediately, before any `hermes update`; (4) only on IDENTICAL, `mv` the scratch file over config.yaml (chmod 600); (5) verify with `hermes config check`, `hermes config get` on the keys that matter (model, fallback_providers, auxiliary.*, delegation.model, display.skin, timezone), `hermes fallback list`, and `hermes gateway status`. Comments are the only loss; secrets are not in config.yaml.
- **Caches.** `rm -rf ~/.hermes/cache/uv` is safe; it rebuilds on demand.
- **`updates.pre_update_backup: true`** makes every `hermes update` zip `~/.hermes` first, protecting local patches. `display.bell_on_prompt` rings the terminal when a clarify/approval is waiting.
- **`hermes config get` warns "unrecognized key"** for some real keys (e.g. `agent.reasoning_effort`); the warning is not proof the setting is ignored, so check the reader in source before acting on it.
- **Provider key permissions.** A tool error like `does not have the 'message_delete' permission` is an API-key scope limit on the provider side, not a Hermes bug; report it, don't work around it.

### Toolsets

Enable/disable via `hermes tools` (interactive) or `hermes tools enable/disable NAME`.
Full enumeration: `TOOLSETS` dict in `toolsets.py` (`_HERMES_CORE_TOOLS` is the default bundle most platforms inherit).

| Toolset | What it provides |
|---------|-----------------|
| `web` / `search` | Web search + extraction / search-only subset |
| `browser` | `browser_exec` (Python CDP helpers), `browser_cdp`, `browser_dialog`, `browser_vault` — local Chromium, a cloud provider, or `/browser connect`. Disable this toolset to turn browsing off |
| `terminal` | Shell commands and process management |
| `file` | File read/write/search/patch |
| `code_execution` | Sandboxed Python execution |
| `coding` | Code-editing helpers (LSP-backed) |
| `computer_use` | Desktop GUI control (cua-driver) |
| `vision` | Image analysis |
| `image_gen` | Image generation and image-to-image editing |
| `video` / `video_gen` | Video analysis / video generation |
| `x_search` | X (Twitter) search (X OAuth or API key) |
| `tts` | Text-to-speech |
| `skills` | Skill browsing and management |
| `memory` | Persistent cross-session memory |
| `session_search` | Search past conversations |
| `context_engine` | Pluggable context-engine hooks |
| `project` | Named multi-folder workspace tools |
| `delegation` | Subagent task delegation |
| `cronjob` | Scheduled task management |
| `clarify` | Ask user clarifying questions |
| `todo` | In-session task planning |
| `kanban` | Multi-agent work-queue tools (gated to workers) |
| `debugging` | Extra introspection tools (off by default) |
| `safe` | Minimal low-risk toolset for locked-down sessions |
| `discord`, `discord_admin`, `feishu_doc`, `feishu_drive`, `yuanbao` | Service integrations (gated on their credentials). `homeassistant` (`ha_*` tools) and `spotify` (`spotify_*` tools) are not built in: they come from the `homeassistant` / `spotify` catalog plugins (`hermes plugins install <name>`) |

Camofox is no longer built in: `hermes plugins install camofox` adds it as the `camofox` toolset with its own browser tools. Existing Camofox setups are migrated automatically on `hermes update`.

Tool changes take effect on `/reset` (new session) — never mid-conversation, to preserve prompt caching.

## Voice

### STT (Voice → Text)

Voice messages from messaging platforms are auto-transcribed.

```yaml
stt:
  enabled: true
  provider: local   # local (faster-whisper, free) | groq | openai | mistral | elevenlabs | deepinfra
  local:
    model: base     # tiny, base, small, medium, large-v3, turbo
```

Auto-detect priority: local faster-whisper (`python -c "import pm; pm.sync_venv(['stt-whisper'], explicit=True)"`) → Groq (`GROQ_API_KEY`, free tier) → OpenAI (`VOICE_TOOLS_OPENAI_KEY`) → Mistral Voxtral (`MISTRAL_API_KEY`).

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
