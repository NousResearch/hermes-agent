# Hermes Agent — Feature Pruning Guide

Generated: 2026-05-04  
Branch: feat/qwen-aware-compaction  
Purpose: Identify what can be safely removed for a CLI-first, local-model-focused, research-capable installation.

---

## Tier 1 — Safe to Delete (code removal, zero risk to core)

These have no reverse dependencies on anything a CLI-first setup needs.

### Messaging Platforms (`gateway/platforms/`)

| Platform | Files | Notes |
|---|---|---|
| Telegram | `telegram.py`, `telegram_network.py` | |
| Discord | `discord.py` | also `tools/discord_tool.py` |
| Slack | `slack.py` | |
| WhatsApp | `whatsapp.py` | |
| Signal | `signal.py`, `signal_rate_limit.py` | |
| Matrix/Element | `matrix.py` | |
| Email | `email.py` | |
| SMS | `sms.py` | |
| DingTalk | `dingtalk.py` | |
| Feishu | `feishu.py`, `feishu_comment.py`, `feishu_comment_rules.py` | also `tools/feishu_doc_tool.py`, `tools/feishu_drive_tool.py` |
| WeCom | `wecom.py`, `wecom_callback.py`, `wecom_crypto.py` | |
| Weixin | `weixin.py` | |
| QQBot | `qqbot/` (directory) | |
| Yuanbao | `yuanbao.py`, `yuanbao_proto.py`, `yuanbao_media.py`, `yuanbao_sticker.py` | also `tools/yuanbao_tools.py` |
| BlueBubbles | `bluebubbles.py` | macOS iMessage relay |
| Mattermost | `mattermost.py` | |
| Home Assistant | `homeassistant.py` | also `tools/homeassistant_tool.py` |
| Webhook (generic) | `webhook.py` | unless actively used |

**Gateway subsystem:** If no platforms are kept, `gateway/` (excluding `gateway/platforms/api_server.py` if TUI is kept) can be deleted entirely.

---

### Plugins (`plugins/`)

| Plugin | Directory | Notes |
|---|---|---|
| mem0 | `plugins/memory/mem0/` | keep one memory plugin if using memory |
| SuperMemory | `plugins/memory/supermemory/` | |
| OpenViking | `plugins/memory/openviking/` | |
| Hindsight | `plugins/memory/hindsight/` | |
| Holographic | `plugins/memory/holographic/` | |
| RetainDB | `plugins/memory/retaindb/` | |
| ByteRover | `plugins/memory/byterover/` | |
| Langfuse observability | `plugins/observability/langfuse/` | unless tracing LLM calls |
| Camofox supervisor | `plugins/camofox_supervisor/` | anti-detection browser, niche |
| Kanban | `plugins/kanban/` | also `tools/kanban_tools.py`, `tools/todo_tool.py` |
| Google Meet | `plugins/google_meet/` | |
| Spotify | `plugins/spotify/` | |
| Hermes Achievements | `plugins/hermes-achievements/` | gamification |
| Strike Freedom Cockpit | `plugins/strike-freedom-cockpit/` | |
| Example Dashboard | `plugins/example-dashboard/` | reference only |
| Disk Cleanup | `plugins/disk-cleanup/` | low priority |

---

### Voice Features
- `tools/voice_mode.py`
- `tools/tts_tool.py`
- `tools/transcription_tools.py`
- `hermes_cli/voice.py`

---

### Browser Anti-Detection & Niche Browser Tools
- `tools/browser_camofox.py`
- `tools/browser_cdp_tool.py`
- `tools/browser_dialog_tool.py`
- `tools/browser_providers/browserbase/` (cloud browser service)

Keep: `tools/browser_tool.py`, `tools/browser_supervisor.py`, `tools/browser_providers/firecrawl/` if web browsing is needed.

---

### Unused Model Adapters
- `agent/codex_responses_adapter.py` — legacy OpenAI Codex API
- `agent/copilot_acp_client.py` — GitHub Copilot ACP
- `agent/gemini_cloudcode_adapter.py` — Cloud Code variant (keep `gemini_native_adapter.py`)
- `agent/moonshot_schema.py` — if not using Kimi/Moonshot

---

### Skills (Inactive Categories)
- `optional-skills/` — entire directory, inactive by default
- From `skills/`: prune `apple/`, `social-media/`, `gaming/`, `gifs/`, `yuanbao/`, `domain/`, `smart-home/` (if no Home Assistant)

---

### Research Tools (only if NOT needed)
> **Do NOT delete if doing RL/training work — these are the research value of Hermes.**
- `batch_runner.py` — parallel batch processing for datasets
- `trajectory_compressor.py` — post-run trajectory compression for training
- `environments/` — Atropos RL environments, SWE-bench suites

---

## Tier 2 — Disable via Config (keep code, don't load)

| Feature | How to disable |
|---|---|
| Honcho memory plugin | Remove from `plugins.enabled` in config |
| Cron scheduler | `cron.enabled: false` |
| TUI | Just don't run `hermes --tui`; no change needed |
| Context compression | `compression.enabled: false` per model |
| Autonomous Curator | Disable in agent config |
| Web search tools | `tools.web.enabled: false` |
| MCP servers | Remove all entries from `mcp_servers` in config |
| Prompt caching | `prompt_caching.enabled: false` (probably leave on) |
| Cost tracking | `track_costs: false` |

---

## Tier 3 — Keep Everything (core + research use case)

| Component | Why keep |
|---|---|
| `run_agent.py` | Core agent loop |
| `cli.py` | CLI interface |
| `hermes_state.py` | Session persistence + FTS5 search |
| `tools/registry.py` | Tool discovery and dispatch |
| `tools/terminal_tool.py` | Shell execution |
| `tools/file_tools.py` | File I/O |
| `tools/code_execution_tool.py` | Python sandbox — foundation for CodeAct mode |
| `tools/web_tools.py` | Web search |
| `tools/session_search_tool.py` | Cross-session recall |
| `tools/delegate_tool.py` | Subagent spawning |
| `tools/memory_tool.py` | Memory interface |
| `tools/skill_manager_tool.py` + `skills_tool.py` | Procedural memory |
| `tools/approval.py` | Security |
| `tools/interrupt.py` | Control |
| `agent/context_compressor.py` | Qwen-aware compaction |
| `agent/compaction_result.py` | Compaction metrics |
| `agent/auxiliary_client.py` | Cheap model routing |
| `agent/anthropic_adapter.py` | Anthropic support |
| `agent/gemini_native_adapter.py` | Gemini support |
| `agent/curator.py` | Autonomous skill curation |
| `agent/skill_utils.py` | Skills infrastructure |
| `agent/memory_manager.py` | Memory orchestration |
| `hermes_cli/config.py` | Configuration |
| `hermes_cli/profiles.py` | Multi-profile support |
| `hermes_logging.py` | Logging |
| `batch_runner.py` | Research/training |
| `trajectory_compressor.py` | Training data |
| `environments/` | Atropos RL |
| OpenRouter, Anthropic, Gemini adapters | Model access |

---

## Estimated Size Reduction

| Category | Files removed | Approx reduction |
|---|---|---|
| 18 platform adapters | ~20 files | ~50k+ LOC |
| Gateway subsystem | ~10 files | ~5k LOC |
| 7 memory plugins | ~14 dirs | ~3k LOC |
| Optional skills | ~15 dirs | ~2k LOC |
| Voice features | ~4 files | ~1k LOC |
| Misc plugins | ~8 dirs | ~2k LOC |
| Unused model adapters | ~4 files | ~1.5k LOC |
| **Total** | **~60+ files/dirs** | **~30-40% of total codebase** |

---

## Suggested Pruning Order

1. Delete all messaging platform adapters + `gateway/` runner — biggest cut, zero risk
2. Delete Tier 1 plugins (memory alternatives, niche plugins)
3. Delete voice features
4. Delete unused optional-skills and skill categories
5. Disable Tier 2 features via config
6. Run `hermes doctor` after each step to catch broken imports

---

## Notes

- All features listed as disableable can be toggled without code changes via `~/.hermes/config.yaml`
- Research tools (batch_runner, trajectory_compressor, environments/) should only be deleted if training/RL use is permanently ruled out
- The gateway `api_server.py` is needed if keeping the TUI (`hermes --tui`)
- After deleting platforms, remove their entries from `gateway/platform_registry.py` to avoid import errors
