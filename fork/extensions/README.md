# Fork Extensions

Official Hermes extends via **plugins** and **skills**. This fork adds operator
plugins, a few **core tool** files, and toolset entries replayed after upstream merge.

## Layout

| Area | Official | Fork additions |
|------|----------|----------------|
| Plugins | `plugins/*` (many ship in-tree) | Same tree — enable in `~/.hermes/config.yaml` → `plugins.enabled` |
| Skills | `skills/`, `optional-skills/` | Custom skills under category dirs; heavy deps → `optional-skills/` |
| Core tools | `tools/*.py` + `toolsets.py` | `harness_tools.py`, `vrchat_osc_tool.py`, `voicevox_tts_tool.py`, evolution tools |
| Model providers | `plugins/model-providers/*` | `freellmapi`, `freebuff`, fork routing presets |
| Web search | catalog plugins | Default: `plugins/web/cloakbrowser/` |

## Representative fork plugins

Enable via `hermes tools` or `plugins.enabled` in config:

| Plugin | Path | Notes |
|--------|------|-------|
| VRChat autonomy | `plugins/vrchat-autonomy/` | OSC + safety ACK for live moves |
| LM Twitterer | `plugins/lm-twitterer/` | X posting; cron needs long `script_timeout_seconds` |
| LINE AI bot | `plugins/line_ai_bot/` | Conversation policy plugin |
| AITuber OnAir | `plugins/aituber_onair/` | Hakua / Galaxy sessions |
| Irodori TTS | `plugins/irodori_tts/` | Local TTS server integration |
| OpenClaw vendor | `plugins/openclaw-vendor/` | Vendor sync surface |
| QuestFrame | `plugins/questframe_fh6vr/` | FH6VR quest tooling |
| FreeBuff / FreeLLMAPI | `plugins/freebuff/`, `plugins/model-providers/freellmapi/` | Free tier routing |

Full list: `glob plugins/**/plugin.yaml` in repo.

## Toolset replay

Fork tool names are injected into `toolsets.py` via merge overlay sanitizer
after upstream changes `_HERMES_CORE_TOOLS` ordering. Do not add fork tools only
in a local edit without updating `hermes-merge-conflict-strategies.json`.

## Third-party product policy

New observability/SaaS backends should ship as **standalone** plugin repos under
`~/.hermes/plugins/`, not in this tree (see root AGENTS.md). Existing in-tree
plugins (kanban, observability, …) are grandfathered.

See [`AGENTS.md`](AGENTS.md) for implementation rules.
