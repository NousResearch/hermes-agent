---
name: opencode-free-rotation
description: "Refresh OpenCode Zen free models and failover chain."
version: 1.0.0
author: Bob Nyan, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [OpenCode, Free-Models, Failover, Catalog, Zen]
    category: autonomous-ai-agents
    related_skills: [opencode]
---

# OpenCode Free Rotation

Keep Hermes on the current OpenCode Zen free catalog and rotate through free models when usage or token limits are hit.

## When to Use

- OpenCode free models change and you want the live list refreshed
- A free model returns `Free usage exceeded` or similar limit errors
- You need to verify the primary + fallback chain before a long session
- You want a scheduled catalog refresh (cron) to stay current

## Prerequisites

- `OPENCODE_ZEN_API_KEY` in `~/.hermes/.env` (from https://opencode.ai/auth)
- Primary provider set to `opencode-zen` with `model.default: auto-free`
- Optional local rollback: `llama-cpp` on `http://127.0.0.1:8080/v1`

Example config: `docs/migration/opencode_free_webui_config.example.yaml`

## How Rotation Works

1. **Primary** — `auto-free` resolves to the first live free model from `https://opencode.ai/zen/v1/models`.
2. **Runtime failover** — `fallback_providers` entry `{"provider": "opencode-zen", "model": "auto-free"}` expands to the full free sequence (deduped).
3. **Limit detection** — `Free usage exceeded` style errors are classified as transient rate limits, triggering the next model.
4. **Local rollback** — Final chain entry `llama-cpp` uses the designated GGUF via llama-server autostart.

## Quick Reference

| Action | Command |
|--------|---------|
| Refresh catalog | `terminal(command="py -3 scripts/refresh_opencode_free_catalog.py --force")` |
| JSON output | `terminal(command="py -3 scripts/refresh_opencode_free_catalog.py --force --json")` |
| Show fallback chain | `terminal(command="hermes fallback list")` |
| Check llama fallback | `terminal(command="py -3 -c \"from hermes_cli.llama_fallback_runtime import is_llama_fallback_ready; print(is_llama_fallback_ready())\"")` |

## Procedure

### 1. Verify credentials and config

```yaml
model:
  provider: opencode-zen
  default: auto-free

fallback_providers:
  - provider: opencode-zen
    model: auto-free
  - provider: llama-cpp
    model: huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s800-curriculum1280-1152-q8_0.gguf
    base_url: http://127.0.0.1:8080/v1
```

### 2. Refresh the live catalog

Run from the Hermes repo root:

```
terminal(command="py -3 scripts/refresh_opencode_free_catalog.py --force")
```

Compare output with `hermes fallback list`. Hermes expands `auto-free` at runtime — no manual model list edits are required when OpenCode adds new free IDs.

### 3. Schedule optional refresh (cron)

Use `cronjob` to refresh weekly and log drift:

```
cronjob(action="add", schedule="every monday 6am", prompt="Run py -3 scripts/refresh_opencode_free_catalog.py --force --json and summarize any new or removed free model IDs. If the catalog fetch fails, note the static fallback floor in hermes_cli/models.py.", no_agent=false)
```

## Pitfalls

- Catalog fetch requires network access to `opencode.ai`; offline mode uses the static free floor in `hermes_cli/models.py`.
- WebUI reads raw `config.yaml` — `auto-free` displays as-is in the picker but resolves correctly at agent runtime.
- llama autostart only runs when `llama-cpp` appears in `fallback_providers` or `providers.llama-cpp`.

## Verification

```
terminal(command="hermes fallback list")
terminal(command="py -3 scripts/refresh_opencode_free_catalog.py --force")
```

Expect a non-empty free model list and a final `llama-cpp` entry when local rollback is configured.
