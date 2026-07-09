---
name: moneyprinter-video
description: "Use when generating short-form videos, video scripts, stock-footage search terms, or polling MoneyPrinterTurbo Video Studio tasks via MCP tools."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [video, moneyprinter, short-form, tts, subtitles, mcp]
    related_skills: []
---

# MoneyPrinter Video Studio

## Overview

Hermes integrates [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) as a **Desktop capability**, not Agent Core. Agents call it through the **moneyprinter MCP server**, which reuses `capabilities/moneyprinter/adapter.py` (same layer as the Desktop Video Studio page).

## When to Use

- User asks to generate a short video / 短视频 / reels / shorts
- User wants a video script only, or B-roll keywords only
- User asks to check video task status or list finished outputs

**Don't use for:** general image generation, long-form film editing, or non-MoneyPrinter video pipelines.

## Prerequisites

1. MCP server registered (once per profile):

```bash
hermes mcp add moneyprinter --command "python -m capabilities.moneyprinter.mcp.server"
hermes mcp test moneyprinter
```

2. Vendored tree present: `external/MoneyPrinterTurbo/`
3. Config has LLM + material keys (via Desktop Video Studio config or `config.toml`)

## MCP Tools

| Tool | Purpose |
|------|---------|
| `moneyprinter_health_check` | Install / config / service status |
| `moneyprinter_start_service` | Start FastAPI sidecar |
| `moneyprinter_generate_video` | Create full render task (returns task_id) |
| `moneyprinter_get_task` | Poll state / progress / outputs |
| `moneyprinter_list_tasks` | List tasks |
| `moneyprinter_list_outputs` | Flatten completed video URLs |
| `moneyprinter_delete_task` | Delete task |
| `moneyprinter_generate_script` | Script only |
| `moneyprinter_generate_terms` | Search terms only |

## Default Parameters

| Param | Default |
|-------|---------|
| aspect | `9:16` |
| count | `1` |
| language | `zh-CN` (or match user language) |
| source | `pexels` |
| voice | `zh-CN-XiaoxiaoNeural-Female` |
| subtitles | `true` |
| bgm | `random` |
| clip duration | `5` seconds |

Map natural language → params:

- "竖屏/抖音/短视频" → `video_aspect=9:16`
- "横屏/YouTube" → `16:9`
- "英文" → `video_language=en-US` + English voice if known
- User provides full script → pass as `video_script` (skip empty)

## Long-Task Workflow (required)

1. `moneyprinter_health_check(start_if_needed=true)` or `moneyprinter_start_service`
2. `moneyprinter_generate_video(video_subject=...)`
3. Tell the user the **task_id** and that render may take minutes
4. Poll `moneyprinter_get_task(task_id)` (do not busy-loop forever in one turn)
5. On complete: report `streamUrl` / `downloadUrl` / file name
6. On failure: summarize error + next step (missing API key, material rate limit, TTS, ffmpeg)

**Completion criterion for generate:** tool returns `ok=true` and a `task_id`.  
**Completion criterion for deliver:** task `state` is complete/success and at least one video URL exists, **or** clear failure with actionable fix.

## Safety Rules

- Never print API keys from config.toml or health payloads beyond "configured / missing"
- Do not invent endpoints or claim Agent Core has built-in MoneyPrinter tools
- Prefer Hermes-safe adapter URLs (`/api/capabilities/moneyprinter/stream/...`) over raw filesystem paths when available

## Common Failures

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| not installed | missing `external/MoneyPrinterTurbo` | re-vendor / check path |
| upstream unreachable | sidecar not running / env pollution | start service; check health |
| LLM error | missing openai/deepseek/etc key | configure via Desktop Video Studio |
| no materials | pexels/pixabay keys empty or rate-limited | set keys / switch source |
| pydantic_core crash on start | Hermes venv leaked into sidecar | fixed via adapter clean env; restart |

## Verification Checklist

- [ ] MCP tools listed after `hermes mcp test moneyprinter`
- [ ] Health reports `installed=true` and config flags
- [ ] Generate returns task_id without waiting for full mp4
- [ ] Get task eventually returns videos or a clear error
- [ ] No secrets leaked in the user-facing summary
