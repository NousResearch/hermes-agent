---
title: Hermes Agent Optional Tooling Status
tags:
  - hermes-agent
  - tooling
  - verification
status: active
updated: 2026-05-20
---

# 10 Optional Tooling Status

## Completed

| Tool / Provider | Status | Evidence |
|---|---|---|
| OpenRouter API | 100 | `hermes doctor` shows `✓ OpenRouter API` |
| MOA | 100 | `hermes doctor` shows `✓ moa` after OpenRouter key was configured |
| Vision | 100 | `hermes doctor` shows `✓ vision` after OpenRouter key was configured |
| Video analysis | 100 | `hermes doctor` shows `✓ video` after OpenRouter key was configured |
| browser-cdp | 100 | Chrome CDP responds at `http://127.0.0.1:9222/json/version`; `hermes doctor` shows `✓ browser-cdp` |
| computer_use binary | 100 | `cua-driver` installed at `/Users/rattanasak/.local/bin/cua-driver`; `hermes doctor` shows `✓ computer_use` |
| Dashboard chat | 100 | `http://127.0.0.1:9119/chat` returns HTTP 200 |

## Requires User Credential Or macOS Permission

| Item | Done % | Remaining % | What is missing |
|---|---:|---:|---|
| CuaDriver Accessibility permission | 50 | 50 | macOS System Settings must grant Accessibility to CuaDriver/terminal process |
| Web search providers | 0 | 100 | No `EXA_API_KEY`, `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, or tool gateway token found |
| x_search | 0 | 100 | No `XAI_API_KEY` or xAI OAuth found |
| image_gen / video_gen | 0 | 100 | No generation backend key found, for example `FAL_KEY` / provider-specific key |
| Feishu | 0 | 100 | No `FEISHU_APP_ID` / `FEISHU_APP_SECRET` found |
| Home Assistant | 0 | 100 | No `HASS_TOKEN` / `HASS_URL` found |
| Yuanbao | 0 | 100 | No `YUANBAO_APP_ID` / `YUANBAO_APP_SECRET` found |
| Discord | skipped | skipped | User does not use it |
| Spotify | skipped | skipped | User does not use it |

## Inspection URLs

- Hermes dashboard chat: http://127.0.0.1:9119/chat
- Chrome CDP health: http://127.0.0.1:9222/json/version
- Hermes Agent Obsidian MOC: obsidian://open?vault=HermesAgent&file=MOC
- Phase delivery report: obsidian://open?vault=HermesAgent&file=reports/2026-05-20-phase-delivery
- Optional tooling report: obsidian://open?vault=HermesAgent&file=reports/2026-05-20-optional-tooling-status
