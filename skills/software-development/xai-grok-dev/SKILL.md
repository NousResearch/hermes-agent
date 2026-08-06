---
name: xai-grok-dev
description: "Use when developing or debugging Hermes xAI/Grok provider."
version: 0.1.0
author: Axl Ibiza, MBA
license: MIT
metadata:
  hermes:
    tags: [xAI, Grok, OAuth, provider, Campaign]
    related_skills: [feature-parity-alignment-campaigns, hermes-agent-troubleshooting]
---

# xAI / Grok developer skill

Campaign home: **NousResearch/hermes-agent#80424** (`Campaign: Grok/xAI`).
Always re-measure paths and line counts at current `origin/main`.

## Surface map (constellation)

| Area | Paths |
|---|---|
| Auth / OAuth | `hermes_cli/auth.py` (xAI-dense), `hermes_cli/xai_retirement.py`, tests `tests/hermes_cli/test_xai_oauth_*.py` |
| Model provider plugin | `plugins/model-providers/xai/` |
| Proxy adapter | `hermes_cli/proxy/adapters/xai.py` |
| Image | `plugins/image_gen/xai/` |
| Video | `plugins/video_gen/xai/`, `tools/xai_video_tools.py` |
| Web / X | `plugins/web/xai/`, `tools/xai_http.py` |
| Shared HTTP | `tools/xai_http.py` |
| TTS | `tools/tts_tool.py` (xAI streaming path) |
| Aux client | `agent/auxiliary_client.py` (xai-oauth transport choice) |
| User guide | `website/docs/guides/xai-grok-oauth.md` |
| Optional skill | `optional-skills/autonomous-ai-agents/grok/` |

## Official docs

- https://docs.x.ai/overview
- https://docs.x.ai/developers/models
- https://docs.x.ai/developers/rest-api-reference/inference/chat
- https://docs.x.ai/developers/model-capabilities/text/reasoning
- https://docs.x.ai/developers/tools/function-calling
- https://docs.x.ai/developers/model-capabilities/imagine
- https://docs.x.ai/developers/model-capabilities/audio/voice

Local campaign corpus (when present): `C:/tmp/xai-campaign/docs/`.

## Auth modes

1. **API key** — `xai` provider / `XAI_API_KEY`
2. **OAuth** — `xai-oauth` device/PKCE flow, refresh, credential pool writethrough

Common failure classes (interlock, do not re-file):
- fallback_providers × OAuth JWT — #54671
- refresh trusts retired endpoint — #68694
- fleet lineage revoke — #77553
- token body uncapped — #55000
- shared multi-root OAuth — #65394

## Chat / Responses

- Prefer Responses-style path for grok-4.5+; Chat Completions fallback is a known footgun (#62881).
- MCP tool schema: `$ref` / nullable unions (#67131), multiline string args (#58345).
- SuperGrok prompt-cache vs weekly quota honesty (#72624) — measure before claiming.

## Multimodal

- Image: unguarded URL fetch is P1 (#44728); Invalid PNG session brick (#69078).
- Video: ephemeral URL must be persisted locally (#57206); OAuth-only fail-closed (#78560).
- TTS: streaming wire protocol must match live xAI (#73985).

## Web / X

- Explicit backend must win over xAI auto-routing (#79177).
- OpenRouter `:online` must not duplicate client `web_search` (#76481).
- `x_search` disable must override credential auto-enable (#68001).

## Flagship — Grok Build runtime

Opt-in parity with Codex App-Server Runtime (#65343):
- Launch official `grok agent stdio` (ACP) — do not vendor/fork Grok Build.
- Hermes keeps CLI/TUI/gateway/transcript/session/skills shell.
- Fail closed without CLI or credentials.

## Diagnostics

```bash
gh api "search/issues?q=repo:NousResearch/hermes-agent+is:issue+is:open+label:provider/xai" --jq .total_count
git ls-tree -r --name-only origin/main | grep -iE 'xai|grok'
git show origin/main:hermes_cli/proxy/adapters/xai.py | wc -l
```

## Campaign rules

- Evidence: docs.x.ai anchor + `file:line` at main.
- Interlock exact `#N`; never duplicate.
- One PR per bug class; suite green; attribution preserved.
- xAI cluster extraction from core god-files coordinates with Kill-All-Gods #78647 (FILE-LIST first).
