# BlackDuckAi Hermes Profile Model Allocation

TOP approved this balanced model allocation on 2026-05-31. It keeps Red as the single Telegram entry point while assigning each specialist profile to the model lane that best matches its role and workload.

## Auth principles

- Claude profiles use Anthropic through Claude Code OAuth / Claude credential store. `ANTHROPIC_TOKEN` and `ANTHROPIC_API_KEY` should stay empty in those profile env files unless TOP explicitly wants legacy API-key billing.
- GPT chat/tool-heavy profiles use `openai-codex` through Codex/ChatGPT OAuth.
- GPT-Image-2 image generation uses `image_gen.provider: openai-codex` and does not require `OPENAI_API_KEY`.
- Gemini remains API-key backed for research and Veo/video lanes.
- Never commit or document credential values.

## Approved allocation

### default / Red — Chief of Staff

```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6
```

Use for executive coordination, Thai nuance, routing, approval control, and risk-aware memos.

### orchestrator / Nexus — CAIO

```yaml
model:
  provider: openai-codex
  default: gpt-5.5
  base_url: https://chatgpt.com/backend-api/codex
```

Use for agent ecosystem planning, orchestration, roadmap design, and tool-heavy coordination. Move Nexus to Claude only when TOP explicitly approves a premium Claude lane.

### automation / Operator — COO

```yaml
model:
  provider: openai-codex
  default: gpt-5.5
  base_url: https://chatgpt.com/backend-api/codex
```

Use for cron, webhook, SOP, pipeline, and ops automation.

### scribe / Ledger — CFO

```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6
```

Use for finance memos, accounting summaries, cashflow/ROI reasoning, and precise Thai executive writing.

### researcher / Magnet — CMO

```yaml
model:
  provider: gemini
  default: gemini-3.1-pro-preview
  base_url: https://generativelanguage.googleapis.com/v1beta
```

Use for market research, competitor scans, multimodal synthesis, and trend analysis.

### facebook / Closer — CSO

```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6
```

Use for Thai sales copy, Facebook comments/inbox drafts, customer nuance, and conversion-focused replies.

### codexworker / Architect — CTO

```yaml
model:
  provider: openai-codex
  default: gpt-5.5
  base_url: https://chatgpt.com/backend-api/codex
```

Use for coding implementation, repo edits, debugging, API/infrastructure work, and tool-heavy development.

### reviewer / Sentinel — QA, Risk & Review

```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6
```

Use for QA, risk review, security review, final checks, and decision-quality feedback.

### shopee / Merchant — Shopee Affiliate Specialist

```yaml
model:
  provider: openai-codex
  default: gpt-5.5
  base_url: https://chatgpt.com/backend-api/codex
```

Use for product sourcing, affiliate link workflows, Shopee API/tool work, and promotion/mission tracking.

### brandcreative / Studio — Creative & Sales Asset Specialist

```yaml
model:
  provider: openai-codex
  default: gpt-5.5
  base_url: https://chatgpt.com/backend-api/codex

image_gen:
  provider: openai-codex
  model: gpt-image-2-high
  openai-codex:
    model: gpt-image-2-high
```

Use GPT/Codex for creative planning and prompt writing, GPT-Image-2 High via Codex OAuth for images, and Gemini/Veo separately for video rendering.

## Verification checklist

- Parse all edited `config.yaml` files.
- Confirm `hermes auth list` shows `anthropic -> claude_code oauth` and `openai-codex -> oauth`.
- Smoke-test one profile per provider lane:
  - Claude: `CLAUDE_PROFILE_OK`
  - Codex: `CODEX_PROFILE_OK`
  - Gemini: `GEMINI_PROFILE_OK`
- Confirm `openai-codex` image provider resolves `gpt-image-2-high` with high quality.
- Run a secret-value scan over touched config/docs before committing.
- Restart gateway or start fresh profile sessions after changing configs.
