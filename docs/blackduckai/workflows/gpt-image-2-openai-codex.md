# GPT-Image-2 via OpenAI Codex OAuth

BlackDuckAi's current Hermes default for Shopee/Facebook ad images is the `openai-codex` image-generation backend.

## Config

```yaml
image_gen:
  provider: openai-codex
  model: gpt-image-2-high # or gpt-image-2-medium
  openai-codex:
    model: gpt-image-2-high # or gpt-image-2-medium
```

## Auth model

- `openai-codex` uses Hermes Codex/ChatGPT OAuth.
- It does not require `OPENAI_API_KEY`.
- Keep the legacy `openai` image backend only as an explicit fallback; that provider uses OpenAI Platform API-key billing and requires `OPENAI_API_KEY`.
- Do not print OAuth tokens, API keys, cookies, or credential file contents.

## Operational guidance

For Shopee Affiliate Facebook creative workflow:

```text
facebook-image-prompt-generator
↓
image_generate
↓
provider: openai-codex
↓
model: gpt-image-2-high or gpt-image-2-medium
↓
save manifest
↓
facebook post after TOP approval
```

Use `gpt-image-2-high` for final high-value/ad creatives. Use `gpt-image-2-medium` when TOP prioritizes throughput or cost control.

## Verification

Before reporting an image-generation blocker:

1. Check the active provider.
2. If provider is `openai-codex`, verify Codex/ChatGPT OAuth/provider readiness and model resolution.
3. If provider is `openai`, verify OpenAI Platform API-key and billing readiness.
4. If required images are incomplete, stop before real Facebook posting and write a partial manifest.

Do not say `OPENAI_API_KEY` is mandatory for `gpt-image-2` unless the active provider is specifically the legacy `openai` backend.
