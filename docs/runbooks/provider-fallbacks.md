# Capy provider fallback checks

Primary model path stays Brendan's preferred Hermes/OpenAI-compatible GPT-5.5 setup:

- Provider: `openai-codex`
- Model: `gpt-5.5`

Fallbacks must be non-destructive: do not rewrite the global default model unless Brendan explicitly asks. Instead, document readiness and provide profile/session-level commands.

## Fallback tiers

1. Primary hosted path
   - Hermes default provider/model from `~/.hermes/config.yaml`.
   - Expected on Capy Mac Studio: `openai-codex` + `gpt-5.5`.

2. Local/free path
   - LM Studio OpenAI-compatible endpoint: `http://127.0.0.1:1234/v1`
   - Hermes profile alias: `lmstudio`
   - Known local model: `qwen3.6-35b-a3b-ud-mlx`

3. Optional hosted paths
   - OpenRouter, Nous Portal, Anthropic, OpenAI Responses/API.
   - Only activate when credentials are present and the user requests it.

## Readiness checks

Use the checker script:

```bash
cd /Users/bschmidy10/.hermes/hermes-agent
./venv/bin/python3 scripts/capy_provider_fallback_check.py
```

The script must not print API key values. It checks model config, LM Studio health, and credential presence by name only.

## Safe switching patterns

Prefer one-off/session switches first:

```bash
hermes --profile lmstudio "quick smoke prompt"
```

or in a gateway chat:

```text
/model qwen3.6-35b-a3b-ud-mlx --provider lmstudio
```

Avoid global changes unless explicitly requested:

```bash
# Only with approval:
hermes profile use lmstudio
```

## Rollback

If a fallback underperforms, return to primary:

```text
/model gpt-5.5 --provider openai-codex
```

Then verify:

```text
/status
/capy
```
