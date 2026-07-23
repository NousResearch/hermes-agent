# OpenCode Zen Model Catalog (May 2026)

## Full Model List

```
claude-opus-4-7
claude-opus-4-6
claude-opus-4-5
claude-opus-4-1
claude-sonnet-4-6
claude-sonnet-4-5
claude-sonnet-4
claude-haiku-4-5
gemini-3.1-pro
gemini-3-flash
gpt-5.5
gpt-5.5-pro
gpt-5.4
gpt-5.4-pro
gpt-5.4-mini
gpt-5.4-nano
gpt-5.3-codex-spark
gpt-5.3-codex
gpt-5.2
gpt-5.2-codex
gpt-5.1
gpt-5.1-codex-max
gpt-5.1-codex
gpt-5.1-codex-mini
gpt-5
gpt-5-codex
gpt-5-nano
glm-5.1
glm-5
minimax-m2.7
minimax-m2.5
kimi-k2.6
kimi-k2.5
qwen3.6-plus
qwen3.5-plus
big-pickle
deepseek-v4-flash-free
qwen3.6-plus-free
minimax-m2.5-free
ring-2.6-1t-free
trinity-large-preview-free
nemotron-3-super-free
```

## Routing Quirks

| Requested Model | Resolves To | Issue |
|---|---|---|
| `minimax-m2.5-free` | `minimax/minimax-m2.7-20260318` | **M2.7 uses reasoning — content:null with low tokens** |
| `minimax-m2.5` | `minimax/minimax-m2.5-20260218` | ✅ Standard output |
| `minimax-m2.7` | `minimax/minimax-m2.7-20260318` | M2.7 — extended reasoning, content in `reasoning` field |

## Pitfall: M2.7 Reasoning Behavior

M2.7 is a reasoning model. Its response structure:
```json
{
  "choices": [{
    "message": {
      "content": null,
      "reasoning": "thinking text here..."
    },
    "finish_reason": "length"
  }]
}
```

With `max_tokens: 10`, all 10 tokens go to `reasoning`, leaving `content: null`.

**Symptoms in Claude Code:** `Retrying in Xs...` loop, `Wandering...` status, total task failure.

**Fix:** Either (a) use `minimax-m2.5` (non-free) or (b) set `maxTokens: 2048` to give reasoning enough room before content fills.

## Query Model Catalog

```bash
curl -s https://opencode.ai/zen/v1/models \
  -H "Authorization: Bearer <YOUR-API-KEY>"
```

Returns all available models with IDs.

## Model Selection Priority

For coding tasks via OpenCode Zen:
1. **`minimax-m2.5`** — default, cheap ($0.12/M in), standard output, no reasoning overhead
2. **`minimax-m2.5-lightning`** — faster, 1M context, 2.4x output cost
3. **`kimi-k2.6`** — fast, no reasoning, alternative to M2.5
4. **`claude-sonnet-4-6`** — premium quality, 25x costlier

For this user: Hermes and Claude Code both route through OpenCode Zen with `minimax-m2.5`.