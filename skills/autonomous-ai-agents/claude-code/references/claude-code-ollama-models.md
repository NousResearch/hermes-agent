# Claude Code + Ollama Models

## Verified Pattern (2026-07-11)

Claude Code can use ollama models via the `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN=ollama` env vars. The model mapping is:

- `sonnet` -> `ANTHROPIC_DEFAULT_SONNET_MODEL` (e.g. minimax-m3:cloud)
- `opus` -> `ANTHROPIC_DEFAULT_OPUS_MODEL` (e.g. glm-5.2:cloud)
- `haiku` -> `ANTHROPIC_DEFAULT_HAIKU_MODEL` (e.g. kimi-k2.7-code:cloud)

## Critical Finding

**`sonnet` (minimax-m3:cloud) FAILS for tool-use tasks through ollama.** It burns all `--max-turns` producing zero useful output. The ollama API compatibility layer doesn't properly handle Claude Code's Anthropic-format tool-use requests with this model.

**`haiku` (kimi-k2.7-code:cloud) WORKS.** It successfully completed a multi-file Rust task (adding types, functions, tests, running cargo verify) in a single pass with 20 max-turns.

**Default model is now glm-5.2:cloud** (mapped to both sonnet and opus). Use `--model haiku` for code tasks that need tool-use.

## Recommendation

For Claude Code + ollama:
- Use `--model haiku` for all code/tool-use tasks
- `glm-5.2:cloud` is the default model (sonnet/opus mapping)
- Avoid `--model sonnet` with minimax-m3:cloud — it does not handle tool-use through ollama
- `--effort high` is fine to pair with haiku
- Keep `--max-turns 20-25` for focused tasks

## Settings

In `~/.claude/settings.json`:
```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11434",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.7-code:cloud"
  }
}
```
