---
name: hermes-provider-config
description: "Configure LLM providers in Hermes — primary model, fallback chains with per-entry auth, custom endpoints, OpenCode Zen/Go, credential pools. Covers undocumented config.yaml fields."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hermes, providers, fallback, config, model-routing, api-keys, opencode-zen, custom-endpoints]
    related_skills: [hermes-agent, ai-agent-delegation]
---

# Hermes Provider Configuration

Configure LLM providers, fallback chains, and custom endpoints in Hermes Agent. Covers both documented CLI flows and undocumented `config.yaml` fields discovered from source code.

## Quick Commands

```bash
hermes model                        # Interactive model/provider picker (primary)
hermes fallback list                # Show fallback chain
hermes fallback add                 # Interactive picker to add fallback (same as hermes model)
hermes fallback remove              # Remove a fallback entry
hermes fallback clear               # Clear all fallbacks
hermes config set model.default MODEL
hermes config set model.provider PROVIDER
hermes config set model.api_key KEY
hermes config set model.base_url URL
```

## Primary Model Configuration

Stored in `~/.hermes/config.yaml` under `model:` key:

```yaml
model:
  default: mimo-v2.5-pro           # Model identifier
  provider: custom                 # Provider name (openrouter, anthropic, opencode-zen, custom, etc.)
  base_url: https://example.com/v1 # Required for custom providers
  api_key: sk-xxx                  # Or use env var (see provider table in hermes-agent skill)
```

## Fallback Providers

Fallback entries are tried in order when the primary fails (rate-limit, 5xx, connection errors). Stored as `fallback_providers` list in `config.yaml`.

### Basic Format

```yaml
fallback_providers:
  - provider: opencode-zen
    model: qwen3.6-plus-free
```

### Per-Entry Authentication (UNDOCUMENTED)

Each fallback entry supports its own auth, independent of the primary. **The `hermes fallback add` interactive picker does NOT expose these fields — manual config.yaml editing is required.**

Supported fields per entry (from `agent/chat_completion_helpers.py` lines 948-955):

```yaml
fallback_providers:
  - provider: opencode-zen
    model: qwen3.6-plus-free
    api_key: sk-specific-key-for-this-fallback     # Direct key (plaintext in config)
    base_url: https://opencode.ai/zen/v1            # Custom endpoint override
    api_mode: chat_completions                       # API mode override (optional)
```

**Alternative: reference an env var name instead of plaintext key:**

```yaml
fallback_providers:
  - provider: opencode-zen
    model: qwen3.6-plus-free
    key_env: OPENCODE_ZEN_FALLBACK_KEY     # Reads from this env var at runtime
```

Both `key_env` and `api_key_env` are accepted as aliases for the env var name.

### Auth Resolution Priority (per fallback entry)

1. `api_key` field in the entry (direct value)
2. `key_env` or `api_key_env` field → looked up via `os.getenv()`
3. Special case: Ollama Cloud endpoints (`ollama.com` base_url) auto-pull `OLLAMA_API_KEY` from env
4. Provider's default env var (e.g. `OPENCODE_ZEN_API_KEY` for `opencode-zen`)

### Multiple Fallbacks

```yaml
fallback_providers:
  - provider: opencode-zen
    model: claude-sonnet-4-6
  - provider: openrouter
    model: google/gemini-2.5-flash
  - provider: custom
    model: my-local-model
    base_url: http://localhost:11434/v1
```

### Fallback Behavior

- Chain is tried in order on each failure
- Rate-limit (429) triggers a 60-second cooldown on the primary
- Billing errors also trigger fallback
- Same (provider, model, base_url) deduplication prevents loops
- Fallback is **turn-scoped** — primary is restored at the start of each new turn (unless still in rate-limit cooldown)
- Chain index resets each turn; exhausted chain means no more fallbacks that turn

## OpenCode Zen Provider

OpenCode Zen provides curated models via `https://opencode.ai/zen/v1`. Env var: `OPENCODE_ZEN_API_KEY`.

```yaml
# As primary
model:
  provider: opencode-zen
  default: claude-sonnet-4-6

# As fallback (uses OPENCODE_ZEN_API_KEY from .env by default)
fallback_providers:
  - provider: opencode-zen
    model: qwen3.6-plus-free

# As fallback with separate key
fallback_providers:
  - provider: opencode-zen
    model: qwen3.6-plus-free
    api_key: ***
```

Known model identifiers (May 2026): `minimax-m2.7`, `minimax-m2.5`, `minimax-m2.5-free`, `kimi-k2.6`, `qwen3.6-plus`, `qwen3.6-plus-free`, `deepseek-v4-flash-free`, `claude-opus-4-6`, `claude-sonnet-4-6`, `gpt-5.4`. **All bare names — no `provider/` prefix.**

**Model naming quirk:** OpenCode Zen uses bare model names (e.g. `qwen3.6-plus-free`, `minimax-m2.5-free`), NOT `provider/model` format. Using `qwen/qwen3.6-plus-free` may 404 or fail silently. Always cross-reference with the model catalog (`curl -s https://opencode.ai/zen/v1/models -H "Authorization: Bearer $KEY"`). Other providers like OpenRouter DO use `provider/model` format (e.g. `google/gemini-2.5-flash`).

## OpenCode Go Provider

OpenCode Go provides open models via subscription ($10/month). Env var: `OPENCODE_GO_API_KEY`. Base URL: `https://opencode.ai/zen/go/v1`.

## Custom Providers (config.yaml)

For self-hosted or non-standard endpoints:

```yaml
providers:
  my-ollama:
    api: http://127.0.0.1:11434/v1
    default_model: llama3
    models:
      - llama3
      - codellama
    name: Ollama
```

## Pitfalls

1. **Fallback picker doesn't expose api_key/key_env** — must edit config.yaml manually for per-entry auth
2. **`hermes fallback add` overwrites primary temporarily** — it snapshots and restores, but if interrupted the primary model config may be left pointing at the fallback. Fix: re-run `hermes model`
3. **Legacy `fallback_model` (single dict) still supported** — merged with `fallback_providers` list, but `fallback_providers` entries take priority. On write, legacy key is removed
4. **Model identifiers vary by provider** — OpenCode Zen uses bare names (`qwen3.6-plus-free`), OpenRouter uses `provider/model` format (`google/gemini-2.5-flash`). Wrong identifier = silent auth failure or 404. See pitfall #7 for OpenCode Zen specifics
5. **Config changes need restart** — CLI: exit and relaunch. Gateway: `/restart`. Fallback config is read at agent init time, not per-turn
6. **Custom provider `base_url` must end with `/v1` for OpenAI-compatible APIs** — missing path causes 404s
7. **Multiple API keys = separate rate limits** — per-entry `api_key` on fallbacks gives independent rate-limit pools (useful when primary is rate-limited)
7. **OpenCode Zen uses bare model names** — `qwen3.6-plus-free` not `qwen/qwen3.6-plus-free`. Other providers (OpenRouter) use `provider/model` format. Wrong format = 404 or silent failure. Verify with `curl -s https://opencode.ai/zen/v1/models -H "Authorization: Bearer $KEY"`. NOTE: `qwen/qwen3.6-plus-free` (with prefix) HAS been observed to work in practice (May 2026) but bare names are the documented format — use bare when possible

## Verification

```bash
# Check current fallback chain
hermes fallback list

# Verify config.yaml is valid YAML
python3 -c "import yaml; yaml.safe_load(open('$HOME/.hermes/config.yaml')); print('OK')"

# Test fallback connection (start a session, let primary fail, observe fallback activation)
hermes chat -q "hello" --verbose
```

## Source References

- Fallback chain init: `agent/agent_init.py:882-905`
- Fallback activation + per-entry auth: `agent/chat_completion_helpers.py:883-962`
- Fallback config normalization: `hermes_cli/fallback_config.py`
- Fallback CLI commands: `hermes_cli/fallback_cmd.py`
- Primary restoration: `agent/agent_runtime_helpers.py:869-920`
