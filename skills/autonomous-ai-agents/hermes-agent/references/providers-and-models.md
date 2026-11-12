# Providers & Model Aliases

Set via `hermes model` (picker) or `hermes setup`. 35+ provider profiles ship as
plugins under `plugins/model-providers/`; user plugins of the same name override.
Full docs: https://hermes-agent.nousresearch.com/docs/integrations/providers

### Providers

| Provider | Auth | Key env var(s) |
|----------|------|----------------|
| openrouter | API key | `OPENROUTER_API_KEY` |
| anthropic | API key | `ANTHROPIC_API_KEY` (also `CLAUDE_CODE_OAUTH_TOKEN`) |
| nous | OAuth device code | `hermes auth add nous` (or `NOUS_API_KEY`) |
| openai-codex | OAuth | `hermes auth add openai-codex` |
| qwen-oauth | OAuth | `hermes auth add qwen-oauth` |
| minimax-oauth | OAuth | `hermes auth add minimax-oauth` |
| gemini | API key | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| xai | API key | `XAI_API_KEY` (SuperGrok OAuth also supported) |
| deepseek | API key | `DEEPSEEK_API_KEY` |
| zai (GLM) | API key | `GLM_API_KEY` / `ZAI_API_KEY` |
| minimax / minimax-cn | API key | `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` |
| kimi-coding / -cn | API key | `KIMI_API_KEY` / `KIMI_CN_API_KEY` |
| alibaba (+coding-plan) | API key | `DASHSCOPE_API_KEY` / `ALIBABA_CODING_PLAN_API_KEY` |
| xiaomi | API key | `XIAOMI_API_KEY` |
| huggingface | Token | `HF_TOKEN` |
| fireworks / novita / nvidia / deepinfra / gmi / arcee / stepfun / upstage / kilocode / ai-gateway / opencode-zen / opencode-go / ollama-cloud | API key | `<NAME>_API_KEY` |
| bedrock / vertex / azure-foundry | Cloud SDK / key | AWS SDK creds / Vertex ADC / `AZURE_FOUNDRY_API_KEY` |
| custom | Config | `model.base_url` + `model.api_key` in config.yaml |

GitHub Copilot (`copilot`, `copilot-acp`) is a labs plugin, not core: `hermes plugins install github-copilot`, then `hermes auth add copilot` (https://github.com/NousResearch/hermes-labs-plugins/tree/main/github-copilot).

### Resilience layers: pool vs fallback vs auxiliary

Pick the layer by what the backup credential IS, not by what the user calls it:

| Backup is... | Layer | How to set |
|---|---|---|
| Another credential for the SAME provider (non-Anthropic, e.g. two OpenRouter keys) | Credential pool, tried before any fallback | User runs `hermes auth add <provider> --type api-key --label <name>` (prompts securely). Verify with `hermes auth list <provider>` |
| Anthropic API-key credits as backup to a Claude subscription (OAuth) | `fallback_providers` entry with its own `key_env` — NOT the pool | See "Claude sub first, API credits second" below |
| A DIFFERENT provider:model | `fallback_providers` (ordered list) | `hermes config set fallback_providers '[{"provider": "openai-codex", "model": "<id>"}]'` (JSON accepted, non-interactive; `hermes fallback add` is an interactive picker), then `hermes fallback list` |
| Cheap model for side tasks (compression, titles, ...) | `auxiliary.<task>.provider/model` | See "Auxiliary split" below |

- Pool strategy defaults to `fill_first`: priority 0 is used until rate-limited/exhausted, then the next. Reorder with `hermes auth priority <provider> <id|label> <n>`.
- Never accept a key in chat or pass it via `--api-key`; hand the user the command to run. When a key must be moved between stores, do it in code that never prints the value (print only length/prefix check).
- Back up `.env` before editing it (scratch dir, `chmod 600`), tell the user the path because it holds secrets, and offer to delete it once the change is verified.
- Validate a moved key with a direct minimal API call (status 200) and report that separately from whether Hermes actually routes to it.

#### Claude sub first, API credits second (Anthropic pool quirks)

The Anthropic pool cannot order subscription OAuth ahead of an API key:
- `hermes auth add anthropic --type api-key` creates a `manual` entry, and Anthropic always ranks manual entries ahead of seeded ones (`claude_code`, `hermes_pkce`); `hermes auth priority` can't override it, so credits burn first.
- `ANTHROPIC_API_KEY` set in `.env` (with no OAuth env var) is read as "user chose API-key mode": Hermes prunes the `claude_code` OAuth entry entirely, so the sub stops being used.

Working setup: store the key under a name Hermes doesn't auto-read and reference it from a fallback entry (fallback `key_env`/`api_key` overrides pool resolution):
```
# key lives in .env as ANTHROPIC_CREDITS_API_KEY=... ; ANTHROPIC_API_KEY stays empty
hermes config set fallback_providers '[{"provider": "anthropic", "model": "<main model>", "key_env": "ANTHROPIC_CREDITS_API_KEY"}, {"provider": "openai-codex", "model": "<id>"}]'
```
Then verify: `hermes auth list anthropic` shows only `claude_code`; `hermes fallback list`; `hermes doctor`. Removing an `env:ANTHROPIC_API_KEY` pool row with `hermes auth remove` clears the var AND permanently suppresses re-seeding it — mention this to the user.

#### Auxiliary split

Every aux task defaults to `auto` = the main model. List them with `hermes config get auxiliary --json`. Proven split when the main model is a top-tier subscription model:
- small fast model (e.g. Haiku): title_generation, approval, memory_query_rewrite, mcp, tts_audio_tags, group_router, profile_describer, triage_specifier, kanban_decomposer, goal_judge, monitor
- mid model (e.g. Sonnet): compression (a bad summary loses the thread)
- keep on main (`auto`): vision, review, background_review, curator, voice_chat, moa_reference, moa_aggregator

Route them through the user's subscription provider (`provider: anthropic` etc.) rather than OpenRouter so they cost nothing extra and inherit the fallback chain. Set `provider` to the real provider id and run `hermes doctor` after setting.
- With an explicit `model.provider` and NO fallback chain, auxiliary tasks on `auto` are SKIPPED (not rerouted) when the main route fails, and every aux task runs on the main (often expensive) model. On fresh setups recommend both a fallback chain and a cheap aux model for compression/title_generation.
- Discover valid model ids before writing config: `~/.hermes/provider_models_cache.json` lists per-provider models (incl. openai-codex); OpenRouter prices via `curl -s https://openrouter.ai/api/v1/models`.

Full docs: /user-guide/features/fallback-providers (aux config keys, per-task `fallback_chain`).

### User-defined model aliases

Work with `/model <name>` in CLI and every gateway platform. Resolved by
`hermes_cli/model_switch.py::resolve_alias()`; user aliases are checked BEFORE
the built-in table, so a user `sonnet`/`grok` shadows the built-in.

```yaml
# Full form
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  local-qwen:
    model: qwen3.5:397b
    provider: custom
    base_url: "https://ollama.com/v1"
  theta:
    model: theta-1
    provider: custom
    base_url: "https://theta.example.com/v1"
    key_env: THETA_API_KEY        # or: api_key: "${THETA_API_KEY}"

# Short form ("provider/model"), also via CLI:
#   hermes config set model.aliases.fav openrouter/anthropic/claude-sonnet-4.6
model:
  aliases:
    fav: openrouter/anthropic/claude-sonnet-4.6
```

`/model fav` — session-scoped; add `--global` to persist as default.

An alias with its own `base_url` authenticates with its own credential
(`api_key`, which also accepts a `"${VAR}"` reference, or `key_env`). With
neither set the key is resolved from the alias HOST, never carried over from
the provider that was active before the switch.

Built-in aliases (catalog-resolved against the active provider): `sonnet`,
`opus`, `haiku`, `claude`, `gpt5`, `gpt`, `codex`, `o3`, `o4`, `gemini`,
`deepseek`, `grok`, `llama`, `qwen`, `minimax`, `nemotron`, `kimi`, `glm`,
`step`, `mimo`, `trinity`.
