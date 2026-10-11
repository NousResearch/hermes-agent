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
| copilot | Token | `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` (Copilot device flow — `gh auth login` tokens do NOT work) |
| copilot-acp | External CLI | Copilot CLI on PATH or `COPILOT_CLI_PATH` |
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

Multiple credentials per provider pool and rotate automatically (`hermes auth`).
Fallback chain when the primary fails: `hermes fallback add|remove|list`.

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

### Wrong context window on a custom-provider model

A custom endpoint that omits `context_length` from `/v1/models` cannot be probed, so Hermes
falls through to the hardcoded family catalog and then the generic fallback. A model id whose
version uses a dot (`foo-v4.1-flash`) misses the dashed catalog key (`foo-v4-flash`) and lands on
the low family catch-all — the window then looks wrong in the picker AND the compressor fires
far too early (threshold = pct x window).

Diagnose, don't guess — re-run the real chain against the live config:

```python
from hermes_cli.config import get_compatible_custom_providers, load_config
from agent.model_metadata import get_model_context_length, _longest_key_match, DEFAULT_CONTEXT_LENGTHS
cps = get_compatible_custom_providers(load_config())
get_model_context_length(MODEL, base_url=URL, api_key='x', provider='custom', custom_providers=cps)
_longest_key_match(DEFAULT_CONTEXT_LENGTHS, MODEL.lower())   # which catalog key won
```

Fix at the provider, by route (works on startup, `/model` switch and picker display alike):
`providers.<name>.models` as a mapping with a per-model `context_length` — the shape the setup
wizard writes. Write it in ONE `hermes config set` (the model id's dot makes a per-key dotted
path ambiguous; a mapping value is parsed as YAML):

```bash
hermes config set providers.<name>.models '{<id>: {}, <dotted-id>: {context_length: 1000000}}'
```

Set the value from the vendor's OWN docs, not from recollection: fetch `<docs-host>/llms.txt` for
the page index, then the page as `.md` (Mintlify serves both), which carries the limits table.
A giant-input probe to discover the ceiling empirically may be blocked before it learns anything
(per-request quota pre-consume, free-tier rate limits), so treat it as a last resort and never as
the first check. Deprecated models often keep appearing in `/v1/models` while their window moves
or stops being documented — say which value is doc-stated and which is inferred from the family.

A dict-shaped `models` is user-curated metadata, so live discovery never overwrites it (a plain
list of bare ids can be replaced). Verify with the agent itself — it prints the resolved window:
`AIAgent(model=..., provider=..., base_url=..., api_key=...)` then read
`agent.context_compressor.context_length` / `.threshold_tokens`.
`model_overrides.<provider>.<model_id>.context_window` works too, but it is keyed on the provider
IDENTITY the caller passes, and that is not stable for custom providers (the runtime resolution
can report `provider: custom` while the agent instance reports the configured name). The
route-scoped per-model `context_length` has no such dependency.

Built-in aliases (catalog-resolved against the active provider): `sonnet`,
`opus`, `haiku`, `claude`, `gpt5`, `gpt`, `codex`, `o3`, `o4`, `gemini`,
`deepseek`, `grok`, `llama`, `qwen`, `minimax`, `nemotron`, `kimi`, `glm`,
`step`, `mimo`, `trinity`.
