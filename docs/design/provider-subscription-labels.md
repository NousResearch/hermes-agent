# Provider subscription labels

## Purpose

Hermes model selection surfaces providers that can bill or consume quota from different accounts. The picker must make the billing source visible in the provider label so a user can tell whether selecting a model hits a subscription, API billing account, or local/free endpoint.

This is especially important for users with multiple authenticated providers such as:

- OpenAI Codex OAuth / ChatGPT subscription quota
- OpenAI API pay-as-you-go billing
- Nous Portal subscription
- GitHub Copilot subscription
- Anthropic Claude/API billing
- Local endpoints such as Ollama

## Current labels

The expected user-facing labels are:

| Provider slug | Label | Billing/quota source |
|---|---|---|
| `openai-codex` | `OpenAI Codex [ChatGPT subscription]` | ChatGPT/Codex OAuth quota when available; not OpenAI API billing |
| `openai-api` | `OpenAI API [API billing]` | OpenAI API pay-as-you-go/API credits; not ChatGPT subscription |
| `nous` | `Nous Portal [Nous subscription]` | Nous Portal subscription/account |
| `copilot` | `GitHub Copilot [Copilot subscription]` | GitHub Copilot subscription/account |
| `copilot-acp` | `GitHub Copilot ACP [Copilot subscription]` | GitHub Copilot subscription/account via ACP process |
| `anthropic` | `Anthropic [Claude/API billing]` | Anthropic API key or Claude/Claude Code OAuth credits, depending on credential used |
| `ollama-local` | `Ollama Local [free/local machine]` | Local Ollama endpoint; no cloud subscription billing by Hermes |

## Implementation map

Provider identity has multiple display paths. Keep them in sync:

1. `hermes_cli/models.py`
   - `CANONICAL_PROVIDERS` controls the `hermes model` setup picker labels/descriptions and some fallback picker rows.

2. `hermes_cli/providers.py`
   - `_LABEL_OVERRIDES` controls `get_label()` for built-in providers and label lookup from shared provider metadata.
   - Include aliases that normalize differently. Example: `copilot` normalizes to `github-copilot`, so both keys need the Copilot subscription label.

3. `hermes_cli/model_switch.py`
   - `list_authenticated_providers()` builds `/model` picker rows.
   - For Hermes-mapped built-ins, use `hermes_cli.providers.get_label()` rather than raw models.dev provider names so subscription labels survive the models.dev merge path.

4. `~/.hermes/config.yaml`
   - User-defined endpoints can set `providers.<slug>.name` for local/custom labels.
   - Current local Ollama label is set with:
     `hermes config set providers.ollama-local.name 'Ollama Local [free/local machine]'`

## Verification

From the Hermes repo:

```bash
cd ~/.hermes/hermes-agent
venv/bin/python - <<'PY'
from hermes_cli.models import provider_label
from hermes_cli.providers import get_label
for p in ['openai-codex','openai-api','nous','copilot','copilot-acp','anthropic']:
    print(f'{p}: {provider_label(p)} | {get_label(p)}')
PY
python -m py_compile hermes_cli/models.py hermes_cli/providers.py hermes_cli/model_switch.py
```

To inspect currently visible `/model` provider rows:

```bash
cd ~/.hermes/hermes-agent
venv/bin/python - <<'PY'
from hermes_cli.config import load_config, get_compatible_custom_providers
from hermes_cli.model_switch import list_authenticated_providers
cfg = load_config()
m = cfg.get('model', {})
rows = list_authenticated_providers(
    current_provider=m.get('provider',''),
    current_model=m.get('default',''),
    current_base_url=m.get('base_url',''),
    user_providers=cfg.get('providers',{}),
    custom_providers=get_compatible_custom_providers(cfg),
    max_models=0,
)
for r in rows:
    print(f"{r['slug']}: {r['name']}")
PY
```

## Maintenance rules

- Do not imply ChatGPT Plus/Pro equals OpenAI API credits. Label `openai-codex` as ChatGPT subscription/OAuth and `openai-api` as API billing.
- If adding a new OAuth/subscription provider, add a bracketed billing source to the display label.
- If a provider has both API-key and subscription/OAuth modes, use a conservative label that names both, e.g. `[Claude/API billing]`.
- If labels appear correct in `hermes model` but not `/model`, check whether the row is coming from models.dev and bypassing `get_label()`.
