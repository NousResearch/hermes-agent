# providers/

Single source of truth for every inference provider Hermes knows about.

Each provider is declared once here as a `ProviderProfile`. Every other layer —
auth resolution, transport kwargs, model listing, runtime routing — reads from
these profiles instead of maintaining its own parallel data.

---

## Directory layout

```
providers/
├── base.py           ProviderProfile dataclass + OMIT_TEMPERATURE sentinel
├── __init__.py       Registry: register_provider(), get_provider_profile()
├── README.md         This file
│
├── # Simple providers — just identity + auth + endpoint
├── alibaba.py        Alibaba Cloud DashScope
├── arcee.py          Arcee AI
├── bedrock.py        AWS Bedrock  (api_mode=bedrock_converse)
├── deepseek.py       DeepSeek
├── huggingface.py    Hugging Face Inference API
├── kilocode.py       Kilo Code
├── minimax.py        MiniMax (international + CN)
├── nvidia.py         NVIDIA NIM  (default_max_tokens=16384)
├── ollama_cloud.py   Ollama Cloud
├── stepfun.py        StepFun
├── xiaomi.py         Xiaomi MiMo
├── xai.py            xAI Grok  (api_mode=codex_responses)
├── zai.py            Z.AI / GLM
│
├── # Medium — one or two quirks
├── anthropic.py      Native Anthropic  (x-api-key header, api_mode=anthropic_messages)
├── copilot.py        GitHub Copilot  (auth_type=copilot, reasoning per model)
├── copilot_acp.py    Copilot ACP subprocess  (api_mode=copilot_acp)
├── custom.py         Custom/Ollama local  (think=false, num_ctx)
├── gemini.py         Google Gemini AI Studio + Cloud Code OAuth
├── kimi.py           Kimi Coding  (OMIT_TEMPERATURE, thinking, dual endpoint)
├── openai_codex.py   OpenAI Codex OAuth  (api_mode=codex_responses)
├── opencode.py       OpenCode Zen + Go  (per-model api_mode routing)
│
├── # Complex — subclasses with multiple overrides
├── nous.py           Nous Portal  (tags, attribution, reasoning omit-when-disabled)
├── openrouter.py     OpenRouter  (provider preferences, public model fetch)
├── qwen.py           Qwen OAuth  (message normalization, cache_control, vl_hires)
└── vercel.py         Vercel AI Gateway  (attribution headers, reasoning passthrough)
```

---

## ProviderProfile fields

```python
@dataclass
class ProviderProfile:
    # Identity
    name: str                    # canonical ID — auto-registered as PROVIDER_REGISTRY key for new api-key providers
    api_mode: str                # "chat_completions" | "anthropic_messages" |
                                 # "codex_responses" | "bedrock_converse" | "copilot_acp"
    aliases: tuple               # alternate names resolved by get_provider_profile()

    # Auth & endpoints
    env_vars: tuple              # env var names holding the API key, in priority order
    base_url: str                # default inference endpoint
    models_url: str              # explicit models endpoint; falls back to {base_url}/models
                                 # set when the models catalog lives at a different URL
                                 # (e.g. OpenRouter: public /api/v1/models vs /api/v1 inference)
    auth_type: str               # "api_key" | "oauth_device_code" | "oauth_external" |
                                 # "copilot" | "aws" | "external_process"

    # Client-level quirks
    default_headers: dict        # extra HTTP headers sent on every request

    # Request-level quirks
    fixed_temperature: Any       # None = use caller's default; OMIT_TEMPERATURE = don't send
    default_max_tokens: int|None # inject max_tokens when caller omits it
    default_aux_model: str       # cheap model for auxiliary tasks (compression, vision, etc.)
                                 # empty string = use main model (default)
```

---

## Hooks (override in a subclass)

| Method | When to override |
|--------|-----------------|
| `prepare_messages(messages)` | Provider needs message pre-processing (Qwen: string → list-of-parts, cache_control) |
| `build_extra_body(*, session_id, **ctx)` | Provider-specific `extra_body` fields (Nous: tags, OpenRouter: provider preferences) |
| `build_api_kwargs_extras(*, reasoning_config, **ctx)` | Returns `(extra_body_additions, top_level_kwargs)` — use when some fields go to `extra_body` and some go top-level (Kimi: `reasoning_effort` top-level; OpenRouter: `reasoning` in extra_body) |
| `fetch_models(*, api_key, timeout)` | Custom model listing (Anthropic: x-api-key header; OpenRouter: public endpoint, no auth; Bedrock/copilot-acp: return None) |

All hooks have safe defaults — only override what differs from the base.

---

## How to add a new provider

### 1. Simple (standard OpenAI-compatible endpoint)

```python
# providers/myprovider.py
from providers import register_provider
from providers.base import ProviderProfile

myprovider = ProviderProfile(
    name="myprovider",           # must match id in hermes_cli/auth.py PROVIDER_REGISTRY
    aliases=("my-provider", "myp"),
    api_mode="chat_completions",
    env_vars=("MYPROVIDER_API_KEY",),
    base_url="https://api.myprovider.com/v1",
    auth_type="api_key",
)

register_provider(myprovider)
```

The default `fetch_models()` will call `GET https://api.myprovider.com/v1/models`
with Bearer auth automatically. No override needed for standard `/v1/models`.

### 2. With quirks (subclass)

```python
# providers/myprovider.py
from typing import Any
from providers import register_provider
from providers.base import ProviderProfile


class MyProviderProfile(ProviderProfile):
    """My provider — custom reasoning header."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if reasoning_config:
            extra_body["my_reasoning"] = reasoning_config.get("effort", "medium")
        return extra_body, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        # Override only if your endpoint differs from standard /v1/models
        return super().fetch_models(api_key=api_key, timeout=timeout)


myprovider = MyProviderProfile(
    name="myprovider",
    aliases=("myp",),
    env_vars=("MYPROVIDER_API_KEY",),
    base_url="https://api.myprovider.com/v1",
)

register_provider(myprovider)
```

### 3. Wire it up

After creating the file, add `name` to the `_PROFILE_ACTIVE_PROVIDERS` set in
`run_agent.py` once you've verified parity against the legacy flag path. Start
with a simple provider (no message prep, no reasoning quirks) and work up.

---

## fetch_models contract

```python
def fetch_models(
    self,
    *,
    api_key: str | None = None,
    timeout: float = 8.0,
) -> list[str] | None:
    ...
```

- Returns `list[str]`: model IDs from the provider's live endpoint.
- Returns `None`: provider doesn't support REST model listing (Bedrock, copilot-acp),
  or the request failed. Callers **must** fall back to `_PROVIDER_MODELS` on `None`.
- Never raises — swallow exceptions and return `None`.
- Default implementation: `GET {base_url}/models` with Bearer auth. Works for any
  standard OpenAI-compatible provider.

**Override when:**
- Auth header is not `Bearer` (Anthropic: `x-api-key`)
- Endpoint path differs from `/models` AND you can't just set `models_url` (OpenRouter: public endpoint, pass `api_key=None` explicitly)
- Response format differs (extra wrapping, non-standard `id` field)
- Provider has no REST endpoint (Bedrock, copilot-acp → return `None`)
- Filtering needed post-fetch (only tool-capable models, etc.)

Use `models_url` instead of overriding when the only difference is the URL:

```python
# No subclass needed — just set models_url
myprovider = ProviderProfile(
    name="myprovider",
    base_url="https://api.myprovider.com/v1",
    models_url="https://catalog.myprovider.com/models",  # different host
)
```

---

## Debugging

### Check if a provider resolves

```python
from providers import get_provider_profile

p = get_provider_profile("myprovider")
print(p)           # ProviderProfile(name='myprovider', ...)
print(p.base_url)
print(p.api_mode)
```

### Check all registered providers

```python
from providers import _REGISTRY
print(list(_REGISTRY.keys()))
```

### Test live model fetch

```python
import os
from providers import get_provider_profile

p = get_provider_profile("myprovider")
key = os.getenv("MYPROVIDER_API_KEY")
models = p.fetch_models(api_key=key, timeout=5.0)
print(models)      # list of model IDs, or None on failure
```

### Test alias resolution

```python
from providers import get_provider_profile

# All of these should return the same profile
assert get_provider_profile("openrouter").name == "openrouter"
assert get_provider_profile("or").name == "openrouter"
```

### Run the provider test suite

```bash
# From the repo root
source venv/bin/activate
python -m pytest tests/providers/ -v
```

### Check ruff + ty compliance

```bash
source venv/bin/activate
ruff format providers/*.py
ruff check providers/*.py --select UP,E,F,I,W
ty check providers/*.py
```

---

## Common mistakes

**Wrong `name`** — must be the same string that appears as the key in
`hermes_cli/auth.py` `PROVIDER_REGISTRY`. New api-key providers auto-register
into `PROVIDER_REGISTRY` from the profile, so the name IS the key. For providers
with a pre-existing `PROVIDER_REGISTRY` entry, use the exact `id` field value.

**Wrong `env_vars`** — separate API-key vars from base-URL override vars in the
tuple. Env vars that end with `_BASE_URL` or `_URL` are treated as URL overrides;
everything else is treated as an API key. Getting this wrong causes the doctor
health check to send a URL string as a Bearer token.

**Wrong `base_url`** — several providers have non-obvious paths:
`stepfun: /step_plan/v1`, `opencode-go: /zen/go/v1`. The profile's `base_url`
is also used as the `inference_base_url` when auto-registering into `PROVIDER_REGISTRY`
for new providers, so it must be correct for auth resolution to work.

**Skipping `api_mode`** — defaults to `chat_completions`. Providers that use
`anthropic_messages`, `codex_responses`, `bedrock_converse`, or `copilot_acp`
must set it explicitly.

**Forgetting `register_provider()`** — auto-discovery runs `pkgutil.iter_modules`
over the package and imports each module, but only if `register_provider()` is
called at module level. Without it the profile is never in `_REGISTRY`.

**`fetch_models` returning the wrong shape** — must return `list[str]` (plain
model IDs), not `list[tuple]` or `list[dict]`. Callers expect plain strings.

**Wrong `build_api_kwargs_extras` return shape** — must return a 2-tuple
`(extra_body_dict, top_level_dict)`. Returning a single dict causes a
`ValueError: not enough values to unpack` in the transport.

**`build_api_kwargs_extras` wrong tuple** — must return `(extra_body_dict,
top_level_dict)`. Returning a flat dict or swapping the order silently sends
fields to the wrong place.
