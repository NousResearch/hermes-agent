# Patch bundle: OpenRouter free guard + provider hard-fail plan

Status: prepared for later continuation.

This file records the intended patch set and key implementation notes so work can resume later.

## Goal

Prevent accidental paid OpenRouter usage by:
- blocking `google/*` and `nvidia/*` requests when no direct provider API key exists
- allowing only curated / explicit free OpenRouter models when `HERMES_SWARM_FORCE_FREE_OPENROUTER=true`
- keeping cooldown DB logic unchanged

## Current repo state already observed

Some partial changes already exist locally in `gateway/platforms/api_server.py`:
- `_openrouter_model_free_cached()` exists
- `google` / `nvidia` branches in `_runtime_kwargs_for_model_id()` already partially return `provider="blocked"`
- `_swarm_model_is_available()` already blocks raw `google/` and `nvidia/` without keys
- `hermes-swarm.env` already prefers explicit `openrouter/*:free` models

These changes still need to be completed and normalized.

---

## Files to finish

### 1) `gateway/platforms/api_server.py`

#### A. Add / keep curated whitelist for reliable free OpenRouter models
Use a whitelist in addition to `:free` suffix because `openrouter/free` routing can be non-deterministic.

Suggested whitelist:
- `openrouter/google/gemma-4-26b-a4b-it:free`
- `openrouter/google/gemma-4-31b-it:free`
- `openrouter/nvidia/nemotron-nano-9b-v2:free`
- `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- `openrouter/qwen/qwen3-coder:free`
- `openrouter/meta-llama/llama-3.3-70b-instruct:free`
- `openrouter/z-ai/glm-4.5-air:free`

#### B. `_openrouter_model_free_cached(model_id)`
Desired behavior:
- return `True` if model is in whitelist
- return `True` if model string contains `:free`
- else optionally query OpenRouter `/models` pricing endpoint
- only return `True` if prompt and completion prices are both `0`
- cache results for ~5 minutes

#### C. `_runtime_kwargs_for_model_id(model)`
Desired behavior:
- for `google/*`:
  - if direct Google key exists -> provider `google`
  - else -> provider `blocked` (NOT OpenRouter fallback)
- for `nvidia/*`:
  - if direct NVIDIA key exists -> provider `nvidia`
  - else -> provider `blocked`
- for `openrouter/*`:
  - keep provider `openrouter`
  - later gate through whitelist / free check

#### D. `_swarm_model_is_available(model)`
Desired behavior:
- if runtime provider resolves to `blocked` -> return `False`
- if runtime provider resolves to `openrouter` and `HERMES_SWARM_FORCE_FREE_OPENROUTER=true`:
  - require `_openrouter_model_free_cached(raw)` to be `True`
  - otherwise return `False`
- preserve cooldown DB check exactly as-is after provider gating

#### E. Complexity-aware routing cleanup
Current code still prefers `google/gemini-2.5-flash` in some heuristic branches.
Replace those medium/cheap hints with deterministic non-paid or explicitly allowed providers.
Suggested replacements:
- balanced: `github-copilot/gpt-5-mini`, `openai/gpt-5-mini`, `opencode-zen/gpt-5-nano`
- cheap: `opencode-zen/gpt-5-nano`, `opencode-zen/ling-2.6-flash-free`, `ollama/qwen3-coder-next`

---

### 2) `agent/model_metadata.py`

Add helper:

```python
def is_model_free(model_id: str) -> bool:
    # True for explicit curated whitelist and :free suffix
```

This helper can stay simple and mirror the whitelist.

---

### 3) `tests/gateway/test_api_server.py`

Add tests for:
- `google/gemini-2.5-flash` blocked when no Google key exists
- `nvidia/...` blocked when no NVIDIA key exists
- curated free OpenRouter model allowed when force-free is enabled
- non-free OpenRouter model blocked when force-free is enabled

Also update any tests that assume:
- role aliases resolve to a fixed concrete backend
- Gemini is a normal balanced fallback

---

### 4) `/Users/tusker/dev/opencode/hermes-swarm.env`

Keep / confirm:
- `HERMES_SWARM_FORCE_FREE_OPENROUTER=true`
- fallback list only includes explicit curated `openrouter/*:free` entries
- no plain `google/gemini-*` or `nvidia/*` fallbacks unless direct provider keys are guaranteed

Current env already looks mostly aligned.

---

## Recommended exact local patch direction

### In `api_server.py`, add something like:

```python
_OPENROUTER_FREE_MODEL_WHITELIST = {
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-nano-9b-v2:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/qwen/qwen3-coder:free",
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/z-ai/glm-4.5-air:free",
}
```

and then in `_openrouter_model_free_cached()`:

```python
if raw in _OPENROUTER_FREE_MODEL_WHITELIST:
    return True
if ":free" in raw.lower():
    return True
```

### In `_runtime_kwargs_for_model_id()`:

For `google` and `nvidia`, remove silent OpenRouter fallback.

Desired output with no key:

```python
runtime_kwargs["base_url"] = ""
runtime_kwargs["api_key"] = ""
runtime_kwargs["provider"] = "blocked"
```

### In `_swarm_model_is_available()`:

Add / confirm:

```python
runtime_kwargs, model_name = _runtime_kwargs_for_model_id(raw)
provider = str(runtime_kwargs.get("provider") or "").strip().lower()
if provider == "blocked":
    return False
if provider == "openrouter" and force_free:
    if not _openrouter_model_free_cached(raw):
        return False
```

---

## Important note for later deploy

Per `AGENTS.md`, do NOT hot-patch containers.
Deployment workflow later should be:
1. edit locally
2. sync repo to `/srv/opencode/hermes-agent/`
3. `docker build -t hermes-agent:latest .`
4. `docker compose down && docker compose up -d`
5. verify `http://127.0.0.1:8642/health`
6. verify public ingress if needed

---

## Suggested commit message later

```text
guard: block paid fallback routing and whitelist free OpenRouter models
```

Optional longer body:
- block google/nvidia requests when no direct provider key exists
- enforce curated free OpenRouter whitelist under force-free mode
- remove non-deterministic OpenRouter paid fallback behavior
- add tests for blocked provider and free-model gating

---

## Suggested sanity tests later

- request `google/gemini-2.5-flash` with no Google API key -> must fail closed
- request `nvidia/...` with no NVIDIA API key -> must fail closed
- request `openrouter/google/gemma-4-26b-a4b-it:free` -> allowed
- request a non-free OpenRouter model with force-free enabled -> blocked
- confirm cooldown DB still scopes by provider/model/base_url

---

## Session note

User explicitly stated:
- `openrouter/free` router may be non-deterministic
- prefer specifically curated `:free` tagged models
- patch locally first; deploy later
- write patches to a file so continuation is easy later

This file is that handoff artifact.
