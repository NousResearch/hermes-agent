## Summary

Fixes #59660: `CustomProfile` was unconditionally forwarding `reasoning_effort` to non-reasoning models on cross-provider fallback, causing HTTP 400 "does not support thinking" errors when the fallback target is a local Ollama plain model (e.g. `llama3.1-8b-64k`).

## The bug

In `plugins/model-providers/custom/__init__.py`, the `CustomProfile.build_api_kwargs_extras` function:

1. Received the model name via `**ctx` but never used it
2. Unconditionally set `top_level["reasoning_effort"]` whenever the user had any non-empty `reasoning_config["effort"]`

This is fine when the model is a known reasoning model (GLM-5.2 on Volcengine ARK, Claude 4.5+, o1/o3, etc.) — the field is accepted. But on cross-provider fallback to a plain local model (e.g. `llama3.1-8b-64k`), the field is not just ignored — it actively 400s the request. The user sees "does not support thinking" with no obvious way back.

Why this was silent in normal use: the user's primary/interactive model is usually a cloud reasoning-capable model (Claude, GPT, etc.) via OpenRouter or Nous — different provider profile, no crash. Only surfaces on cross-provider fallback to a plain model.

## The fix

Add a model-aware allowlist check (mirroring `ZaiProfile._model_supports_thinking`):

- New helper `_model_supports_reasoning_effort(model)` returns True for known reasoning model families
- `build_api_kwargs_extras` now takes an explicit `model` parameter and gates the `reasoning_effort` emission on it
- Models NOT on the allowlist produce an empty `top_level` dict — graceful degradation, the endpoint's server default applies instead of 400
- The `think=False` disable path is preserved for all models (it's the explicit "turn off thinking" hint, a no-op for non-reasoning models but valid)

## The allowlist

```python
_REASONING_MODEL_TOKENS = (
    "glm-4.5", "glm-4.6", "glm-5",     # GLM 4.5+ and 5.x
    "o1", "o3", "o4",                   # OpenAI reasoning
    "qwq", "qwen3",                     # Qwen QwQ and Qwen3 hybrid
    "deepseek-r",                       # DeepSeek-R
    "claude-sonnet-4.5", "claude-opus-4.5",  # Claude 4.5+
    "magistral",                        # Mistral reasoning
)
```

Substring match (not exact) so vendor prefixes (`z-ai/glm-5.2`), alias spellings (`glm-5-2`, `glm-5p2`), and the canonical name all hit. Tokens are deliberately conservative — every name on this list is a verified reasoning-capable model family. A miss is benign (omits the field rather than 400ing); a false positive would 400.

Plain models (llama, mistral, qwen2.5, gemma, phi-3, etc.) are excluded by design.

## Files changed

- `plugins/model-providers/custom/__init__.py` — new `_model_supports_reasoning_effort` helper, explicit `model` parameter on `build_api_kwargs_extras`, gating logic
- `tests/plugins/model_providers/test_custom_profile.py` — new `TestCustomReasoningModelAware` class with 9 new tests (6 parameterized non-reasoning models + 3 safety tests)

## Failing-test-first verification

`tests/plugins/model_providers/test_custom_profile.py` — 22 tests total:
- 7 new tests in `TestCustomReasoningModelAware` **FAIL on unfixed code**, PASS after fix:
  - `test_non_reasoning_model_omits_reasoning_effort[llama3.1-8b-64k]` (the canonical repro)
  - 5 other parameterized non-reasoning models
  - `test_unknown_model_defaults_to_omitting_reasoning_effort`
- 15 pre-existing tests still pass:
  - 13 original tests in `TestCustomReasoningWireShape` + `TestCustomReasoningWithNumCtx` (including the `qwen3` case that was added to the allowlist to match the existing test's intent)
  - 2 new safety tests in `TestCustomReasoningModelAware` (known reasoning model still passes through, non-reasoning model disable still sends `think=False`)

285/285 broader `tests/plugins/model_providers/` + `tests/providers/` tests still pass. No regressions.

## Trade-offs

- **Conservative allowlist**: only known reasoning-capable models are listed. New reasoning models need to be added explicitly. This is intentional — a false positive 400s, a miss omits silently. Better to be safe.
- **No way to override the allowlist**: users with a non-listed reasoning model can't easily opt in. The fallback (`reasoning_effort` omitted) is graceful — the user gets a response without the effort hint, no crash. A future PR could add a config option if this becomes a real pain point.
- **`qwen3` inclusion is opinionated**: plain `qwen3` is a hybrid reasoning model, but the name is also used as a family prefix for non-reasoning variants. The existing `test_num_ctx_with_effort` test (which used `model="qwen3"` and expected `reasoning_effort` to be passed) set the precedent. I kept the existing behavior by adding `qwen3` to the allowlist.

Refs: #59660