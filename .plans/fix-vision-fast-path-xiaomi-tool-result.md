# fix(tools): native vision fast path incorrectly enabled for providers rejecting multimodal tool results

## What

Fixes a hallucination bug where `vision_analyze` silently produced fabricated image
descriptions when used with providers that support vision in *user messages* but
**reject** image content inside *tool-result messages* (e.g. Xiaomi/mimo-v2.5).

Two functions in `tools/vision_tools.py` each had the same class of mistake —
they consulted `profile.supports_vision` but ignored
`profile.supports_vision_tool_messages`:

### Bug 1 — `_supports_media_in_tool_results()`

```python
# Before (wrong): only checks supports_vision
if profile is not None and profile.supports_vision:
    return True

# After (correct): both flags must be True
if profile is not None and profile.supports_vision:
    if getattr(profile, "supports_vision_tool_messages", True):
        return True
```

### Bug 2 — `_should_use_native_vision_fast_path()`

```python
# Before (wrong): models.dev metadata alone enables native path
return (
    _supports_media_in_tool_results(provider, model)
    or _lookup_supports_vision(provider, model, cfg) is True
)

# After (correct): only explicit config override is the escape hatch
tool_result_support = _supports_media_in_tool_results(provider, model)
if tool_result_support:
    return True

# Only fall back to native when the user EXPLICITLY set model.supports_vision
# in config.yaml — do NOT use models.dev metadata alone.
vision_override = _lookup_supports_vision(provider, model, cfg)
if vision_override is True:
    from agent.image_routing import _supports_vision_override
    explicit = _supports_vision_override(cfg, provider, model)
    if explicit is True:
        return True

return False
```

## Why

`ProviderProfile` already has a `supports_vision_tool_messages` field (default `True`)
specifically for this class of provider — the Xiaomi plugin even sets it to `False`
with a comment explaining that their API returns `400 "text is not set"` on
list-type tool content. Neither `_supports_media_in_tool_results` nor
`_should_use_native_vision_fast_path` read that field, making the field
effectively dead code and the protection it was supposed to provide absent.

### Failure chain (before fix)

1. `decide_image_input_mode("xiaomi", "mimo-v2.5", cfg)` → `"native"` ✓  
   (models.dev says `attachment: true`)
2. `_supports_media_in_tool_results("xiaomi", ...)` → **`True`** ✗  
   (reads `profile.supports_vision=True`, ignores `supports_vision_tool_messages=False`)
3. `_should_use_native_vision_fast_path()` → **`True`** ✗
4. `vision_analyze` returns a `_multimodal` envelope with `image_url` content part
5. Xiaomi API rejects / silently drops the image from the tool-result message
6. Model never sees the pixels → hallucinates a description on the next turn

## Behavior matrix

| Provider | Before | After |
|---|---|---|
| Anthropic / OpenAI / OpenRouter | `native` ✓ | `native` (unchanged) ✓ |
| Xiaomi / mimo-v2.5 | `native` ✗ (bug) | aux-LLM fallback ✓ |
| Any future provider with `supports_vision_tool_messages=False` | `native` ✗ | aux-LLM fallback ✓ |
| Custom provider + `model.supports_vision: true` in config.yaml | `native` ✓ | `native` (escape hatch preserved) ✓ |

## Files changed

| File | Change |
|---|---|
| `tools/vision_tools.py` | Fix `_supports_media_in_tool_results` profile branch + rewrite `_should_use_native_vision_fast_path` |
| `tests/tools/test_vision_fast_path.py` | **New** — 10 tests covering the regression, escape hatch, allowlist providers, text-mode short-circuit, exception safety |

## How to reproduce (before fix)

```bash
# Set provider to xiaomi, model to mimo-v2.5 in ~/.hermes/config.yaml:
# model:
#   provider: xiaomi
#   name: mimo-v2.5

hermes chat -q "Describe this image: /path/to/any/image.png"
# → tool returns "Image attached natively for the main model"
# → next turn: model generates a hallucinated description
```

## How to verify (after fix)

```bash
# With the same config, same command:
hermes chat -q "Describe this image: /path/to/any/image.png"
# → tool routes through auxiliary vision LLM instead of native fast path
# → model receives an actual text description of the image content

# Run the unit tests:
pytest tests/tools/test_vision_fast_path.py tests/run_agent/test_vision_tool_messages.py -v
# → 27 passed
```

## Platforms tested

- Windows 11 (development machine, Python 3.11.13)
- All existing vision-related tests pass: `test_vision_tool_messages.py` (17 tests) + new `test_vision_fast_path.py` (10 tests)

## Related

- Provider profile field `supports_vision_tool_messages` was added in the same
  commit that added the Xiaomi profile — this fix activates the protection
  the field was always intended to provide.
- The `run_agent._provider_supports_vision_tool_messages()` / 
  `_tool_result_content_for_active_model()` path (tested in
  `test_vision_tool_messages.py`) already correctly uses `supports_vision_tool_messages`
  to downgrade *received* multimodal tool results. This fix closes the
  complementary gap: preventing the fast path from *producing* them in the first
  place.
