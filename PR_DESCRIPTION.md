# Fix: Prevent credential pool poisoning from model-not-found 401s and api_mode persistence bug

## Summary

This PR fixes two related bugs that cause the credential pool to become permanently exhausted, preventing ALL subsequent API calls on the affected provider — even with valid API keys.

## Root Cause Analysis

### Bug 1: Model-not-found 401s poison the credential pool

**Symptom**: When a user selects a model that doesn't exist on the target provider (e.g., `mimo-v2-pro` on opencode-go), the 401 response is classified as `FailoverReason.auth`, which marks the credential pool entry as exhausted. This exhaustion is permanent — the pool entry never recovers, even though the API key is valid.

**Root cause**: The error classifier (`agent/error_classifier.py`) treats ALL 401 responses as auth failures. However, opencode-go returns 401 for model-not-found errors (e.g., "Not supported model mimo-v2-pro"). The correct classification should be `FailoverReason.model_not_found`, which skips pool exhaustion.

**Impact**: A single model-not-found 401 exhausts the credential pool for the entire provider. All subsequent model switches on that provider fail with "no available entries (all exhausted or empty)".

**Fix**: Check model-not-found patterns BEFORE defaulting to auth for 401 responses. Also add "not supported model" to the pattern list (opencode-go's exact phrasing).

### Bug 2: api_mode lost during session override rehydration

**Symptom**: When switching from one provider to another (e.g., ollama-cloud → opencode-go), models that use the Anthropic Messages wire protocol (minimax-m3, qwen3.7-plus) fail with 401 "Invalid API key" because the OpenAI SDK is used instead of the Anthropic SDK.

**Root cause**: The session override system persists `model`, `provider`, and `base_url` to disk, but NOT `api_mode`. During rehydration (every turn), `api_mode` is re-resolved from the provider's transport config, which returns `chat_completions` (the default). For OpenCode providers that serve both `chat_completions` and `anthropic_messages` models, this means the wrong SDK is used for `anthropic_messages` models.

**Impact**: After any gateway restart, models like minimax-m3 and qwen3.7-plus on opencode-go fail because:
1. `api_mode` defaults to `chat_completions` (wrong)
2. OpenAI SDK is used instead of Anthropic SDK
3. Request hits `/chat/completions` instead of `/v1/messages`
4. Returns 401 "Invalid API key" (wrong endpoint)

**Fix**: Add `api_mode` to `PERSISTABLE_MODEL_OVERRIDE_KEYS` and resolve it per-model during rehydration for OpenCode providers.

## Changes

### 1. Error classifier (`agent/error_classifier.py`)
- Add "not supported model" to `_MODEL_NOT_FOUND_PATTERNS`
- Check model-not-found patterns for 401 BEFORE defaulting to auth

### 2. Session override persistence (`gateway/session.py`)
- Add `api_mode` to `PERSISTABLE_MODEL_OVERRIDE_KEYS`

### 3. Session override rehydration (`gateway/run.py`)
- Use persisted `api_mode` when available
- For legacy sessions without `api_mode`, resolve per-model using `opencode_model_api_mode()`

### 4. Model catalog (`hermes_cli/models.py`, `hermes_cli/setup.py`)
- Remove `mimo-v2-pro` and `mimo-v2-omni` from opencode-go model list (these models are not served by opencode-go per official docs)

### 5. Tests (`tests/agent/test_error_classifier.py`)
- Add 3 test cases for 401 model-not-found classification

## Investigation Notes

### What we tried

1. **Credential pool reset script** — Reset exhausted entries in auth.json
   - Result: Works temporarily, but pool gets re-poisoned immediately by the next model-not-found 401

2. **Base_url normalization fix** — Ensure `/v1` suffix is correct for Anthropic SDK
   - Result: Base_url was already correct; the issue was wrong SDK being used (OpenAI instead of Anthropic)

3. **Session override api_mode persistence** — Persist api_mode in session overrides
   - Result: Helps NEW sessions but doesn't fix legacy sessions without persisted api_mode

4. **Per-model api_mode resolution** — Resolve api_mode from model name during rehydration
   - Result: Fixes the SDK selection issue for OpenCode providers

### What we discovered but couldn't fully fix

**Provider resolution on cross-provider switches**: When switching from opencode-go → ollama-cloud → back to a model that exists on opencode-go, the `model_switch` resolves the model to the native provider (e.g., MiniMax) instead of opencode-go. This is because:
1. The session override's `current_provider` is `ollama-cloud` (from the previous switch)
2. `model_switch` uses `current_provider` as a hint but the model's direct alias overrides it
3. The model resolves to its native provider instead of the user's configured provider

**Workaround**: Always use `--provider opencode-go` when switching to opencode-go models:
```
/model minimax-m3 --provider opencode-go
```

**Full fix needed**: The slash command should detect when a model exists on the previous provider and prefer that provider over the model's native provider. This requires changes to `model_switch.py`'s provider resolution logic.

## Testing

1. Start with opencode-go/mimo-v2.5 (works)
2. Switch to ollama-cloud (triggers 429/403)
3. Switch to minimax-m3 on opencode-go
4. Verify: should use Anthropic SDK, not OpenAI SDK
5. Verify: credential pool should NOT be exhausted

## Related Issues

- Credential pool exhaustion during desktop model switching
- OpenCode Go minimax-m3/qwen3.7-plus 401 errors after provider switch
- Session override api_mode not persisted across gateway restarts
