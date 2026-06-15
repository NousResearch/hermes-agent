# API Key Hotswap Feature Report

## Summary

Implemented a mid-session API key hotswap mechanism for Hermes Agent. Users can now switch API keys (and therefore providers) on the fly via the new `/apikey` slash command, without needing to start a new session with `/new` or `/reset`.

## Motivation

Hermes already supported switching models and providers mid-session via `/model`, but there was no clean way to swap the underlying API key for the current provider. When a key hits a rate limit, quota cap, or runs out of credits, the only recovery path was to edit `~/.hermes/.env` and restart the session. This feature closes that gap.

## Design

The feature is built on top of the existing, battle-tested `switch_model()` runtime path, which already handles safe client rebuilding, rollback on failure, context-compressor updates, and prompt-cache invalidation.

A new thin wrapper, `switch_api_key()`, keeps the current model and provider but replaces the API key, reusing all of `switch_model()`'s safety machinery.

The `/apikey` slash command is available in both the CLI and the messaging gateway.

## Usage

```text
/apikey                  show current provider, model, and masked key
/apikey <new-key>        hotswap key for the current provider
/apikey --save <new-key> hotswap and persist the key to ~/.hermes/.env
/apikey --reload         reload .env and rebuild the client
```

Alias: `/key`

## Typical "Out of Credits" Scenario

1. You are in a long session and the provider returns a 429, 402, or auth/quota error.
2. Obtain a fresh API key from your provider dashboard (a second account, a new key, or a backup provider key).
3. In the active Hermes session, type:

   ```text
   /apikey sk-your-fresh-key
   ```

   Hermes will rebuild the live client with the new key and keep the conversation context intact.

4. If you want the new key to survive the current session, use:

   ```text
   /apikey --save sk-your-fresh-key
   ```

   This writes the key to `~/.hermes/.env` under the correct env var for the current provider and reloads the environment.

5. If you edited `.env` manually instead, run:

   ```text
   /apikey --reload
   ```

   to pick up the change without leaving the session.

6. Send your next prompt; it will use the new credentials.

### Switching provider at the same time

If the new key belongs to a different provider, use `/model` first:

```text
/model <model-name> --provider <provider>
```

Then `/apikey <new-key>` if you need to supply a specific key for that provider. In most cases `/model` will resolve the provider's default key from `.env` automatically.

## Implementation Details

### New / modified files

| File | Change |
|------|--------|
| `agent/agent_runtime_helpers.py` | Added `switch_api_key()` helper that reuses `switch_model()` while keeping model/provider stable. |
| `run_agent.py` | Added `AIAgent.switch_api_key()` forwarder. |
| `hermes_cli/apikey_switch.py` | New module: argument parsing, provider-to-env-var resolution, `.env` persistence, status formatting, and the shared `apply_api_key_switch()` orchestrator. |
| `hermes_cli/commands.py` | Registered `/apikey` (alias `/key`) in `COMMAND_REGISTRY`. |
| `cli.py` | Added `_handle_apikey_command()` for the interactive CLI. |
| `gateway/run.py` | Added dispatch for `canonical == "apikey"`. |
| `gateway/slash_commands.py` | Added async `_handle_apikey_command()` for the messaging gateway, including session-override persistence and cache eviction. |

### Key behaviors

- **In-place client rebuild**: the live `AIAgent` gets a new OpenAI/Anthropic client without discarding conversation history.
- **Atomic rollback**: if the client rebuild fails, `switch_model()`'s existing snapshot/rollback restores the previous state.
- **Env-var awareness**: the helper resolves the correct `.env` key for the provider (e.g. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, or custom `key_env` values from user-defined providers).
- **Secure persistence**: `--save` writes through the existing `save_env_value()` helper, which enforces the env-var denylist and owner-only file permissions.
- **Gateway session isolation**: per-session overrides in `_session_model_overrides` are updated, and the cached agent is evicted so the next turn builds with the new credentials.

## Testing

New tests:

- `tests/run_agent/test_switch_api_key.py` — runtime helper behavior and rollback safety.
- `tests/hermes_cli/test_apikey_switch.py` — argument parsing, env-var resolution, masking, formatting, and `apply_api_key_switch()` outcomes.
- `tests/gateway/test_apikey_command.py` — gateway handler status display, live hotswap, and `--save` persistence.

All new tests pass, and existing `switch_model` tests continue to pass.

## Verification

```bash
cd ~/.hermes/hermes-agent
venv/bin/python -m pytest \
  tests/run_agent/test_switch_api_key.py \
  tests/hermes_cli/test_apikey_switch.py \
  tests/gateway/test_apikey_command.py \
  tests/run_agent/test_switch_model_rollback.py \
  tests/run_agent/test_switch_model_fallback_prune.py \
  tests/run_agent/test_switch_model_context.py \
  -v --tb=short -o 'addopts='
# 25 passed
```

A broader run of `hermes_cli` + `gateway` tests also passes (132 tests).

## Future Improvements

- Add `--provider <slug>` support to `/apikey` for one-step provider+key swaps.
- Add gateway locale strings (`gateway.apikey.*`) for translated responses.
- Surface a validation probe (lightweight `/v1/models` ping) before confirming the swap.
