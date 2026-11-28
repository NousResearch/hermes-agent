---
name: hermes-multi-profile-provider-setup
description: Add provider to all profiles with glicid failover.
tags: [hermes, provider, config, failover, multi-profile]
category: devops
---

# Configure Multi-Profile Provider with Failover

Use when adding a new LLM provider that must be available in every Hermes profile, with the primary provider (glicid) failing over to the new provider.

## Procedure

1. **Add provider to global config** (`~/.hermes/config.yaml`):
   ```bash
   hermes config set --force providers.<name>.name "<Display Name>"
   hermes config set --force providers.<name>.base_url "https://<endpoint>/v1"
   hermes config set --force providers.<name>.type "openai"
   hermes config set --force providers.<name>.key "<api-key>"  # or .key_env "ENV_VAR" for env-based keys
   ```

2. **Set failover on global config**:
   ```bash
   hermes config set --force fallback_model.provider "<name>"
   hermes config set --force fallback_model.model "default"
   ```

3. **Add provider to each profile** (`--profile <name>`):
   ```bash
   for profile in default portfolio-traderepublic interencheres rtl-sdr; do
     hermes config --profile $profile set --force providers.<name>.name "<Display Name>"
     hermes config --profile $profile set --force providers.<name>.base_url "https://<endpoint>/v1"
     hermes config --profile $profile set --force providers.<name>.type "openai"
     hermes config --profile $profile set --force providers.<name>.key "<api-key>"  # or .key_env "ENV_VAR"
     hermes config --profile $profile set --force fallback_model.provider "<name>"
     hermes config --profile $profile set --force fallback_model.model "default"
   done
   ```

4. **Add model aliases for UI visibility** (`model.aliases.<alias>` = `provider/model-id`):
   ```bash
   for profile in default portfolio-traderepublic interencheres rtl-sdr; do
     hermes config --profile $profile set --force model.aliases.<alias1> "<provider>/<model-id>"
     hermes config --profile $profile set --force model.aliases.<alias2> "<provider>/<model-id>"
     # ... repeat for each model
   done
   ```

5. **For the PRIMARY provider (glicid), also configure model_aliases to reference the provider**:
   ```bash
   hermes config set --force model_aliases.glicid.provider "glicid"
   hermes config set --force model_aliases.glicid.model "Qwen/Qwen3.6-35B-A3B-FP8"
   hermes config set --force model_aliases.glicid.base_url "https://copilot.glicid.fr/api/v1"
   # Repeat for each profile with --profile
   ```

4. **Verify**:
   ```bash
   hermes config --profile <name> show | grep -A5 "providers."
   hermes config --profile <name> show | grep -A2 "fallback_model"
   ```

## Trigger Conditions
- Primary provider: `glicid` (Qwen/Qwen3.6-35B-A3B-FP8 via OpenWebUI at copilot.glicid.fr)
- Failover triggers on: 429 (rate limit), 503 (unavailable), 529 (overload), connection failures
- Failover provider uses its default model (`model: default`)

## Example: ilass Provider
- Name: `ilass`
- URL: `https://llm.ilaas.fr/v1`
- API Key: stored in `key` field (not `key_env`)
- Type: `openai`
- Models: gemma-4-31b, llama-3.1-8b, mistral-medium-latest, mistral-small-4-119b, qwen-3.6-35b-instruct, qwen-3.8-27b

**Before adding model aliases, fetch available models**:
```bash
curl -s https://llm.ilaas.fr/v1/models -H "Authorization: Bearer <api-key>"
```

## Notes
- `hermes config` writes to the active profile's config.yaml; use `--profile` to target others
- Global config at `C:\Users\support\AppData\Local\hermes\config.yaml` applies to `default` profile
- Profile configs live under `profiles/<name>/config.yaml`
- The `fallback_model` key is documented in the config.yaml comments under "Fallback Model"
- Always use `--force` for `providers.*` and `fallback_model.*` keys (not recognized by running version)
- Model aliases must reference a defined provider in `model_aliases.<alias>.provider` or they won't appear in the UI
- The primary provider (glicid) needs BOTH `providers.glicid` AND `model_aliases.glicid.provider = "glicid"` to appear in the model selector — missing either causes it to vanish from the UI
- Use `key_env` for env-based API keys, `key` for hardcoded keys — don't mix them; if switching, clear the other field
- After config changes, restart Hermes to reload providers and model lists
- Hermes masks API keys with `***` in terminal output and `«redacted:sk-...»` in config.yaml — the real value IS stored; verify with `xxd` or Python if unsure
- When updating `.env` via terminal, the masked output `***` is a display filter — the file may already have the correct value; check raw bytes before re-writing
- If a provider disappears from the model selector, check: (1) `providers.<name>` exists in the profile's config, (2) `model_aliases.<name>.provider` matches the provider name exactly, (3) both are in the SAME profile config being used