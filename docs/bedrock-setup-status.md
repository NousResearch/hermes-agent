# Bedrock / Mantle Setup Status

## Current State

Hermes is installed locally and the `claudio-lab` profile works with both the Bedrock Mantle path used by Codex and native AWS SDK Bedrock.

- Hermes version: `0.17.0`.
- Profile path: `~/.hermes/profiles/claudio-lab`.
- Profile `.env` is private and not committed.
- Working provider: `bedrock-mantle`.
- Working base URL: `https://bedrock-mantle.us-east-1.api.aws/openai/v1`.
- Working model: `openai.gpt-5.5`.
- Working wire mode: `codex_responses` / Responses API.
- Native SDK provider: `bedrock`.
- Native SDK smoke model: `us.amazon.nova-pro-v1:0`.
- Native SDK IAM user: `claudio-hermes-bedrock-invoke`.
- Last verified locally: both smoke tests passed after restoring the profile config.

Smoke test passed:

```bash
claudio-lab chat \
  --provider bedrock-mantle \
  --model openai.gpt-5.5 \
  --query "Reply with exactly: Hermes Mantle OK" \
  --quiet
```

Result: `Hermes Mantle OK`.

Native Bedrock SDK smoke test passed:

```bash
claudio-lab chat \
  --provider bedrock \
  --model us.amazon.nova-pro-v1:0 \
  --query "Reply with exactly: SDK Bedrock OK" \
  --quiet
```

Result: `SDK Bedrock OK`.

## Why This Shape

Codex is configured to use Bedrock Mantle as an OpenAI-compatible Responses endpoint, not the native Bedrock Converse provider:

```toml
model = "openai.gpt-5.5"
model_provider = "bedrock"

[model_providers.bedrock]
base_url = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
wire_api = "responses"
```

Hermes needs the equivalent as a named custom provider with `api_mode: codex_responses`.

## Hermes Profile Config Shape

Do not commit the real token. The profile config shape is:

```yaml
model:
  provider: bedrock-mantle
  default: openai.gpt-5.5
providers:
  bedrock-mantle:
    name: AWS Bedrock Mantle
    base_url: https://bedrock-mantle.us-east-1.api.aws/openai/v1
    key_env: BEDROCK_MANTLE_API_KEY
    api_mode: codex_responses
    default_model: openai.gpt-5.5
agent:
  approval_mode: smart
```

The private profile `.env` contains:

```bash
BEDROCK_MANTLE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
```

## Keeping Both Providers Working

Use two separate auth paths in the same Hermes profile:

- `bedrock-mantle` is a named OpenAI-compatible provider for `openai.gpt-5.5` through the Mantle Responses endpoint. It uses only `BEDROCK_MANTLE_API_KEY`.
- `bedrock` is Hermes' native AWS SDK provider for Bedrock Converse models. It uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and the AWS region variables.

Do not try to make the native `bedrock` provider call `openai.gpt-5.5` unless AWS exposes that model as a Bedrock Converse `modelId` for this account and region. Until then, use `bedrock-mantle` for GPT-5.5 and `bedrock` for SDK-native Bedrock models.

Useful checks:

```bash
claudio-lab chat \
  --provider bedrock-mantle \
  --model openai.gpt-5.5 \
  --query "Reply with exactly: Hermes Mantle OK" \
  --quiet

claudio-lab chat \
  --provider bedrock \
  --model us.amazon.nova-pro-v1:0 \
  --query "Reply with exactly: SDK Bedrock OK" \
  --quiet
```

## Notes

- Native Hermes `provider: bedrock` uses the scoped `claudio-hermes-bedrock-invoke` IAM user. Keep this separate from the Mantle token.
- Plain `provider: custom` was not enough because Hermes conservatively ignores `codex_responses` for generic custom endpoints. A named provider preserves the Responses wire mode.
- The older `AWS_BEDROCK_USAGE_*` keys are usage-reader credentials and should not be used for inference.
- Do not set `AWS_BEARER_TOKEN_BEDROCK` in the Hermes profile `.env`; it can shadow standard AWS SDK credentials.
