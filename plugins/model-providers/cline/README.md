# Cline provider

Routes Hermes inference through the [Cline API](https://docs.cline.bot/api/overview)
(`https://api.cline.bot/api/v1`), an OpenAI-compatible Chat Completions gateway that
fronts Anthropic, OpenAI, Google and the ClinePass open-model catalog.

## Setup

1. Create an API key at <https://app.cline.bot> → **Settings → API Keys**.
2. Put it where Hermes reads secrets (`.env`), not in `config.yaml`:

   ```dotenv
   CLINE_API_KEY=your_key_here
   ```

3. Select the provider:

   ```bash
   hermes auth add cline          # records the key in the credential pool
   hermes model                   # pick a cline-pass/* or anthropic/* model
   ```

   or set it directly in `config.yaml`:

   ```yaml
   model:
     provider: cline
     default: cline-pass/deepseek-v4-pro
   ```

## Models

- **ClinePass** (`$9.99/mo`, 2-5x usage on open coding models): `cline-pass/deepseek-v4-pro`,
  `cline-pass/qwen3.8-max`, `cline-pass/glm-5.3`, `cline-pass/kimi-k3`, `cline-pass/mimo-v2.5-pro`, …
- **Usage-billed**: `anthropic/claude-sonnet-4-6`, `openai/gpt-4o`, `google/gemini-2.5-pro`, …

The live catalog is fetched from `{base_url}/models` when the key is present; the
`fallback_models` list above is shown otherwise.
