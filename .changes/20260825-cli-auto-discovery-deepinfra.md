## Reviewer feedback addressed (Enough1122 AI review, PR #86612)

Added missing DeepInfra provider to the PROVIDER_REGISTRY in `hermes_cli/auth.py` to match the title/description which mentioned four providers (Mistral, Cohere, DeepInfra, SiliconFlow).

- Added `deepinfra` entry with:
  - `id`: "deepinfra"
  - `name`: "DeepInfra"
  - `auth_type`: "api_key"
  - `inference_base_url`: "https://api.deepinfra.com/v1/openai"
  - `api_key_env_vars`: ("DEEPINFRA_API_KEY",)
  - `base_url_env_var`: "DEEPINFRA_BASE_URL"
