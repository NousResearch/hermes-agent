# Model Profiles (chat / planning / coding / research)

This feature lets users route different work types to different models/providers.

Instead of one global model for everything, users can configure:
- chat model (main conversation)
- coding model (terminal/file-heavy delegated tasks)
- planning model (specs/roadmaps)
- research model (web/browser-heavy tasks)

## What was added

- New config section: `model_profiles`
- New delegation option: `model_profile` (tool argument)
- New delegation default: `delegation.model_profile` (config)
- Automatic profile inference for delegated tasks:
  - `terminal` or `file` toolsets => `coding`
  - `web` or `browser` toolsets => `research`
  - otherwise => `planning`
- `chat` profile is applied to CLI, gateway temporary agents, and cron jobs
- Legacy `delegation.model` / `delegation.provider` behavior remains and takes precedence

## CLI onboarding (no manual file edits)

Use:

```bash
hermes model
```

After provider/model setup, Hermes now asks:
- "Configure model profiles for chat/coding/planning/research now?"
- then walks each profile interactively

Provider suggestions in that flow are filtered to providers that currently resolve with working credentials.
Model suggestions are fetched from each provider's live `/models` endpoint when available.

## Config format

Equivalent config written by the wizard goes to `~/.hermes/config.yaml`:

```yaml
model:
  default: anthropic/claude-sonnet-4
  provider: openrouter

model_profiles:
  chat:
    model: anthropic/claude-sonnet-4
    provider: openrouter

  coding:
    model: openai/gpt-5-codex
    provider: openrouter

  planning:
    model: google/gemini-3-flash-preview
    provider: openrouter

  research:
    model: perplexity/sonar-deep-research
    provider: openrouter

  # Optional catch-all profile if you explicitly choose model_profile=delegation
  delegation:
    model: google/gemini-3-flash-preview
    provider: openrouter

# Optional default profile for delegate_task when not specified
delegation:
  model_profile: planning
```

Each profile supports these fields:

- `model`: model slug
- `provider`: provider name (`openrouter`, `openai-codex`, `nous`, `zai`, `kimi-coding`, `minimax`, etc.)
- `base_url`: optional custom OpenAI-compatible endpoint
- `api_key_env`: environment variable name to read API key from
- `api_key`: optional inline key (not recommended)

## Local model usage (Ollama / vLLM / LM Studio)

Use a profile with custom `base_url` and an env var key:

```yaml
model_profiles:
  coding:
    model: qwen2.5-coder:32b
    provider: openrouter
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
```

Then set the env var:

```bash
export OLLAMA_API_KEY=dummy
```

Notes:
- Many local OpenAI-compatible servers accept any non-empty key.
- `api_key_env` is preferred over hardcoding `api_key` in config.

## delegate_task usage

You can force a profile per call:

```json
{
  "goal": "Create an implementation plan for migration",
  "model_profile": "planning"
}
```

Or per item in batch mode:

```json
{
  "tasks": [
    {"goal": "Refactor parser", "toolsets": ["terminal", "file"], "model_profile": "coding"},
    {"goal": "Research alternatives", "toolsets": ["web"], "model_profile": "research"}
  ]
}
```

If omitted, profile is inferred from toolsets (or defaults to planning).

## Precedence rules

### Main chat model
1. CLI arg `--model`
2. `model_profiles.chat.model`
3. `model.default`
4. built-in fallback

### Main chat provider/base URL/key
1. explicit CLI/provider args
2. `model_profiles.chat` provider/base/key
3. existing provider resolution (`HERMES_INFERENCE_PROVIDER`, `model.provider`, env/auth store)

### delegate_task routing
1. legacy `delegation.model` / `delegation.provider` (if set) => override everything
2. explicit `model_profile` on tool call / task item
3. `delegation.model_profile`
4. inferred profile from toolsets

## Backward compatibility

Existing configs continue to work.

If users do nothing, Hermes behavior remains effectively unchanged except that profile-based routing becomes available.

## Validation checklist for end users

- `hermes` starts with expected chat model/provider
- `delegate_task` coding-style tasks hit coding profile
- `delegate_task` web-style tasks hit research profile
- cron jobs use chat profile model
- gateway temporary agents (e.g. memory flush paths) use chat profile model

## Troubleshooting

- If a profile has `provider` but no credentials, delegation will return a clear error.
- If a profile has only `model`, Hermes keeps parent credentials and only changes model.
- If local endpoint auth fails, set `api_key_env` and verify the env var is visible to the process.
