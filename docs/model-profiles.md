# Model Routing Profiles

This feature lets users route different work types to different models/providers.

Instead of one global model for everything, users can configure context-specific routing profiles.

Built-in starter categories:
- chat
- coding
- planning
- research
- delegation

Users may also add custom categories such as `ops`, `fast`, `cheap`, `vision`, or `review`.

## What was added

- New config section: `model_profiles`
- New delegation option: `model_profile` (tool argument)
- New delegation default: `delegation.model_profile` (config)
- Automatic profile inference for delegated tasks uses configurable ordered routing rules when present.
- Built-in toolset heuristics remain as a fallback only.
- `chat` is intended for direct conversation surfaces; CLI/cron should normally inherit the primary model unless explicitly routed.
- Legacy `delegation.model` / `delegation.provider` behavior remains and takes precedence

## Routing configuration (no manual file edits)

Use:

```bash
hermes configure-model-routing
```

This command is separate from `hermes model`.

- `hermes model` remains the primary/default model selector.
- `hermes configure-model-routing` configures context-specific routing profiles.

The routing flow uses the same Hermes-style interactive picker logic:

```bash
hermes configure-model-routing              # interactive setup (auto-backup)
hermes configure-model-routing --reset      # reset to defaults (auto-backup)
hermes configure-model-routing --restore    # roll back routing sections from backup
hermes configure-model-routing --restore-full  # roll back entire config from backup
hermes configure-model-routing --list-backups  # show available backups
```

Provider suggestions in that flow are filtered to providers that currently resolve with working credentials.

Every config-modifying operation auto-creates a timestamped backup in `~/.hermes/config-backups/` before writing. Restores can target just model routing sections (leaving other config untouched) or the full config. Even restore creates a `pre_restore` backup, so undo is itself undoable. Auto-rotates at 10 backups.
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

If omitted, profile is inferred from routing rules first, then fallback toolset heuristics.

## Configurable routing rules

Optional ordered rules live under `model_routing.rules`:

```yaml
model_routing:
  rules:
    - if_toolsets_any: [terminal, file]
      profile: coding
    - if_toolsets_any: [web, browser]
      profile: research
    - if_goal_matches: [deploy, infra, migration]
      profile: ops
```

First matching rule wins.

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
4. inferred profile from ordered routing rules
5. fallback heuristic from toolsets

## Backward compatibility

Existing configs continue to work.

- If users do nothing, `hermes model` still behaves as the default-model selector.
- Routing remains opt-in and can be reset independently.

## Validation checklist for end users

- `hermes` starts with expected chat model/provider
- `delegate_task` coding-style tasks hit coding profile
- `delegate_task` web-style tasks hit research profile
- direct chat surfaces use the expected chat profile model
- delegated tasks use either explicit profile, routing rules, or fallback inference

## Troubleshooting

- If a profile has `provider` but no credentials, delegation will return a clear error.
- If a profile has only `model`, Hermes keeps parent credentials and only changes model.
- If local endpoint auth fails, set `api_key_env` and verify the env var is visible to the process.
