# Hermes Agent Configuration Guide

## Per-Model Reasoning Effort Overrides

You can configure different reasoning effort levels for different models. This allows you to set `high` effort for complex reasoning models like `claude-opus-4.5` while keeping `medium` for faster models like `gemini-flash`.

### Configuration

Edit your `config.yaml` (typically at `~/.hermes/config.yaml`):

```yaml
agent:
  reasoning_overrides:
    claude-opus-4.5: high
    gemini-flash: medium
    gpt-4.5: high
```

### Key Matching

The model name matching is **spelling-tolerant**. All of these variations will match:
- `claude-opus-4.5`, `claude-opus-4-5`, `claude-opus.4.5`
- `anthropic/claude-opus-4.5`, `openrouter/anthropic/claude-opus-4.5`
- With or without provider prefixes

Exact matches take precedence over variants.

### Resolution Order

When determining reasoning effort for a model, Hermes checks in this order:

1. **Session override**: `/reasoning high` (current session only)
2. **Per-model override**: `agent.reasoning_overrides.<model>` from config.yaml
3. **Global default**: `agent.reasoning_effort` from config.yaml

### How It Works

The override applies automatically in these scenarios:

- **CLI startup**: Uses the override for the configured default model
- **Gateway messaging**: Each gateway session uses the override for its model
- **Desktop/TUI**: Uses the override for the configured model
- **Model switching**: When you switch models, the reasoning effort updates to the new model's override
- **Fallback activation**: When the primary model fails and Hermes falls back to a secondary model, it uses that fallback model's override
- **Reasoning recovery**: When the primary model recovers after a fallback, the original model's override is restored

### Examples

#### Example 1: High effort for Opus, medium for others
```yaml
agent:
  reasoning_overrides:
    claude-opus-4.5: high
```

#### Example 2: Different efforts per model
```yaml
agent:
  reasoning_overrides:
    claude-opus-4.5: high
    gemini-2.0-flash: low
    gpt-4.5: high
    o3-mini: medium
```

#### Example 3: With provider prefixes
```yaml
agent:
  reasoning_overrides:
    anthropic/claude-opus-4.5: high
    google/gemini-2.0-flash: low
```

All of these are equivalent — the provider prefix is optional.

### Disabling Reasoning for Specific Models

Set the override to `none` to disable reasoning for a specific model:

```yaml
agent:
  reasoning_overrides:
    gemini-flash: none
```

### Troubleshooting

**Override not taking effect?**
- Check the exact model name in your config with `/model`
- Verify the override is under `agent.reasoning_overrides` (not `agent.reasoning_effort`)
- Restart the gateway or CLI session after editing config.yaml
- Check logs for parsing errors

**Override applies but reasoning doesn't work?**
- Not all models support reasoning (e.g., `gemini-flash` has limited support)
- Check the model's documentation for reasoning capability
- Use a model that explicitly supports extended thinking

**Session override not respecting per-model override?**
- Session overrides take precedence (by design)
- Clear the session override with `/reasoning default` to return to the per-model override
