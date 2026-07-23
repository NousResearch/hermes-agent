# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install langfuse
hermes plugins enable observability/langfuse
```

## Required credentials and consent

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK, credentials, or explicit export consent, the plugin does not
register export hooks and does not initialize a Langfuse client.

Consent belongs in `config.yaml`, not `.env`:

```yaml
observability:
  langfuse:
    export: metadata   # none, metadata, or content
```

## Exported fields

`metadata` consent exports trace/session ids, task id, turn id, API request id,
platform, provider, model, API mode, base URL, message/tool counts,
approximate input tokens, request character count, response character count,
tool names, tool call ids, duration, finish reason, usage details, and cost
details.

`content` consent exports everything in `metadata`, plus the last user message,
serialized request messages, assistant content/reasoning/tool calls, tool
arguments, tool results, and final trace output.

`none` is the default and exports nothing.

## Verify

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # then check Langfuse for a "Hermes turn" trace
```

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
hermes plugins disable observability/langfuse
```
