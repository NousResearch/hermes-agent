# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

> **⚠️ WARNING: SDK dependency can silently disappear after updates**
>
> The `langfuse` Python SDK (`pip install langfuse`) is **required** for tracing
> to work. If the SDK is missing from the active Hermes environment, tracing
> silently stops — the plugin fails open and produces no output or errors.
>
> After a Hermes update (`pip install --upgrade hermes-agent`), a `venv` refresh,
> or a reinstall, the `langfuse` SDK may be removed from the environment, causing
> tracing to silently stop. Always reinstall the SDK after such operations:
>
> ```bash
> pip install langfuse
> ```
>
> The plugin logs a one-time warning at startup if the SDK is missing; check
> your logs if you suspect tracing has stopped.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install langfuse
hermes plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open. See the **⚠️ warning above** about the SDK dependency.

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
