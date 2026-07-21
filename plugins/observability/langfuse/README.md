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

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

### Credential validation

When the plugin starts up with credentials set, two checks run before any
traces are queued so a broken config can never silently sit in production
(issue #29332):

1. **Prefix guard** — `HERMES_LANGFUSE_PUBLIC_KEY` must start with `pk-lf-`
   and `HERMES_LANGFUSE_SECRET_KEY` must start with `sk-lf-`. Anything else
   (`placeholder`, `your-langfuse-key`, `xxx`, etc.) trips a single
   warning naming the env var and the plugin short-circuits without
   queueing traces.
2. **Server-side `auth_check()`** — for credentials that pass the prefix
   guard but are otherwise wrong (typo, revoked key, wrong project, cloud
   vs. self-hosted mismatch) the plugin probes the Langfuse server in a
   background thread immediately after construction. An explicit rejection
   logs a single warning naming both env vars + the base URL, and every
   subsequent hook becomes inert. Transient errors and the documented
   `langfuse-python <=3.0.4` `AttributeError` quirk
   ([langfuse/langfuse#7456](https://github.com/langfuse/langfuse/issues/7456))
   are treated as inconclusive — observability is not disabled.

Set `HERMES_LANGFUSE_SKIP_AUTH_CHECK=1` to bypass the server probe in
offline / restricted-network setups where it would always-fail.

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
HERMES_LANGFUSE_SKIP_AUTH_CHECK=1    # bypass the startup auth_check probe
                                     # (use for offline / restricted-network
                                     # installs; see "Credential validation")
```

## Disable

```bash
hermes plugins disable observability/langfuse
```
