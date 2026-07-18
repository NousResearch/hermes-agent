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

## Trusted-edge correlation

Correlation metadata is disabled by default. To emit a pseudonymous identifier
on Langfuse root traces, enable the generic observability gate in
`~/.hermes/config.yaml`:

```yaml
observability:
  correlation:
    scheme: hmac-sha256-v1:k1
    emitter_metadata_enabled: true
```

Store the secret only in `~/.hermes/.env`:

```bash
HERMES_OBSERVABILITY_CORRELATION_HMAC_K1=base64:<strict-unpadded-base64url>
```

The decoded key must be 32–64 bytes. The payload accepts only `A-Z`, `a-z`,
`0-9`, `_`, and `-`; padding, whitespace, `+`, and `/` are rejected. Restart
Hermes after changing the gate or secret because the plugin loads this
configuration once when it registers.

When enabled with a valid key, root metadata gains
`correlation_scheme: hmac-sha256-v1:k1` and a lowercase 64-hex
`correlation_id`. Missing or invalid configuration emits neither field and
leaves all existing Langfuse tracing behavior unchanged.

### Privacy

The HMAC key is never sent to Langfuse or written to `config.yaml`, and the
plugin does not log the key, task identifier, or derived digest. The digest is
a pseudonymous join key, not encryption; access to it should be limited to the
trusted observability systems that need correlation. Existing `task_id`
metadata remains in Langfuse for dashboard compatibility.

## Disable

```bash
hermes plugins disable observability/langfuse
```
