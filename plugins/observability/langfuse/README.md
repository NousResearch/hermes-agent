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

Generation observations include the Hermes system prompt when the provider
uses a separate `system` param (Anthropic Messages API). Open an **LLM call**
child span to inspect `role: system` (truncated via `HERMES_LANGFUSE_MAX_CHARS`).

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # capture content for 50% of turns
HERMES_LANGFUSE_PSEUDONYM_KEY=...    # dedicated HMAC key for exported IDs
HERMES_LANGFUSE_PSEUDONYM_KEY_VERSION=v1  # rotation/version label
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_CAPTURE=sanitized    # content capture mode (see below)
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Capture modes

`HERMES_LANGFUSE_CAPTURE` controls how much *content* (prompts, responses,
tool arguments/results) is exported. Structural metadata — IDs, roles, tool
names, token usage, cost, timing — is always captured in every mode.

| mode | behavior |
|------|----------|
| `metadata` | No content. Each content field is replaced by a shape/size stub (`{"omitted": true, "type": "text", "chars": N}`). |
| `sanitized` | **(default)** Content is exported after secret-pattern redaction (API keys, tokens, JWTs, private keys, `password=`-style assignments) and truncation. Redaction runs *before* truncation. |
| `full` | Content without the plugin's pattern sanitization, truncated and still subject to the mandatory SDK export mask. Explicit opt-in. |

The active mode is recorded on every trace as `metadata.capture_mode`.
Sampling applies only to content. Every turn can still export operational
metadata, including failures, token/cost usage, counts, statuses, and retry
outcomes, when its content is not sampled. A non-sampled turn is an absolute
content boundary: prompts, responses, tool arguments/results, merged tool-call
records, MoA advisor outputs, subagent goals/summaries, and subagent tool
history payloads are replaced by content-free shape/size stubs. This keeps
failure and feedback correlation visible without forcing content capture.

Note: `sanitized` is pattern-based defense in depth, not a DLP guarantee.
For personal sessions or shared Langfuse projects, prefer `metadata`.

## Telemetry contract and privacy

Root traces carry `telemetry_schema_version = "hermes.telemetry.v1"` plus the
Hermes release when available. The v1 metadata contract includes stable
configuration, prompt, and tool-policy fingerprints; enabled-tool count and
serialized tool-schema bytes when tools are present; exact context character
counts; provider-reported token-source buckets; and route/retry/fallback/quota
outcomes when their lifecycle payloads provide those values. Missing values
are omitted rather than estimated.

Session/task/turn/request and platform-native identifiers are never exported raw. Set a
dedicated `HERMES_LANGFUSE_PSEUDONYM_KEY` of at least 16 UTF-8 bytes to export
keyed, domain-separated HMAC pseudonyms, and set
`HERMES_LANGFUSE_PSEUDONYM_KEY_VERSION` to a rotation label understood by your
downstream consumers. When the key is absent or too short, these identifier
fields are omitted; Hermes does not fall back to unsalted hashes.
The key itself is never attached to trace metadata.

All capture modes use the strongest masking boundary supported by the installed
Langfuse SDK. Newer SDKs use the OpenTelemetry export-stage
`mask_otel_spans` callback, which masks final span attributes including those
from third-party instrumentation. If masking fails, the callback returns the
SDK's explicit `drop=True` result instead of exporting the original content.
SDKs without that API use their supported synchronous `mask` callback; failures
there return a fixed redacted sentinel rather than the original value.
Observability failures remain non-blocking for agent serving.

### Trace-ID migration

Trace IDs created by this version include immutable `turn_id` identity even
when `task_id` is present. Consecutive turns in one gateway session therefore
appear as separate backend traces. Historical traces created by older Hermes
versions may already contain multiple turns merged under one trace ID; this
change does not split or rewrite those records. Dashboards and exports that
assumed one trace per session/task must use `session_id` grouping (when keyed
pseudonyms are enabled) and treat the deployment boundary as a migration cut.

## Error + shutdown coverage

- Failed model requests (`api_request_error` hook) close their generation
  with `level=ERROR`, status code, retry counters, and a capture-mode-scrubbed
  error message. Non-retryable failures also finish the turn trace.
- Session end/finalize closes any still-open traces for that session and
  flushes queued events, so interrupted or tool-only turns don't dangle.

## Disable

```bash
hermes plugins disable observability/langfuse
```
