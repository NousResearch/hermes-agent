# Provider/Fallback Guard

This note covers the gateway case where platform commands work but natural
language agent turns fail before a provider returns a normal HTTP response.

## Meaning

- The messaging gateway can be online while the agent turn path is unavailable.
- Status commands should still work when provider validation fails.
- Natural-language turns fail closed with
  `provider_config_or_fallback_failure` instead of posting raw Python
  exceptions or request dumps.

## Misconfiguration Signals

- Provider name, model, endpoint/base URL, or auth material is missing.
- `fallback_model` or `fallback_providers` is present but null or malformed.
- Fallback entries resolve to the same provider, model, and base URL as the
  failed primary without an explicit same-provider fallback flag.
- `openai-codex` pointed at the `chatgpt.com` Codex backend is a fragile
  runtime binding. Prefer a stable OpenAI Responses-compatible configuration
  when the operator has approved credentials and model access.

## Safe Inspection

Use these commands for local diagnosis. Do not paste secrets or dump contents.

```bash
hermes gateway status
systemctl --user status hermes-gateway --no-pager -l
journalctl --user -u hermes-gateway -n 100 --no-pager
```

Request debug dumps, when enabled, are local diagnostics. Logs may reference
the dump path, but Discord-visible output must never include dump JSON, token
values, cookies, authorization headers, API keys, or raw Discord IDs.

## 2026-05 Recovery Note

The observed Discord failure was not caused by Discord routing. The configured
provider, model, and host were unchanged from the pre-update local snapshot:
`openai-codex`, `gpt-5.5`, and `chatgpt.com`. Structural auth was present and
the Codex model catalog returned `gpt-5.5`.

The live failure reproduced in the OpenAI SDK streaming helper for the Codex
backend: `responses.stream(...)` raised a local `TypeError` before a normal HTTP
response was available. The same endpoint and token completed successfully via
`responses.create(stream=True)`. Hermes now routes that SDK-local stream helper
failure through the existing `create(stream=True)` fallback and backfills Codex
terminal events whose `response.output` is `None` from streamed output items or
text deltas.

Recovery verification should use a provider-backed prompt that is not handled by
the Discord local command fast path. A successful local smoke looks like:

```bash
cd /home/hragj/.hermes/hermes-agent
venv/bin/hermes -z 'Reply exactly: HERMES_AGENT_PROVIDER_SMOKE_OK'
```

The expected output is exactly `HERMES_AGENT_PROVIDER_SMOKE_OK`. If that command
returns the controlled `provider_config_or_fallback_failure` diagnostic instead,
inspect redacted gateway logs and auth status; do not paste tokens, cookies,
authorization headers, request dumps, or raw environment values.

## Patch Preservation

The local guard fix is preserved before any Hermes update or provider recovery
work:

- Guard commit:
  `4b11b6977a7b4fec10db3e2d4873e712d91f61ec`
- Patch artifact:
  `/home/hragj/.hermes/patches/hermes-provider-fallback-guard-4b11b697.patch`
- Patch SHA-256:
  `8be6f8cce3741ccdb3b94d3b557363472c4ceedf3a5b4b199e0d9b1b0a48157d`
- Bundle artifact:
  `/home/hragj/.hermes/patches/hermes-provider-fallback-guard-4b11b697.bundle`
- Bundle SHA-256:
  `c867dee6478cc6b7f14ef8938498a743ed65582ffe7bb934132a20229a07f449`

Do not run `hermes update` until this patch is preserved and rollback is
understood. If a future update removes the guard, restore it from the patch on
a clean branch:

```bash
cd /home/hragj/.hermes/hermes-agent
git switch -c fix/restore-provider-fallback-guard
git am /home/hragj/.hermes/patches/hermes-provider-fallback-guard-4b11b697.patch
```

If the local commit object is needed for forensic comparison, inspect the
bundle without applying it:

```bash
git bundle verify /home/hragj/.hermes/patches/hermes-provider-fallback-guard-4b11b697.bundle
git log --oneline /home/hragj/.hermes/patches/hermes-provider-fallback-guard-4b11b697.bundle
```

## Safe Behavior

Natural-language process-control requests such as `hermes gateway restart`
are advisory only. Use a governed launcher or service command outside chat for
gateway process mutation.
