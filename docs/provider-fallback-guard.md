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

## Safe Behavior

Natural-language process-control requests such as `hermes gateway restart`
are advisory only. Use a governed launcher or service command outside chat for
gateway process mutation.
