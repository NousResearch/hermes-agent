# MiniMax OAuth auxiliary routing

## Symptom

When `minimax-oauth` is selected for an auxiliary task or a MoA reference,
`resolve_provider_client()` returns `(None, None)` and the request may fall
through to another provider or surface a misleading missing-API-key error.
The main MiniMax OAuth conversation path remains functional.

## Root cause

`hermes_cli.auth.PROVIDER_REGISTRY` registers `minimax-oauth` with
`auth_type="oauth_minimax"`, while the generic OAuth dispatch in
`agent/auxiliary_client.py` handles only `oauth_device_code` and
`oauth_external`. The auxiliary resolver therefore never reaches the working
MiniMax OAuth runtime credential path.

## Fix

Add a first-class `minimax-oauth` branch beside the other named OAuth
providers. It calls
`resolve_minimax_oauth_runtime_credentials(as_token_provider=True)` and uses:

- the callable token provider, so every request can observe persisted refreshes;
- the runtime-resolved base URL, including a persisted region-specific endpoint;
- `build_anthropic_client()` and `AnthropicAuxiliaryClient`, matching MiniMax's
  Anthropic Messages transport;
- `is_oauth=False`, because that wrapper flag means native Anthropic/Claude
  Code wire transforms, not generic bearer authentication. The callable token
  provider supplies the MiniMax OAuth bearer independently.

The branch intentionally runs before the generic `ProviderConfig` dispatch.
No environment API key or synthetic `MINIMAX-OAUTH_API_KEY` is introduced.

## Regression coverage

`TestResolveMiniMaxOAuthForAux` verifies that:

1. a callable token provider and a persisted regional URL reach the Anthropic
   client unchanged;
2. the returned wrapper preserves the requested model without enabling Claude
   Code identity/tool-name transforms;
3. the actual request-builder call receives `is_oauth=False`;
4. a missing login returns `(None, None)` rather than falling through to a
   fabricated API-key path.

Run:

```bash
scripts/run_tests.sh tests/agent/test_auxiliary_client.py tests/test_minimax_oauth.py tests/run_agent/test_moa_loop_mode.py -q
```
