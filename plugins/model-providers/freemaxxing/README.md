# Freemaxxing — proven-free local failover for Hermes

Freemaxxing exposes one stable Hermes route:

```text
provider = freemaxxing
model    = freemaxxing
```

Behind that identity is an authenticated loopback router. It uses existing
Hermes credentials, selects only routes whose free status is positively known,
and fails over without rewriting the configured provider or model.

## Cost contract

Freemaxxing does **not** treat “an API token exists” as proof that a route is
free.

| Tier | Backend | Admission rule |
|---|---|---|
| 0 | Nous Portal | Only the explicitly allow-listed free DeepSeek Flash route is admitted |
| 1 | OpenRouter | Only catalog IDs ending in `:free` are admitted |

Hugging Face is deliberately **not** auto-enrolled. Hugging Face routed
inference can consume included credits and then become pay-as-you-go; silently
adding every model returned by `/v1/models` would violate the no-paid-fallback
contract. Unknown or ambiguous pricing fails closed.

## Profile isolation

Freemaxxing is currently supported only in classic/single-profile runtimes.
When `gateway.multiplex_profiles` is enabled, the router rejects initialization
before reading Nous, OpenRouter, or fallback credentials. This is intentional:
a process-global loopback pool cannot safely represent multiple profile secret
scopes after an HTTP/thread boundary.

A future multiplex implementation needs a profile-addressed capability token
mapped to a per-profile pool. Until that wire exists, native providers remain
the correct choice for multiplexed gateways.

## Local listener security

The listener binds only to `127.0.0.1` and requires a random, per-process bearer
on every operational endpoint. The token is generated at startup, placed in the
in-process provider credential path, and never written to disk.

- `GET /v1/healthz` is a non-sensitive liveness marker.
- `GET /healthz`, `GET /v1/models`, and `POST /v1/chat/completions` require the
  bearer.
- A listener cannot be silently downgraded to unauthenticated mode or reused
  with a different token.

Provider discovery opens only this authenticated listener. Upstream credentials,
catalogs, and pool state are constructed lazily after the first authenticated
runtime request.

## Failure behavior

- `429`: bounded `Retry-After` cooldown, then fail over.
- `5xx`, timeout, reset, DNS failure: short cooldown, then fail over.
- Model-not-found: skip the backend without poisoning general health.
- `401/403`: one serialized credential refresh where supported, then skip.
- Other `4xx`: return a client error without replaying a malformed request.
- HTTP 200 with truncated, oversized, non-UTF-8, malformed, or non-object JSON:
  classify as transient and fail over before committing a response.
- Streaming: fail over only before response commit; bounded SSE lines/events;
  downstream client cancellation does not cool down the upstream backend.
- Exhaustion: OpenAI-shaped `503` with the last failure class.

## Setup

Select **Freemaxxing** in `hermes model`. Zero additional setup is needed when a
working Nous Portal login already exists. Optionally add an OpenRouter key to
make its `:free` catalog the second tier:

```bash
hermes secret set OPENROUTER_API_KEY sk-or-...
```

The local port defaults to `11435` and can be changed with the deployment-level
`FREEMAXXING_PORT` environment variable.

## Limits

- Text/tool-calling only; `supports_vision=False`.
- Single-profile runtime only; multiplexed gateways fail closed.
- No latency or quality weighting within a tier; eligible peers round-robin.
- The free-route allow-list is intentionally conservative. A newly launched
  route is not admitted until its free status can be proved in code or catalog
  metadata.
