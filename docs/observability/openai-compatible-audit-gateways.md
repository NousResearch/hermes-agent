# OpenAI-Compatible Audit Gateways

Hermes can route model traffic through an OpenAI-compatible gateway when an
operator needs provider-neutral audit evidence, local runtime governance, or
centralized routing policy outside Hermes itself. This keeps Hermes responsible
for the agent loop while a gateway records independent request receipts.

## When To Use This

Use an audit gateway when you need to answer operational questions such as:

- which agent, user, or service made a provider request
- whether a request stayed local or reached a hosted provider
- which backend and model served the request
- whether a policy gate, fallback, or human-confirmation path applied
- whether the audit log can be verified after the fact

Do not use this pattern to replace Hermes' own planning, tools, skills,
memory, approval UX, or Mixture-of-Agents behavior.

## Gateway Requirements

An audit gateway should expose an OpenAI-compatible `/v1` endpoint and preserve
the request/response shape expected by the configured Hermes provider. For
enterprise evidence, prefer gateways that can:

- write append-only request receipts
- verify receipt integrity with a hash chain or comparable mechanism
- avoid raw prompt and response retention by default
- record hashed actor, team, org, or host-agent identifiers
- export evidence in compliance-friendly formats

ModelRouter is one example of this pattern, but the same guidance applies to
any compatible gateway.

## Identity Headers

If the gateway supports request attribution, Hermes deployments can send stable
identity hints through headers such as:

```text
X-ModelRouter-Actor-ID: service-account-or-user-id
X-ModelRouter-Team-ID: platform
X-ModelRouter-Org-ID: example-org
X-ModelRouter-Host-Agent-ID: hermes-prod
```

Gateways should hash these values before storing them. Do not put raw API keys,
tokens, private prompts, or secrets in identity headers.

## Verification Loop

A typical operator flow is:

1. Start the local or enterprise audit gateway.
2. Configure Hermes' provider base URL to point at the gateway's `/v1`
   endpoint.
3. Send a test prompt through Hermes.
4. Verify the gateway receipt ledger.
5. Export receipts for security or compliance review.

This creates a clear boundary: Hermes remains the agent system of record, while
the gateway becomes the provider-routing evidence layer.
