# WhatsApp adaptive routing

The opt-in `gateway.whatsapp_adaptive_routing` lane gives authenticated
WhatsApp text turns a bounded fast path while preserving the normal Hermes
agent path for work that needs tools or more context:

- `DIRECT` uses one native Gemini `generateContent` call on the stable
  `gemini-3.1-flash-lite` model. The request contains only a short router
  instruction and the current message; it has no Hermes tools, tool search,
  raw session history, or tool results. The structured JSON response contains
  the direct answer, so no second Gemini answer call is needed.
- `AGENTIC` hands the original user message to the existing Hermes agent
  pipeline. Provider, model, credentials, channel overrides, profile scope,
  session overrides, tools, and other runtime behavior are resolved by Hermes
  exactly as for a normal turn. The adaptive feature does not define an
  agentic provider or model.

Activation is intentionally separate from this code change:

```yaml
gateway:
  whatsapp_adaptive_routing:
    enabled: true
```

The fast model is discovered through Gemini `ListModels` and must advertise
`generateContent`; the stable Flash-Lite model is accepted only when its
structured-output capability is known. The operator references are the
[Gemini model catalog](https://ai.google.dev/gemini-api/docs/models) and the
[structured output guide](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).

A Gemini `429 RESOURCE_EXHAUSTED` or equivalent fast-lane quota failure is
handled as one bounded AGENTIC handoff. The fast request is not retried by
the adaptive router, the same turn never re-enters the fast lane, and the
normal Hermes provider fallback chain remains active for the AGENTIC handoff.
Because the adaptive decision is consumed once at the gateway boundary, a
normal fallback that happens to use Gemini is still an AGENTIC runtime call,
not a second adaptive-router invocation. Other fast-lane failures fail closed
to the normal agentic lane.

If the normal Hermes runtime is configured to use the same provider/model as
the fast lane, the adaptive router still runs only once and does not dispatch
back into FAST. The ordinary agent path remains bounded by its existing error
handling; no second adaptive fallback is created.

Rollback is configuration-only: set `enabled: false` or remove the
`whatsapp_adaptive_routing` block. This change does not deploy, restart the
gateway, pair WhatsApp, or edit live configuration.
