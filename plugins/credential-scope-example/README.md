# credential-scope-example

A minimal example consumer of the
[credential-scope contract](../../website/docs/developer-guide/credential-scope.md).
It aggregates API attempt outcomes per `(provider_id, credential_scope_id)`
from the `pre_api_request` / `post_api_request` / `api_request_error` hooks.

It is deliberately boring:

- no credential access, no network calls, no config changes;
- in-memory counters only (a real quota/health consumer would persist them);
- `unknown` / `unavailable` scopes are counted provider-wide, never guessed
  into an account bucket;
- if the `llm.credential_scope` capability is not granted, the host strips
  `credential_scope` from this plugin's payloads and the plugin stays idle.

To try it: declare the capability, grant it through the normal consent flow,
and watch the `credential-scope-example` debug logs during a session.
