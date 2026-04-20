# Richard-Only Service Principal Policy

Date: 2026-04-19

This note defines the principal-gating model for new external productivity
services exposed to Hermes, specifically Zoom, Grain, and Granola for
`rj@stratminds.vc`.

## Goal

Hermes should be able to use Richard's service access without teaching the
container or MCP subprocesses raw upstream secrets and without exposing those
services to any non-Richard sender identity.

## Canonical principal

The canonical human principal is:

- `rj@stratminds.vc`

Transport-level aliases such as Telegram sender IDs are treated as aliases of
that canonical principal. The important local alias currently observed by Hermes
is:

- `telegram:1643294159`

## Policy model

A service is usable only when:

1. the active sender resolves to the canonical Richard principal
2. the requested service is explicitly allowed for that principal
3. the service bridge or MCP wrapper uses host-side auth and never expects the
   raw upstream token inside the Hermes container

The policy file format is a JSON object at:

- `~/.hermes/service_principals.json`

Shape:

```json
{
  "version": 1,
  "default_deny": true,
  "principals": {
    "rj@stratminds.vc": {
      "aliases": [
        "rj@stratminds.vc",
        "telegram:1643294159",
        "1643294159"
      ],
      "services": {
        "zoom": {
          "allow": true,
          "account": "rj@stratminds.vc"
        },
        "grain": {
          "allow": true,
          "account": "rj@stratminds.vc"
        },
        "granola": {
          "allow": true,
          "account": "rj@stratminds.vc"
        }
      }
    }
  }
}
```

## Implementation support

Tracked helper:

- `hermes_cli/service_principals.py`

The helper currently provides:

- policy loading from `~/.hermes/service_principals.json`
- canonical principal resolution from aliases
- allowed-service lookup for a principal
- boolean allow checks for a service/principal pair

This is intentionally small and reusable so future bridges can call it before
performing any service-specific action.

## Enforcement contract for future bridges

Future Zoom, Grain, and Granola bridges should:

1. resolve the active sender to a canonical principal
2. call the service-principal policy helper
3. deny by default when the service or principal is not allowed
4. use host-side OAuth/API credentials scoped to Richard's account
5. avoid passing those credentials into Hermes container env

Recommended denial behavior:

- return a structured authorization error
- do not leak whether alternate principals exist
- log the denied principal alias and requested service on the host side

## Relationship to memory and sender mapping

This policy is parallel to, but distinct from, the existing MemOS/MemPalace
principal mapping. Memory mapping decides who a conversation is about and which
scopes are valid for retrieval. Service-principal policy decides which external
accounts a sender is allowed to invoke.

For Richard's setup, both should resolve to the same canonical identity:

- `rj@stratminds.vc`

## Current limitation

This change introduces the policy and helper, not the service bridges
themselves. Hermes still needs actual Zoom, Grain, and Granola MCP or proxy
bridges before those services become callable.

## Hermes Spark memory identity status

Update: 2026-04-20

For the live local Hermes deployment on `rj-spark`, Richard is now the
canonical memory and service principal:

- `rj@stratminds.vc`

That canonical identity is used for:

- MemOS writes and lookups
- MemPalace writes and lookups
- service-principal policy gating for future Zoom, Grain, and Granola bridges

The Telegram alias observed by Hermes remains mapped to that canonical
principal, so memory and service authorization stay aligned across transport and
storage layers.
