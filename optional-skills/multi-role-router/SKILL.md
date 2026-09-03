---
name: multi-role-router
description: >
  Hook that automatically routes each inbound message to the worker profile
  (role) best suited to handle it, using a fast auxiliary LLM classifier and
  a short conversation-history window to keep continuations in the current
  session. Installs as a message:pre_route hook under ~/.hermes/hooks/.
metadata:
  hermes:
    tags:
      - routing
      - multi-role
      - hook
      - automation
triggers: []
---
# Multi-Role Router Hook

Reference hook implementation for `message:pre_route` auto-routing.

Copy this directory to `~/.hermes/hooks/multi-role-router/` and restart
Hermes to activate it. Configure roles in `~/.hermes/config.yaml` under
`roles:` (see README.md for full configuration options).

## Known limitations

**First-turn isolation gap**: when the classifier picks a new role that has
no saved session for it yet, the current turn is delivered in the existing
inbound session. The role is written to `meta.yaml` immediately so the
gateway creates the new session on this turn, and isolation begins from
the next message onward. See README.md § "Pre-seeding sessions" to work
around this for strict isolation requirements.
