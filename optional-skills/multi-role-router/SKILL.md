---
name: multi-role-router
description: Routes inbound gateway messages to per-role sessions.
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

- **No per-role system prompts yet**: routing isolates sessions/context, but each
  role does not yet carry its own system prompt. The role is selected by the
  classifier and its session is used, but the underlying agent behaviour is
  otherwise unchanged. A future version could inject a role-specific
  `system_prompt` override along with the session switch.
