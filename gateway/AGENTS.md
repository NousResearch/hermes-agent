# Gateway Subtree Instructions

This file scopes messaging/gateway guidance to `gateway/` work. Root `AGENTS.md` still contains the non-negotiable project rules.

## Gateway invariants

- Preserve message role alternation. Do not inject synthetic user messages mid-loop.
- Do not mutate tools, memories, or system prompt state mid-conversation; defer cache-invalidating changes to a new session unless an explicit `--now` path exists.
- Gateway runtime reads user YAML directly in places; if adding config, check both CLI and gateway loader paths.
- Messaging working directory is `terminal.cwd` from `config.yaml`; do not revive `MESSAGING_CWD` as the canonical setting.
- Platform adapters with unique credentials should use scoped token locks from `gateway.status` to prevent two profiles using the same credential.

## Control-command guard pitfall

The gateway has two guards while an agent is running: the base adapter queues normal messages, then `gateway/run.py` intercepts control commands. New approval/control commands that must work while an agent is blocked must bypass both guards and dispatch inline.

## Verification

- Test the real gateway path when changing session routing, approvals, background notifications, or platform adapters.
- Run tests with `scripts/run_tests.sh ...`, never direct `pytest`.
