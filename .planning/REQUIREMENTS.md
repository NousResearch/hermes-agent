# Requirements

## Requirements

- [ ] **REQ-001** — All agent-creation surfaces resolve output caps with the same precedence: `HERMES_MAX_TOKENS` > `model.max_tokens` > matched provider `max_output_tokens` > provider-profile default. (milestone: v0.0)
- [ ] **REQ-002** — Interactive CLI, background CLI, gateway, oneshot, cron, TUI, and ACP agent constructors forward the resolved cap without silently falling back to the custom profile floor. (milestone: v0.0)
- [ ] **REQ-003** — Gateway `/model`, channel-override, and session-rehydration paths preserve provider-specific caps when no global cap is configured. (milestone: v0.0)
- [ ] **REQ-004** — Focused regression tests and configuration documentation prove the behavior, and the repository test gate passes with failures classified against baseline. (milestone: v0.0)
