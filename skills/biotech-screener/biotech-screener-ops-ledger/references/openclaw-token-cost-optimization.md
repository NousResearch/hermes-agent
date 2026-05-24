# OpenClaw Token Cost Optimization (Absorbed)

Source skill absorbed: `openclaw-cost-optimization` (2026-05-18 consolidation pass).

This reference captures cost-control heuristics for OpenClaw-driven biotech operations.

## Core Heuristics

- Token-consuming jobs are typically cron jobs that invoke LLM reasoning (skill-attached or prompt-heavy jobs).
- High-frequency schedules (daily or more often) dominate spend.
- Large-context synthesis jobs are cost multipliers.

## Cost-Control Levers

1. **Reduce frequency for non-critical synthesis jobs** (e.g., daily -> weekly).
2. **Pause non-essential jobs** during maintenance or budget pressure windows.
3. **Prefer script-only watchdogs** where deterministic checks are enough.
4. **Use local delivery where possible** for internal monitors.

## Operator Review Questions

- Which jobs must remain daily for risk control?
- Which jobs are informational and safe to run weekly?
- Which jobs can become deterministic script checks instead of LLM calls?
- What is the target weekly token budget ceiling?

## Biotech Fleet Pattern (Example)

- **Keep daily**: fleet triage, model tracker, production contract checks, streak/risk monitors.
- **Move to weekly** when non-urgent: broad synthesis ledgers, weekly sweeps, memory/skill harvesting.
- **Keep script-only**: auth sync, API rate-limit watchdogs, simple liveness checks.

## Validation After Schedule Changes

- Confirm schedules with `hermes cron list`.
- Compare token usage trend before/after over at least one full weekly cycle.
- Verify no critical control-plane blind spots were introduced.

## Guardrail

Cost reduction must never disable required governance checks or first-fire validation gates. Preserve risk controls first, then optimize frequency.