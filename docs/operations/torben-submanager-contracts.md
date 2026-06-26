# Torben Submanager Contracts

Torben is the only user-facing Signal operator. EA, GTM/Magnus, and
Finance/Ratatosk are backend capability scopes behind Torben.

## Shared Contract

Resolved gate policy as of 2026-06-26:

- Public/personal mutations are allowed only for narrow synthetic test canaries
  or after a specific approved action handle. Unattended public X/LinkedIn
  writes remain blocked.
- Live finance is approved only for a tiny testing canary after mandate,
  consent, kill switch, pre-trade guard, and reconciliation gates pass. The
  current Ratatosk circuit-breaker halt remains a hard block.

Every backend scope must emit Torben-owned artifacts with these fields:

- `task`: stable task name.
- `wakeAgent`: whether Signal should receive anything.
- `generated_at`: UTC timestamp.
- `actions`: staged Torben `ActionRecord` objects when user action is possible.
- `public_actions_taken`: integer, always zero unless a public-write adapter was explicitly approved and executed.
- `external_mutations`: integer, always zero unless an approved mutation adapter executed.
- `text`: Signal-safe copy; empty when `wakeAgent=false`.

Every action must include:

- handle prefix: `EA-*`, `GTM-*`, or `FIN-*`.
- evidence ids or source refs.
- allowed next actions.
- mutation type.
- mutation status.
- provider/source.
- approval boundary.
- rollback or irreversibility note when a mutation can execute.

## Scope Ownership

EA:

- Owns Gmail realtime triage, relationship learning, calendar alignment,
  meeting prep, daily brief, email hygiene recommendations, and staged replies.
- May maintain private busy blocks where policy allows.
- Must not send email, trash/archive/delete mail, or edit attendee-impacting
  calendar events without an approved handle.

GTM / Magnus:

- Owns intelligence collection, source health, article opportunities, drafts,
  X/LinkedIn response candidates, visuals, Grok/X Search judgment, and X
  algorithm pressure tests.
- Public writes stay blocked until a public mutation decision is resolved and
  adapter canaries pass.

Finance / Ratatosk:

- Owns Robinhood v0.1 research, watchlist/candidate generation, risk context,
  performance scoring, and later live execution only after explicit mandate.
- Research signals can create candidates, but cannot trade directly.
- Broker order submission remains blocked until mandate, consent, kill switch,
  guard, and reconciliation canary pass.

## Cron Policy

Crons are triggers, not static decision makers.

- Static code may collect evidence, cap budgets, invoke a bounded LLM run,
  validate schema, enforce policy, dedupe, and stage actions.
- LLMs may judge nuanced signal, synthesize context, draft copy, and rank
  candidates.
- Successful no-op maintenance stays silent.
- Repeated failure fingerprints should be deduped or routed through the
  live-profile investigation loop rather than spamming Signal.

## Production Invariants

- `external_mutations=0` unless an approved action handle was resolved and the
  provider returned a mutation result.
- `public_actions_taken=0` unless an approved X/LinkedIn adapter executed.
- `broker_orders_submitted=0` unless the live finance decision, mandate,
  consent, kill switch, pre-trade guard, and reconciliation canary all pass.
- A hidden submanager must not send its own user-facing Signal messages.
- Torben owns final copy, dedupe, action handles, approval state, and health
  escalation.
