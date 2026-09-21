# Fleet Reference Comparison and Execution-Ready Decision Records

Session source: 01/06/26 validation of multi-agent fleet comparison against three Hermes reference docs.

## When this applies

Use this when Sahil asks to validate an architecture analysis, turn it into a decision record, or proceed from comparison into implementation.

## Pattern

1. Validate claims against source docs and live state.
2. Preserve original direction if valid.
3. Correct overstated claims.
4. Label broken/unverified items as open risks, not resolved work.
5. Add implementation/wiring considerations before calling it execution-ready.
6. Separate governance/docs changes from runtime/code changes when reporting back.

## Required decision-record sections

- Decision
- Current state
- Open risks
- Implementation plan
- Ownership
- Verification steps
- Next actions

## Wording rule

Never write “treated as resolved for this analysis” for known broken items. Use:

> Known broken items excluded from reference-fit scoring, but still open operational risks.

## Common checks from the session

- Profile count may differ from stale analysis. Verify live count.
- “All profiles have SOUL.md” is stronger than “all active leads have SOUL.md”. Check before claiming.
- Webhook autonomy should be called a gap unless live webhook subscriptions/triggers are evidenced.
- Knowledge accumulation is not solved by memory alone. Wiki/runbook/provenance layers must be valid.
- Specialist gateways must not own cron/Kanban dispatch. Kensei owns cron + dispatcher.

## Reporting preference learned

When Sahil asks what was done, keep it short and explicitly answer:

- whether KenseiAgent was changed
- what was completed
- what was not completed
- what needs Sahil approval

This prevents governance artefacts being mistaken for runtime implementation.
