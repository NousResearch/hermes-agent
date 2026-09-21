# Execution-Ready Decision Records

Use when Sahil asks to convert analysis into an implementation-ready decision record, especially for KENSEI/Hermes architecture or governance changes.

## Required sections

A decision record should include:

1. Decision
2. Current state
3. Open risks
4. Implementation plan
5. Ownership
6. Verification steps
7. Next actions

## Wording discipline

Do not treat excluded broken items as resolved. If something is not verified or is known broken, label it as one of:

- open risk
- gap
- required follow-up
- not evidenced live
- blocked pending verification

Avoid phrases like “treated as resolved for this analysis” unless the item has actually been verified as resolved.

## Implementation and wiring considerations

A strategic decision record should be actionable, not just analytical. Include:

- exact files or governance artifacts to create/update
- command-level verification steps
- owner and reviewer for each workstream
- sequencing: documentation first, then templates/scripts, then runtime changes
- approval gates for broad config sweeps, provider/fallback/auth changes, service restarts, destructive cleanup, and profile deletion/deactivation
- what is intentionally not being changed yet

## Incremental implementation pattern

For governance/architecture implementation tasks:

1. Run read-only discovery first.
2. Create the decision record.
3. Create templates/schemas needed for execution.
4. Add read-only audit/verification scripts if useful.
5. Run syntax/format checks.
6. Run the audit script and report findings.
7. Stop before broad config edits or service changes unless Sahil has explicitly approved them.

## Example from 01/06/26 fleet architecture session

Created governance artifacts under `/home/kensei/.hermes/governance/`:

- `decision-records/multi-agent-fleet-architecture.md`
- `templates/goal-subgoal-kanban-template.md`
- `templates/sub-agent-profile-schema.md`
- `templates/temporary-sub-agent-log-schema.md`
- `templates/tool-skill-request-schema.md`
- `scripts/fleet_architecture_audit.py`

Verified with:

```bash
python3 /home/kensei/.hermes/governance/scripts/fleet_architecture_audit.py --profiles
python3 /home/kensei/.hermes/governance/scripts/fleet_architecture_audit.py --dispatch
python3 -m py_compile /home/kensei/.hermes/governance/scripts/fleet_architecture_audit.py
```

Reported open risks rather than fixing them silently:

- profile count 51
- four `dezzy-*` profiles missing SOUL.md
- knowledge layer unreliable until GitNexus/LadybugDB/GBrain/Light workflow is verified
- webhook autonomy not evidenced live
- skill cross-pollination not yet formalised in runtime
