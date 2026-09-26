# Test — Continuity Forge Round-Trip

**Input**: Symbolic architecture exported, ingested into Forge, then project state re-hydrated in a later session with some Forge-side scene changes.

**Expected**:
- Motif IDs, observed forms, and provenance match.
- Ledger is correctly re-hydrated.
- Revision diff is automatically suggested for any Forge-induced changes.
- Mutation requirements are still active for future proposals.
