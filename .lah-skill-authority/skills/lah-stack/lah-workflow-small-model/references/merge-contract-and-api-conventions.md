# Merge Contract & API Conventions

## Merge Contract (Gate 9)

A tranche is NOT complete until:
1. PR is created
2. PR is reviewed (whole-branch review)
3. PR is verified (all gates green)
4. PR is merged
5. Remote main reflects the merge
6. Post-merge smoke runs clean
7. Merge SHA is recorded

PR creation alone is insufficient. Do not proceed to the next tranche until the merge is confirmed on the remote main branch.

## Non-Destructive GET Convention

GET endpoints MUST be observational and non-destructive:
- MUST NOT mutate state, clear buffers, delete data, or trigger side effects
- MUST return the same data on repeated identical calls
- If explicit state mutation is needed (e.g., clearing a cache), use a separate DELETE or POST endpoint as the mutation boundary

### Pattern

```
GET  /api/resource?sessionId=X  → read-only, returns data
DELETE /api/resource?sessionId=X → explicit mutation, clears data
```

### Verification

Tests must verify:
- Repeated identical GET returns the same data (non-destructive)
- DELETE clears the data (explicit mutation)
- GET summary does not mutate the buffer
- DELETE does not affect other sessions