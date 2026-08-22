# LAH Repo Router schema v4 regression

Run:

```bash
node scripts/test-v4-router.cjs
node scripts/validate-routing-drift.cjs
```

The v4 fixture corpus lives in `v4-adversarial-fixtures.json`. It covers role-aware routing, cross-role writes, explicit target conflicts, archived memory, skills, business assets, CodeGraph fail-closed behavior, negative routing, and order-independent semantics.

Prior v3 result files are historical evidence only, not certification evidence.
