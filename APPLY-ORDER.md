# Operator replay guide — apply the PR set onto v0.17.0 (or any later release)

This manifest lets you re-apply every private-overlay change onto a fresh upstream
release without reconstructing anything by hand. Proven against
**v0.17.0 = `2bd1977d8fad185c9b4be47884f7e87f1add0ce3`**
(see `APPLY-MATRIX-v0.17.0.txt` for the per-PR result and `REPRODUCE.sh` for the
end-to-end check).

## TL;DR — the set is almost entirely order-independent

The apply matrix proves **39 of 40 feature PRs apply cleanly onto a bare v0.17.0
checkout individually**, in any order — they touch disjoint or non-drifted files.
You do **not** need a strict dependency chain for those 39.

Only **2 PRs** overlap with v0.16.0 -> v0.17.0 upstream drift and need their
pre-resolved forward-compat branch instead of the raw PR:

| PR | feature | conflict on v0.17.0 | use this instead |
|----|---------|---------------------|------------------|
| #50056 | sqlite driver selection | `tests/hermes_cli/test_kanban_db.py` (test-only drift) | branch `forward-compat/50056-on-v0.17.0` (`e55b6481d`, v0.17.0 IS ancestor) |
| #48069 | mcp keepalive in-flight race | `tools/mcp_tool.py` (upstream refactor, keep-both) | branch `forward-compat/48069-on-v0.17.0` (`bcf9ff2eb`, v0.17.0 IS ancestor) |

## Deterministic recipe

```bash
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
git checkout -b replay/on-v0.17.0 "$V017"

# 1. The 38 clean feature PRs: cherry-pick each PR's net contribution (origin/main...<head>).
#    Order does not matter for these (disjoint/non-drifted). Logical grouping only:
#    - Foundational/independent first (reasoning, schema, prelude, routing, limits):
#        #48024 #48057 #48065 #48101 #49184 #49449 #49644
#    - Providers/CLI identity:  #50031 #50033 #50038 #50039 #50064
#    - Features/fixes:          #49915 #49916 #49917 #50021 #50022 #50032 #50040 #50041
#                               #50042 #50045 #50046 #50047 #50048 #50049 #50053 #50054
#                               #50055 #50068 #50073 #50146 #50155 #50296
#    - Test catch-ups (last):   #50066 #50078 #50080 #50086
#
# 2. The 2 drift PRs: take the forward-compat branch tip, NOT the raw PR:
git cherry-pick <range-for forward-compat/50056-on-v0.17.0>
git cherry-pick <range-for forward-compat/48069-on-v0.17.0>
#
# 3. Deferred-by-design (private/account-specific): apply from the deferred/ patch set
#    in THIS branch only if you want the private overlay (NOT public-contributable):
#      for p in deferred/*/*.patch; do git apply --3way "$p"; done
```

## Why "order does not matter" is safe here

Each PR was assembled as a single logical change off `origin/main`
(one-PR-per-concern), and the apply matrix 3-way-applies each PR's
`git diff origin/main...<head>` onto a *pristine* v0.17.0 — 39/40 succeed with no
prior PR applied. That is the strongest possible evidence of order-independence:
they don't depend on each other to apply. The only sequencing constraint is
"use the forward-compat branch for the 2 drifted PRs," captured above.

## Verification

`bash REPRODUCE.sh` (from this branch checkout) re-derives, from committed state:
- 139/139 src files covered (138 feature-PR + the 1 deferred opus-context test), 0 unmapped
- clean-checkout reproduction == working tree
- the integration branch stacks onto v0.17.0 with 0 conflict markers
`APPLY-MATRIX-v0.17.0.txt` lists the per-PR CLEAN/CONFLICT result with head SHAs.
