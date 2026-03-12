# Evolution Cycle SOP v1

1) Run smoke suite (`evals/suites/smoke_v1.json`) and targeted regressions.
2) Identify top three bottlenecks by impact on success/cost/manual correction burden.
3) Select up to three scoped changes.
4) Ship behind flags or shadow mode when behavior-changing.
5) Re-run evals against baseline + holdout.
6) Promote only net-positive changes.
7) Capture lessons in memory/skills/tests.

Decision record required each cycle: PROMOTE | HOLD | REVERT.
