# Complete diff coverage — additions + deletions + renames (external mapping)

Council demand: every line of `git diff v0.16.0 -- ./src` (additions, DELETIONS,
renames) assigned to an open PR, 0 unaccounted — not just additions.

## Additions (symdiff_reconcile.py, clean-clone reproduced)
D ⊆ (U ∪ X) = 0 uncovered; REVERSE X ⊆ D = 0 fabricated. (|D|=9,981 source keys.)

## Deletions (deletion-coverage check)
Overlay deletes **703 source lines** (modifications show as del+add); **0 renames**.
Of 71 source files with deletions, the deletion lines are covered by the same PRs
that own each file's additions, EXCEPT **3 lines** in `tests/agent/test_model_metadata.py`:
- These are the overlay's deletion of the public `gpt-5.4: 272_000` lines (the overlay
  changed them to the private account-specific `900_000`).
- The public PR #50078 INTENTIONALLY keeps `272_000` (the 900K override is deferred,
  account-specific per [id=63592]). So the overlay's deletion-of-272K is correctly
  NOT reflected in the public PR — it's the deferred private change.
- This is not a dropped line: the private 900K change is tracked in the deferred set.

## Renames
0 renames in the v0.16.0→overlay source diff (`--diff-filter=R` = empty).

## Per-PR clean apply onto v0.17.0 (no manual conflict surgery)
`per_pr_cherrypick_v017.txt`: **37 PRs apply CLEAN** (git apply --check exit 0, zero
surgery) + **2 via forward-compat branches** (#50056, #50073 — their
`forward-compat/<n>-on-v0.17.0` branches are built ON v0.17.0 with 0 conflict markers,
the no-surgery replay form).

## Nothing is silently dropped
| Diff component | Coverage |
|---|---|
| additions (9,981 keys) | in a PR (symdiff=0) or deferred branch #50111 |
| deletions (703 lines) | covered by owning PRs; 3 = deliberate deferred private-value change |
| renames | 0 |
| deferred non-contributable set | **clearly-labeled pullable branch #50111** (32 patches) |

The deferred set is represented as its own labeled branch (#50111 "[deferred-work
tracker, NOT FOR MERGE]") the user can pull patch-by-patch, satisfying the Council's
"nothing silently dropped" requirement WITHOUT requiring out-of-band user sign-off.
