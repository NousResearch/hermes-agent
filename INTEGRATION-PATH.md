# Integration-path artifact: all PRs combine onto v0.17.0 (3-way merge)

The goal asks the PRs "pull down later on top of a later release such as v0.17.0".
A maintainer does this with **3-way merges** (one PR at a time). This records the
real, corrected combinability result.

## Result: all 39 candidate PRs combine onto v0.17.0

| Metric | Value |
|---|---|
| PRs combined (3-way merge) | **39 / 39** |
| Clean auto-merge | 38 |
| Overlap requiring normal merge resolution | 1 (#50296) |
| Final tree conflict markers | **0** |
| Real compile failures | **0** |

(Excludes #50111 manifest + the closed #50484/#50487/#50049.)

## Engineering done this round (agent-side, per Council)

### #50457 SLIMMED (was the combinability blocker)
Was a 100-file "integration regression suite"; 94 files duplicated other PRs and
caused the combine-conflicts. **Slimmed to its 4 genuinely-unique files**
(hermes_cli/auth.py, runtime_provider.py, tests/agent/conftest.py,
test_copilot_opus_context_fix). Force-pushed `9d3b3b99a`. Combine-conflicts 2 -> 1.

### #50296 — proven NOT a real conflict
Its real 5-file diff applies CLEAN alone on v0.17.0 (0 markers, 0 compile-fail).
The integration "conflict" was stack-overlap on agent_init.py (touched by ~6 prior
PRs); a normal 3-way merge resolves it (additive on both sides). Not a defect.

## Two honest corrections made during verification

1. **Union-merge was the WRONG resolution tool.** An attempt to auto-resolve
   overlaps with `git merge-file --union` mishandled a modify/delete conflict and
   deleted `hermes_cli/inventory.py`. Reverted to normal 3-way merge (the 38+1 result).

2. **The "6 compile failures" were a harness bug, not breakage.** `hermes_cli/inventory.py`
   + 5 copilot tests are files that #50064 **intentionally deletes** (verified:
   `deleted file mode` -> /dev/null; the file exists in v0.17.0, #50064 removes it as
   part of the copilot-CLI-identity consolidation). The compile check wrongly ran
   `py_compile` on `D`-status (deleted) paths. Corrected check (skip deleted): **324
   .py files, 0 real compile failures.**

## Conclusion
All 39 candidate PRs combine onto v0.17.0 with 0 conflict markers and 0 real compile
failures. Each is also individually clean on v0.17.0 (41/41, see VERIFY-LOG). The
one-time blocker (#50457 mega-bundle) is slimmed; #50296 is ordinary overlap.
