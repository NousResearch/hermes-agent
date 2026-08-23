# SOL implementation result

## Outcome

PASS. The sealed QUIZ contract at commit `bfc933073b` is implemented without
changing any sealed test or fixture. The requested external result path is
outside this worktree write scope, so this report is preserved at the specified
fallback path.

## Implementation

- Added pure `hermes_cli.compute_routing.route_task` with deterministic
  precedence for all five compute classes.
- Preserved role, assignee, write scope, workspace, approval, toolsets, and
  master prohibition byte-for-byte in the returned task contract.
- Added explicit route metadata persistence output, capability gating,
  availability-only fallback with a Core ceiling of one, attended routing,
  verification requirements, and separate actual-route observations.
- Added shadow behavior that records the recommendation while returning empty
  dispatch overrides.
- Connected `reasoning_effort` through the existing `kanban_create` schema and
  handler to DB normalization, created event, show output, and worker argv.
- Added no dependency and performed no delegation, commit, merge, or push.

## Sealed artifact integrity

Working-tree hashes equal the hashes in `bfc933073b`:

```text
tests/test_compute_routing_contract.py
0a3d37e764f35357a2dd616ca24a342f10d94d00 == 0a3d37e764f35357a2dd616ca24a342f10d94d00
tests/tools/test_kanban_compute_routing_contract.py
e67470e2df2e95f03048f709376a8058c91621f4 == e67470e2df2e95f03048f709376a8058c91621f4
tests/fixtures/compute_routing_contract_v1.json
2ca4fdfb7bac121627ad34a358deb6e93e9c7cb0 == 2ca4fdfb7bac121627ad34a358deb6e93e9c7cb0
tests/fixtures/quiz_result.md
b0d05b3ec16d0f4c5af9379b89e114c09a22ba18 == b0d05b3ec16d0f4c5af9379b89e114c09a22ba18
```

## Normal and mutant raw result

Command: `pytest -vv --tb=short tests/test_compute_routing_contract.py`

```text
collected 20 items
test_fixture_is_closed_complete_and_structurally_valid PASSED
N01-specialist-precedence PASSED
N02-architect-precedence PASSED
N03-deep-precedence PASSED
N04-standard-precedence PASSED
N05-quick-precedence PASSED
I01-role-boundary-mutation PASSED
I02-posthoc-class-inference PASSED
I03-provider-without-model PASSED
I04-invalid-reasoning-default-fallback PASSED
I05-catalog-only-capability PASSED
I06-work-failure-model-fallback PASSED
I07-third-spawn-chain PASSED
I08-fallback-widens-safety-boundary PASSED
I09-verified-without-actual-model PASSED
I10-quick-skips-verification PASSED
I11-unattended-architect-spawn PASSED
I12-quota PASSED
I12-work PASSED
test_i12_expected_contracts_are_not_aliases PASSED
20 passed in 1.07s
```

The normal routes are 5/5 PASS and the known-bad invariant set is 12/12 DROP.
I12 has two variants to prove quota and work failures remain distinct.

## Verification

```text
Locked combined gate:
23 passed in 3.35s

Existing Kanban model/spawn regression:
25 passed in 16.10s

Direct shadow and positive actual-observation assertions:
shadow_and_actual_observation: PASS

Ruff:
All checks passed!

git diff --check:
PASS
```

Regression files:

- `tests/plugins/test_kanban_model_override.py`
- `tests/hermes_cli/test_kanban_worker_spawn_toolsets.py`
- `tests/hermes_cli/test_kanban_dispatch_tick_hook.py`

## Production diff/stat

```text
hermes_cli/compute_routing.py | 345 insertions (new)
hermes_cli/kanban_db.py       |   1 insertion
tools/kanban_tools.py         |  14 insertions
3 production files changed, 360 insertions
```

Changed files including this required report: 4.

```text
M  hermes_cli/kanban_db.py
M  tools/kanban_tools.py
?? hermes_cli/compute_routing.py
?? tests/fixtures/sol_result.md
```
