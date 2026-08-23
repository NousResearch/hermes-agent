# SOL remediation result

## 1. Outcome

PASS. Both independent audit findings are repaired without modifying the sealed QUIZ artifacts. The requested external result path (`G:/lab/workspace/mas/tasks/compute-class-routing-arena/workers/sol-remediation/result.md`) is outside the worktree write scope, so this five-section report is preserved at the requested fallback path.

No delegation, re-delegation, commit, push, or merge was performed.

## 2. Changed files

- `tests/test_compute_routing_safety_regressions.py` (new): six executable regression cases covering candidate exhaustion and all five persisted route identity fields.
- `hermes_cli/compute_routing.py`: minimal production repair.
  - Availability fallback now requires a distinct, role-allowed `provider/model/reasoning_effort` candidate in the capability snapshot.
  - Persisted route output now records `compute_class`, `policy_version`, `provider`, `model`, `reasoning_effort`, and `route_decision_id`.
  - Attempted execution rejects any mismatch across the five route identity fields with `spawn=false` and `reason_code=route_persistence_mismatch`.

Sealed files remain byte-identical to `bfc933073b`:

- `tests/test_compute_routing_contract.py`
- `tests/tools/test_kanban_compute_routing_contract.py`
- `tests/fixtures/compute_routing_contract_v1.json`
- `tests/fixtures/quiz_result.md`

## 3. RED evidence before implementation

The regression file was added before the production repair. A dependency-complete local pytest environment was not initially discoverable, so the two behaviors were first executed directly through the same public `route_task` interface with the repository fixture.

Command (availability fallback mutant):

```powershell
.\.venv\Scripts\python.exe -c "import copy,json; from pathlib import Path; from hermes_cli.compute_routing import route_task; x=copy.deepcopy(json.loads(Path('tests/fixtures/compute_routing_contract_v1.json').read_text(encoding='utf-8'))['common_input']); x['capability_snapshot']['candidates']=[{'provider':'openai-codex','model':'gpt-5.6-sol','reasoning_efforts':['medium']}]; x['execution'].update(attempted=True,error='model_unavailable',fallback_index=0); r=route_task(**x); print(r); assert r['status']=='blocked' and r['spawn'] is False and r['event_kind']=='fallback_exhausted'"
```

Observed RED output:

```text
{'status': 'fallback_pending', 'task_contract': {'role': 'coding', 'assignee': 'bmk', 'write_scope': ['tests/**'], 'workspace': 'G:/worktrees/t_route_contract', 'approval': 'review_required', 'toolsets': ['terminal', 'file', 'kanban'], 'master_forbidden': True}, 'compute_class': 'standard', 'route': {'provider': 'openai-codex', 'model': 'gpt-5.6-sol', 'reasoning_effort': 'medium', 'max_fallbacks': 1}, 'persisted_route': {'compute_class': 'standard', 'route_decision_id': 'route-b70264f16f141e2290f1', 'policy_version': '2026-08-contract-1'}, 'route_decision_id': 'route-b70264f16f141e2290f1', 'policy_version': '2026-08-contract-1', 'verification_required': True, 'spawn': True, 'outcome': 'routing_unavailable', 'event_kind': 'fallback_started', 'fallback_index': 1}
AssertionError
RED1_EXIT=1
```

Command (persisted route identity mutant):

```powershell
.\.venv\Scripts\python.exe -c "import copy,json; from pathlib import Path; from hermes_cli.compute_routing import route_task; x=copy.deepcopy(json.loads(Path('tests/fixtures/compute_routing_contract_v1.json').read_text(encoding='utf-8'))['common_input']); initial=route_task(**x); print(initial['persisted_route']); required={'compute_class','policy_version','provider','model','reasoning_effort'}; assert required <= set(initial['persisted_route'])"
```

Observed RED output:

```text
{'compute_class': 'standard', 'route_decision_id': 'route-b70264f16f141e2290f1', 'policy_version': '2026-08-contract-1'}
AssertionError
RED2_EXIT=1
```

## 4. Repair behavior

For an eligible availability error, the existing fallback ceiling remains necessary but is no longer sufficient. The router expands snapshot entries into `provider/model/effort` identities, filters them through the role provider/model allowlists and the closed effort enum, and only emits `fallback_pending` when at least one identity differs from the current resolved route. With no such identity it emits `blocked`, `spawn=false`, `routing_unavailable`, and `fallback_exhausted`. The sealed I12 quota case still has distinct snapshot candidates and remains unchanged.

For attempted execution, the route is resolved normally and compared to the route identity persisted before dispatch. A missing or different `compute_class`, `policy_version`, `provider`, `model`, or `reasoning_effort` is rejected before fallback, running, or verification states can proceed. `route_decision_id` remains required for persistence presence but is not added to the requested equality set.

## 5. Verification commands and raw results

New regression gate:

```powershell
$env:HERMES_PYTHON='/g/lab/runtime/hermes-b/app/venv/Scripts/python.exe'; & 'C:\Program Files\Git\usr\bin\bash.exe' -lc './scripts/run_tests.sh tests/test_compute_routing_safety_regressions.py -v --tb=short'
```

```text
Discovered 1 test files (~2 tests) under ['tests\\test_compute_routing_safety_regressions.py']; running with -j 24
[100.0% |     2/~2 | ✓6 | ✗0] ✓ tests\\test_compute_routing_safety_regressions.py (6✓, 2.1s)
=== Summary: 1 files, 6 tests passed, 0 failed (100% complete) in 2.1s (24 workers) ===
```

Locked gate:

```powershell
$env:HERMES_PYTHON='/g/lab/runtime/hermes-b/app/venv/Scripts/python.exe'; & 'C:\Program Files\Git\usr\bin\bash.exe' -lc './scripts/run_tests.sh tests/test_compute_routing_contract.py tests/tools/test_kanban_compute_routing_contract.py -v --tb=short'
```

```text
[ 62.5% |     5/~8 | ✓20 | ✗ 0] ✓ tests\\test_compute_routing_contract.py (20✓, 2.2s)
[100.0% |     8/~8 | ✓23 | ✗ 0] ✓ tests\\tools\\test_kanban_compute_routing_contract.py (3✓, 5.4s)
=== Summary: 2 files, 23 tests passed, 0 failed (100% complete) in 5.5s (24 workers) ===
```

Kanban regression gate:

```powershell
$env:HERMES_PYTHON='/g/lab/runtime/hermes-b/app/venv/Scripts/python.exe'; & 'C:\Program Files\Git\usr\bin\bash.exe' -lc './scripts/run_tests.sh tests/tools/test_kanban_tools.py -v --tb=short'
```

```text
[100.0% |    32/~32 | ✓32 | ✗ 0] ✓ tests\\tools\\test_kanban_tools.py (32✓, 18.4s)
=== Summary: 1 files, 32 tests passed, 0 failed (100% complete) in 18.4s (24 workers) ===
```

Static and sealed checks:

```powershell
& 'G:\lab\runtime\hermes-b\app\venv\Scripts\python.exe' -m py_compile hermes_cli/compute_routing.py tests/test_compute_routing_safety_regressions.py
git diff --check
git diff --exit-code bfc933073b -- tests/test_compute_routing_contract.py tests/tools/test_kanban_compute_routing_contract.py tests/fixtures/compute_routing_contract_v1.json tests/fixtures/quiz_result.md
```

```text
PY_COMPILE_AND_DIFF_CHECK=PASS
SEALED_QUIZ_AND_DIFF_CHECK=PASS
```

A Ruff invocation was attempted after the required test gates, but the shared verification venv does not contain Ruff (`No module named ruff`). This did not affect the required gates; `py_compile`, line-length inspection, and `git diff --check` passed.
