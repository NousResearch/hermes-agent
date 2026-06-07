# QA Report — Factory Runtime Remediation

## Test evidence

### Prior runtime autonomy repair

Focused Factory suite used during the first remediation phase:

```bash
./scripts/run_tests.sh tests/hermes_cli/test_factory_canonical_runtime.py tests/hermes_cli/test_factory.py tests/tools/test_factory_tools.py
```

Observed result during that phase:

```text
23 tests passed, 0 failed
```

### Repo-first Runtime Contract v1

RED was verified first: the new tests failed because `docs_not_indexed`, `uncommitted_project_artifacts`, and critical readiness commit/index blockers did not exist yet.

RED command:

```bash
/home/jean/Projects/hermes-agent-original/venv/bin/python -m pytest tests/hermes_cli/test_factory_canonical_runtime.py -q
```

Observed RED excerpt:

```text
FAILED test_factory_reconciliation_detects_docs_missing_from_documentation_index
FAILED test_factory_reconciliation_detects_uncommitted_project_artifacts
FAILED test_factory_critical_readiness_requires_index_and_commit_checkpoint
3 failed, 17 passed
```

GREEN commands:

```bash
/home/jean/Projects/hermes-agent-original/venv/bin/python -m pytest tests/hermes_cli/test_factory_canonical_runtime.py -q
/home/jean/Projects/hermes-agent-original/venv/bin/python -m pytest tests/hermes_cli/test_factory_canonical_runtime.py tests/tools/test_factory_tools.py -q
```

Observed GREEN:

```text
20 passed in 1.66s
22 passed, 1 warning in 7.80s
```

## Compile evidence

```bash
/home/jean/Projects/hermes-agent-original/venv/bin/python -m py_compile hermes_cli/factory_pg.py
```

Observed result: success.

## Live smoke evidence

### Blocker detector

```bash
python3 ~/.hermes/scripts/factory_blocker_detector.py
```

Observed before this increment:

```text
db_backend=agent_core_postgres
classified=0
alerts=0
needs_attention=false
```

### Controlled orchestrator tick

```bash
FACTORY_TICK_PROJECT_ID=factory-runtime-remediation python3 ~/.hermes/scripts/factory_orchestrator_tick.py
```

Observed before this increment:

```text
db_backend=agent_core_postgres
claimed=null
alerts=[]
needs_attention=false
```

## QA verdict

GREEN for the code-level repo-first Runtime Contract v1 in the dedicated worktree. Final project delivery still requires a clean git checkpoint and any separate Notion/human PM projection reconciliation Jean wants to enforce.
