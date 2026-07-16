# BETA-004 — Structured delegation contract

`agent.beta.delegation` converts routing decisions into isolated `DelegationTask` objects and submits them as one synchronous batch to Hermes' existing `delegate_task`. Hermes remains responsible for child lifecycle, parallel execution, concurrency limits, progress events, timeouts, and isolated conversations.

Each child receives only the task's minimal context, constraints, risk, resolved tool allow-list, expected deliverable, timeout, task ID, and correlation ID. It is a leaf subagent and is instructed not to contact the Chief or write Beta's strategic memory.

Specialist output must validate as `SpecialistResult`. Invalid JSON, schema violations, or mismatched identifiers become `contract_error`; timeout/failure entries do not discard valid sibling results.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_delegation.py
```
