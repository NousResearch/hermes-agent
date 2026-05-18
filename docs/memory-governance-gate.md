# Memory Governance Gate

## Purpose

The memory governance gate classifies candidate information before any caller
decides where durable knowledge should go. It is intentionally review-only:
it does not write Hermes memory, does not write Obsidian, does not edit Honcho,
and does not create skills.

Obsidian remains the canonical curated knowledge store. Honcho remains runtime
recall only. Hermes persistent memory remains limited to boot-critical stable
facts and preferences.

## API

The implementation lives in `agent/memory_governance.py`.

Primary entry points:

- `classify_memory_candidate(candidate, source_type="unknown")`
- `enqueue_memory_governance_review(decision, metadata=None)`
- `load_memory_governance_review_queue()`
- `memory_governance_queue_path()`

The classifier returns a `MemoryGovernanceDecision` with:

- `label`
- `confidence`
- `reason`
- `requires_approval`
- `destructive`
- `source_type`
- `candidate_summary`
- `suggested_artifact_path` when a likely artifact destination is known

## Labels

- `HERMES_MEMORY`: Boot-critical user preferences, profile role facts, or
  environment facts that should be visible at startup. These decisions require
  approval because persistent Hermes memory is injected into future sessions.
- `HONCHO_RUNTIME_ONLY`: Runtime recall context that may help conversational
  continuity but should not become curated knowledge or persistent memory.
- `OBSIDIAN_PROMOTE`: Canonical project decisions, research syntheses, and
  source-backed evidence that should be reviewed for curated Obsidian promotion.
- `SKILL`: Reusable procedures, checklists, workflows, or runbooks.
- `PROJECT_STATE`: Working state such as Money Flow Radar cursors, checkpoints,
  alerts, and task progress. This is explicitly not USER memory.
- `SESSION_ONLY`: Short follow-up replies, cron replies, and transient progress.
- `REJECT`: Secrets, raw credentials, empty candidates, or destructive memory
  removal and compaction proposals.

## Review Queue Storage

Review items are stored under the active profile-local Hermes data directory:

```text
${HERMES_HOME}/memory_governance/review_queue.json
```

The queue stores the decision and sanitized candidate summary. It does not write
to:

- `${HERMES_HOME}/memories/USER.md`
- `${HERMES_HOME}/memories/MEMORY.md`
- Obsidian vault paths
- Honcho provider storage

## Limits

This is not a full policy engine. The classifier is a conservative heuristic
gate designed to prevent obvious durable-storage mistakes and produce reviewable
queue items. Callers are responsible for deciding whether and how to act on a
queued decision.

The review queue is local JSON storage, not a cross-device durable workflow
queue. Secret-looking candidates are rejected and summarized with redaction.
Destructive compaction or removal proposals are marked `destructive` and
`requires_approval`.

## Verification

Targeted classifier, queue, and existing memory provider tests:

```bash
scripts/run_tests.sh tests/agent/test_memory_governance.py tests/agent/test_memory_provider.py -q
```

The local canonical wrapper was attempted first. In this environment it stopped
before test collection because the shared Hermes virtualenv lacks `pip` while
the wrapper tried to install `pytest-split`. The same test target was then run
with the wrapper-selected Python:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts= tests/agent/test_memory_governance.py tests/agent/test_memory_provider.py -q
```

Result:

```text
72 passed
```

Additional checks:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m py_compile agent/memory_governance.py tests/agent/test_memory_governance.py
git diff --check
```
