# Managed-Agent Sandbox-Capable Workers

Status: in_progress
Owner: Hermes main / Codex worker
Started: 2026-05-11 23:08
Branch: feat/managed-agent-sandbox-workers
Multica project: Hermes Agent / 966bde1c-667a-42e9-8088-56e3cbe97b79
Multica parent: JEF-219 / ee176b2d-8671-4506-87b6-b3fd37af0bbf
Multica implementation: JEF-220 / 3f996e02-fd46-4d94-a3cf-86c0dbfd9da3
Multica review gate: JEF-221 / 36a751a2-0899-437d-bcda-5f97e4857f03

## Scope
Implement the first production-safe slice of the Managed-Agents style Multica/Hermes blueprint inside Hermes Agent's Kanban worker runtime:

- Coordinator + task graph remains Kanban/Multica-like task routing.
- Workers gain clearer stateless/sandbox-capable contracts.
- Workspaces gain pre-run checkpoint evidence and post-run artifact/diff evidence.
- Worker policy is explicit and injected into runtime context/env.
- True OS/container sandbox stays opt-in/future; this phase must not claim container isolation when it is only process/workspace/worktree isolation.

## Phases

### Phase 1 — Stateless workspace evidence
Status: done

Acceptance:
- Worker dispatch records a pre-run workspace snapshot/checkpoint event.
- Worker run metadata includes workspace path/kind and checkpoint id/path where available.
- Post-run completion/block metadata can include artifact/diff evidence without manual shell scraping.

### Phase 2 — Permission/policy layer
Status: done

Acceptance:
- Kanban tasks can carry a policy/profile such as `standard`, `read_only`, `code_edit`, `test_only`, `sandbox_strict`.
- Policy is validated, stored, visible in task output/context, and exposed to workers via environment.
- Default remains backward compatible.
- Policy language is explicit: it is a runtime contract/guardrail, not hard OS sandbox unless the workspace backend provides it.

### Phase 3 — Checkpoint/fork/rollback semantics
Status: done

Acceptance:
- Workspace checkpoint data is deterministic and bounded.
- For git workspaces, diff/status evidence is captured in run metadata/events.
- A rollback/discard-oriented helper/API exists for workspace snapshots where safe.
- Tests cover checkpoint creation and evidence capture without touching real user repos.

### Phase 4 — Sandbox runner interface
Status: done

Acceptance:
- Code exposes a sandbox-capability descriptor for each workspace/policy combination.
- Dispatch metadata distinguishes `workspace_isolation`, `policy_contract`, and `os_sandbox` capabilities.
- Docs/tests prevent overclaiming true sandbox.

## Files likely affected
- hermes_cli/kanban_db.py
- hermes_cli/kanban.py
- tools/kanban_tools.py
- tests/hermes_cli/test_kanban_db.py or new focused tests
- docs/website docs if needed

## Safety boundaries
- No production deploys.
- No secrets, `.env`, tokens, DB dumps, raw videos, or model weights in `.hermes`.
- Do not modify user config unless explicitly needed.
- If implementing true OS/container sandbox requires external runtime setup, leave it behind capability detection and tests; do not install dependencies automatically.

## Verification plan
- Run focused pytest for new/changed Kanban tests.
- Run `python -m pytest tests/hermes_cli/test_kanban_db.py tests/tools/test_kanban_tools.py -q` if feasible.
- Run `git diff --check`.
- Inspect git status and sensitive-file patterns before commit.

## Discussion / decision notes
- 2026-05-11: Dragon clarified current system has no true sandbox. Target is sandbox-capable workers; current safe implementation should start with worktree/workspace isolation, explicit policy, checkpoints, and evidence.
- 2026-05-11: Implemented backward-compatible worker_policy/checkpoint_policy fields, checkpoint/evidence metadata, capability descriptors with os_sandbox=false for local workers, safe rollback-plan descriptor (no execution), CLI/dashboard create support, and tests.
- 2026-05-11 verification: `python -m pytest tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py tests/tools/test_kanban_tools.py -q` => 136 passed, 4 dependency deprecation warnings. `git diff --check` => clean.
