# Workflow E2E Findings Remediation Verification

Date: 2026-07-07
Branch: `feat/workflow-graph-engine`
Baseline before remediation: `44d81bcd2 fix(workflows): address remaining review comments`

## Finding coverage

| Finding | Fix | Evidence |
|---|---|---|
| WF-E2E-001 unsupported primitives | Added central runtime capability registry and wired unsupported primitive rejection through CLI validation, dashboard API, workflow tools, and assistant validation. | `tests/hermes_cli/test_workflows_capabilities.py`, `tests/hermes_cli/test_workflow_cli.py`, `tests/tools/test_workflow_tools.py`, `tests/plugins/test_workflows_dashboard_plugin.py`; manual smoke: `workflow validate /tmp/unsupported-send-message.yaml` exited 1 with `unsupported node type: send_message on node start`. |
| WF-E2E-002 result_contract array/object | Dispatcher now enforces `array` and `object` result contract types in addition to scalar/enums. | `tests/hermes_cli/test_workflows_dispatcher.py`; full gate below. |
| WF-E2E-003 cycles | `validate_graph` rejects self and multi-node cycles before execution. | `tests/hermes_cli/test_workflows_spec.py`, updated engine coverage; manual smoke: `workflow validate /tmp/accidental-cycle.yaml` exited 1 with `workflow graph contains cycle: loop -> loop`. |
| WF-E2E-004 repair_attempts | Assistant draft/refine now retries invalid drafts with repair prompts bounded by `repair_attempts`. | `tests/hermes_cli/test_workflows_assistant.py`; full gate below. |
| WF-E2E-005 prompt-first 500s | Dashboard assistant endpoints now return typed, privacy-safe error envelopes and the UI renders remediation hints. | `tests/plugins/test_workflows_dashboard_plugin.py`, `tests/plugins/test_workflows_dashboard_assets.py`; full gate below. |
| WF-E2E-006 dashboard a11y/responsive | Dashboard includes a non-canvas workflow cell list, keyboard/ARIA edit targets, responsive editor layout, and minimap styling. | `tests/plugins/test_workflows_dashboard_assets.py`; `node --check plugins/workflows/dashboard/dist/index.js`. |
| WF-E2E-007 docs/examples | Workflow examples now declare enforced `result_contract` blocks; docs explain contracts, validation, unsupported primitives, assistant errors, and privacy. | `tests/hermes_cli/test_workflows_docs_examples.py`; `workflow validate examples/workflows/code-change-review.yaml` and `workflow validate examples/workflows/research-triage.yaml` both printed `OK: ... v1`. |
| WF-E2E-008 redaction warnings | Added recursive workflow payload redaction and dashboard warnings; dashboard display responses redact inputs/context/node runs/events while raw local execution storage remains intact for dispatcher correctness. | `tests/hermes_cli/test_workflows_redaction.py`, `tests/plugins/test_workflows_dashboard_plugin.py`, `tests/plugins/test_workflows_dashboard_assets.py`; full gate below. |
| WF-E2E-009 body size cap | Dashboard workflow definition/assistant endpoints enforce a 1,000,000 byte request cap and return HTTP 413 with `workflow_request_too_large`. | `tests/plugins/test_workflows_dashboard_plugin.py`; manual smoke via FastAPI TestClient returned `oversized_status=413` and `workflow_request_too_large`. |

## Focused verification evidence

- Focused workflow regression suite: `256 passed in 10.69s`.
- Targeted compile/syntax: `compileall` on touched modules passed; `node --check plugins/workflows/dashboard/dist/index.js` passed.
- Manual unsupported primitive smoke: exit 1, `Error: unsupported node type: send_message on node start`.
- Manual cycle smoke: exit 1, `Error: workflow graph contains cycle: loop -> loop`.
- Manual example validation smoke:
  - `OK: code-change-review v1`
  - `OK: research-triage v1`
- Manual oversized dashboard body smoke: HTTP 413, detail code `workflow_request_too_large`.
- Frontend/source check: `node --version` = `v26.0.0`, `npm --version` = `11.12.1`; `node --check plugins/workflows/dashboard/dist/index.js` passed; `npm install --workspace web --package-lock=false --no-audit --no-fund && npm run build --workspace web` completed successfully. The web build writes ignored `hermes_cli/web_dist` assets and does not generate `plugins/workflows/dashboard/dist/*`, so no unrelated generated web output was committed.

## Final quality gate

Command:

```bash
PY=/Users/christopherwilloughby/.hermes/hermes-agent/venv/bin/python
$PY -m pytest \
  tests/hermes_cli/test_workflows_capabilities.py \
  tests/hermes_cli/test_workflows_assistant.py \
  tests/hermes_cli/test_workflows_spec.py \
  tests/hermes_cli/test_workflows_db.py \
  tests/hermes_cli/test_workflows_db_versions.py \
  tests/hermes_cli/test_workflows_engine.py \
  tests/hermes_cli/test_workflows_expr.py \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/hermes_cli/test_workflows_e2e.py \
  tests/hermes_cli/test_workflow_cli.py \
  tests/hermes_cli/test_workflows_redaction.py \
  tests/hermes_cli/test_workflows_docs_examples.py \
  tests/tools/test_workflow_tools.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_assets.py \
  tests/gateway/test_workflow_dispatcher_integration.py \
  tests/test_toolsets.py \
  tests/tools/test_delegate_composite_toolsets.py \
  -q
$PY -m compileall hermes_cli tools plugins/workflows/dashboard
node --check plugins/workflows/dashboard/dist/index.js
```

Result: `379 passed in 13.06s`; compileall passed; node syntax check passed.

## Commit summary after baseline

```text
402dc935c fix(workflows): centralize implemented primitive validation
7da30c94f fix(workflows): reject unsupported primitives in CLI validation
c587ae144 fix(workflows): enforce runtime capabilities in API and tools
bd96d1448 refactor(workflows): share assistant capability validation
484feb79d fix(workflows): enforce array and object result contracts
7fad4a824 fix(workflows): reject cyclic workflow graphs
03528c2d3 fix(workflows): make assistant repair attempts retry invalid drafts
da15314f2 fix(workflows): return actionable assistant errors
d8cdff598 fix(workflows): show assistant remediation hints in dashboard
5895db8c3 fix(workflows): cap dashboard workflow request bodies
7f941060a feat(workflows): add workflow payload redaction helper
609eceb00 fix(workflows): warn and redact sensitive dashboard payloads
2e7e0419c fix(workflows): improve dashboard cell editor accessibility
97fa40b66 docs(workflows): enforce contracts in workflow examples
8d79e59b4 docs(workflows): document contracts privacy and runtime limits
13bf5dc40 test(workflows): lock docs and examples alignment
b7396e6ed test(workflows): align engine cycle coverage with validation
```
