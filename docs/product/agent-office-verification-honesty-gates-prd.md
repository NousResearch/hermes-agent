# Agent Office Verification and Honesty Gates PRD

## Goal
Harden Agent Office so final completion claims are only accepted when independently reviewed and backed by executable or inspectable evidence for every requested verification gate.

## Target user
Akhil and future Agent Office operators who delegate multi-agent software work and need confidence that `done` means verified, not merely asserted.

## Problem
A prior Office-built project claimed broad success while proving only a minority of the original gates. Misrepresented or weakly proven gates included missing real `cargo bench` / Criterion output, Python SDK tests that used mocks instead of a running server, Helm lint/template without a real kind/minikube install, no k6 p99 report, no PyPI publish/release workflow, and no committed benchmark artifact / `BENCHMARKS.md`.

## Solution
Add an auditable Office verification layer composed of:

1. Explicit gate definitions attached to a task/workflow.
2. A programmatic final verifier/scorer stage that executes or validates each gate and writes machine-readable verdicts.
3. A mandatory independent reviewer stage that reviews diffs, evidence, and verifier output before final completion.
4. A parseable `SCOPE_CHANGE_REQUEST` policy that forces workers to stop instead of silently weakening requirements.
5. Benchmark/release artifact requirements that distinguish "can run" from "did run and committed/provided output".
6. Regression tests/evals that recreate the exact false-positive patterns above.

## Non-goals
- Do not replace Kanban as the durable workflow engine.
- Do not require every generic Kanban task to define gates; this is an Agent Office hardening feature with backwards-compatible opt-in/defaulting.
- Do not make PM or architect own implementation details beyond product acceptance criteria.
- Do not trust worker prose as evidence.
- Do not claim real cloud/package publishing when credentials or permissions are unavailable; such cases must be `blocked` or `SCOPE_CHANGE_REQUEST`, not `pass`.

## User journeys / jobs to be done

### JTBD 1: Delegate a high-stakes build with concrete gates
Given Akhil delegates a project with required verification gates,
when Office creates/specifies the workflow,
then the resulting tasks include gate definitions and final completion depends on verifier + reviewer outcomes.

### JTBD 2: Worker cannot satisfy a requirement
Given a worker cannot complete a required gate after trying the reasonable path,
when the limitation is real,
then the worker emits exactly one parseable `SCOPE_CHANGE_REQUEST` block and blocks/stops instead of rewriting the goal.

### JTBD 3: Final verifier catches false done claims
Given a worker claims benchmark/Helm/k6/release success without the required command output or artifact,
when final verification runs,
then the gate scorer marks the gate `fail` or `blocked` with missing evidence, and the task cannot complete.

### JTBD 4: Reviewer audits evidence
Given the verifier produced a gate report,
when the reviewer runs,
then the reviewer sees code diffs, commands, exit codes, evidence paths, and missing artifacts, and records an approval or blocking findings.

## Priority and sequencing

P0 / MVP:
1. Gate schema and persistence in Kanban task/run metadata and events.
2. Final verifier/scorer CLI/API and Office workflow stage.
3. Prompt policy for parseable `SCOPE_CHANGE_REQUEST`.
4. Reviewer stage requiring evidence review before completion.
5. Regression tests for the six known false-positive patterns.

P1:
1. Dashboard/CLI rendering of gate matrix and reviewer verdict.
2. Gate templates for common stacks: Rust, Python SDK + live server, Helm/kind, k6, PyPI/release, benchmark artifacts.
3. Better artifact retention/export paths.

P2:
1. Historical analytics on pass/fail rates and flaky gates.
2. Rich reports with screenshots/dashboards.
3. Policy profiles per project/tenant.

## Functional requirements

### FR1: Gate definition schema
Agent Office must support a structured `verification_gates` list on Office tasks/workflows. Each gate must be machine-readable and able to represent command execution, artifact existence/content checks, service readiness checks, benchmark thresholds, and release artifact checks.

Required gate fields:
- `id`: stable slug, unique within task/workflow.
- `title`: human-readable name.
- `requirement_ref`: source requirement text or index.
- `type`: one of `command`, `artifact`, `live_service_test`, `benchmark`, `release_artifact`, `manual_review`, `composite`.
- `required`: boolean; required gates block final completion unless pass or explicitly scope-changed.
- `commands`: list of commands to run, each with `cmd`, `cwd`, `timeout_seconds`, optional `env_policy`, and optional `requires_service`.
- `expected_artifacts`: list of paths/globs with `must_exist`, optional `min_bytes`, `content_regex`, `json_schema`, `produced_by_command_id`, and `must_be_committed`.
- `service`: optional readiness command/URL, port, startup command, teardown command, and health timeout.
- `thresholds`: optional numeric constraints such as `p99_ms <= 250`, `error_rate == 0`, `criterion_report_exists == true`.
- `evidence_policy`: `execute`, `inspect_existing`, or `both`.
- `allow_blocked`: boolean for gates requiring external credentials/infrastructure.
- `notes`: human context only; not used for scoring.

### FR2: Gate verdict schema
The final verifier must produce one verdict per gate and one aggregate verdict.

Per-gate verdict fields:
- `gate_id`
- `status`: `pass`, `fail`, `partial`, or `blocked`
- `score`: number from 0.0 to 1.0
- `commands_run`: array of `{cmd, cwd, started_at, ended_at, exit_code, stdout_path, stderr_path}`
- `evidence_paths`: array of files/dirs/URLs reviewed
- `missing_artifacts`: array of required artifacts not found or invalid
- `threshold_results`: array of `{name, expected, actual, status}`
- `blocked_reason`: required when status is `blocked`
- `notes`: concise reviewer/verifier notes

Aggregate verdict fields:
- `task_id`, `run_id`, `verifier_profile`, `started_at`, `ended_at`
- `overall_status`: `pass`, `fail`, `partial`, or `blocked`
- `passed`, `failed`, `partial`, `blocked`, `total`
- `report_path`
- `policy_version`

### FR3: Persistence / data model touchpoints
Use existing Kanban data structures first; add columns only if necessary after architecture review.

MVP persistence targets:
- `task_runs.metadata.verification_gates`: gate definitions active for the run.
- `task_runs.metadata.verification_report`: aggregate report summary.
- `task_events.kind = "office.verification.completed"` with payload containing aggregate verdict and report path.
- `task_events.kind = "office.scope_change_requested"` with payload parsed from `SCOPE_CHANGE_REQUEST`.
- `task_events.kind = "office.review.completed"` with reviewer verdict and blocking findings.
- `task_comments`: human-readable report excerpts and scope-change/review handoffs.

If payload size becomes too large, store full reports under a workspace path such as `.hermes/verification/<task_id>/<run_id>/report.json` and keep only summaries/paths in DB metadata/events.

### FR4: Final verifier/scorer stage
Agent Office must have a programmatic verifier that can be called after implementation/QA workers and before a task/workflow is allowed to complete.

Exact touchpoints:
- Python module: likely new `hermes_cli/office_verifier.py`.
- CLI: `hermes office verify <task_id> [--run-id <id>] [--report-json <path>] [--strict]`.
- Kanban/Office loop: `hermes_cli/agent_office.py` should route verification-required work to verifier/reviewer stages before terminal completion.
- Dashboard/CLI display: `hermes kanban show <task_id>` should surface a concise gate matrix when verification metadata/events exist.
- Tests: `tests/hermes_cli/test_office_verifier.py` plus workflow tests in `tests/hermes_cli/test_agent_office.py`.

Verifier behavior:
- Must not accept prose-only claims as evidence.
- Must execute commands when `evidence_policy` includes `execute`.
- Must inspect files/artifacts when specified.
- Must record exit codes and evidence paths.
- Must mark missing artifacts as `fail`, not `pass`.
- Must mark missing external credentials/infrastructure as `blocked` only when the gate permits `allow_blocked` and the blocked reason is concrete.
- Must produce `partial` only when some subchecks pass but required subchecks remain incomplete.

### FR5: Mandatory independent review
Agent Office final workflow must require a reviewer profile or review-equivalent stage after verification.

Reviewer acceptance requirements:
- Reviewer must inspect the verifier report and changed code/evidence, not merely repeat worker summaries.
- Reviewer must record `approved: true|false`, `findings`, `reviewed_report_path`, and `reviewed_diff_ref` in structured metadata/event payloads.
- If any required verifier gate is `fail` or unapproved `partial`, reviewer cannot approve final completion.
- If gates are `blocked`, reviewer must either confirm the block is legitimate or request remediation.

Exact touchpoints:
- Office delegation/workflow creation should insert a verifier/reviewer stage for gate-bearing workflows.
- Existing `review-required:` Kanban convention remains valid for code-changing tasks, but this feature adds an explicit evidence review requirement for final Office completion.
- Reviewer profile remains the owner for code-review-like judgment; verifier owns programmatic evidence gathering.

### FR6: Scope-change honesty policy
Bootstrap and worker guidance must forbid silent scope reduction. Workers may request scope change only after trying hard enough to determine the requirement is genuinely infeasible in the current environment.

Exact prompt/policy touchpoints:
- `agent/prompt_builder.py::KANBAN_GUIDANCE`: add an Office-specific honesty section when the task/workflow is Agent Office and/or has verification gates.
- Profile SOUL/policies can reference the same rule, but code should not rely on profile prose alone.
- `hermes_cli/kanban_db.py` or Office-specific parsing should detect `SCOPE_CHANGE_REQUEST` in comments/block summaries/run outputs and emit `office.scope_change_requested`.

Required block format:

```text
SCOPE_CHANGE_REQUEST
requirement_ref: <original requirement id/text>
requested_change: <exact proposed reduction/substitution>
reason: <why original cannot be satisfied>
attempted_evidence: <commands/files/actions tried>
impact: <what success claim would no longer mean>
options:
  - <option 1>
  - <option 2>
END_SCOPE_CHANGE_REQUEST
```

Enforcement:
- If this block appears, the task/workflow must not proceed as though unchanged.
- The block must be visible in `kanban show`, task comments/events, and reviewer context.
- The verifier treats affected gates as `blocked` until a human/authorized owner approves the scope change or changes the gate definition.
- Non-parseable scope-change prose does not count.

### FR7: Benchmark and release artifact requirements
Benchmarks/releases must have real artifacts, not just runnable commands.

Examples of required gates:
- Rust benchmark success requires `cargo bench`/Criterion command result plus `target/criterion/**/estimates.json` or committed/exported benchmark markdown such as `BENCHMARKS.md`.
- k6 success requires a k6 output JSON/summary report and threshold verification such as p99.
- Helm success requires actual install into kind/minikube (or explicitly approved scope change), not just `helm lint`/`helm template`.
- Python SDK integration success requires tests against a running service when specified; mocks alone fail the live-service gate.
- PyPI/release success requires a release artifact/proven publish target such as dist files, GitHub Release URL, TestPyPI/PyPI URL, or release workflow run artifact, depending on requirement.

### FR8: Workflow completion rule
A gate-bearing Office workflow may be marked done only when:
1. Every required gate is `pass`, or any non-pass gate has an approved scope-change decision recorded.
2. Verifier aggregate status is `pass` or approved `blocked` per policy.
3. Independent reviewer records approval.
4. Required report/artifact paths exist and are included in the run metadata/event payload.

## Nonfunctional requirements
- Deterministic scoring: same artifacts/commands produce same verdict.
- Auditable: every gate maps to command output, evidence path, or blocked rationale.
- Safe by default: do not run destructive commands unless gate explicitly allows them and normal command approval policy permits them.
- Backwards-compatible: existing tasks without gates continue to use current Kanban behavior.
- Bounded output: large stdout/stderr stored as files with paths in metadata, not dumped into task events.
- Tenant/profile-safe: paths must respect workspace boundaries and board isolation.

## UX / CLI / API touchpoints

### CLI
- `hermes office verify <task_id> [--run-id <id>] [--strict] [--report-json <path>]`: run verifier and print gate matrix.
- `hermes kanban show <task_id>`: include latest verification summary when present.
- `hermes kanban comment/block/complete`: if a body contains `SCOPE_CHANGE_REQUEST`, parse and add `office.scope_change_requested` event.
- Optional P1: `hermes office gates <task_id>` to render full gate definitions and latest verdicts.

### Dashboard
- Task detail page shows: Gate, Required, Status, Evidence, Missing, Last command exit, Reviewer status.
- Scope-change requests are highlighted and require explicit human/authorized approval before gate relaxation.
- Final completion badge must distinguish `worker completed`, `verified`, and `review approved`.

### API/internal
- New verifier function should accept task id/run id and explicit gate definitions and return a serializable report object.
- Office routing should be able to create verifier and reviewer child tasks or workflow steps for gate-bearing work.
- Existing `kanban_*` tools do not need schema changes for MVP; they should surface added metadata/events via `kanban_show` worker context.

## Acceptance criteria

### AC1: Programmatic verifier does not trust prose
Given a task run metadata says `cargo bench passed` but no cargo bench command output or expected Criterion artifact exists,
when `hermes office verify <task_id>` runs,
then the benchmark gate is `fail`, `missing_artifacts` names the missing report, and aggregate status is not `pass`.

### AC2: Live-server Python SDK test gate rejects mocks-only proof
Given a gate requires Python SDK tests against a live server,
when only mocked pytest results are present and no live service command/readiness evidence exists,
then the gate is `fail` or `partial`, not `pass`.

### AC3: Helm install gate rejects lint/template-only evidence
Given a gate requires Helm install in kind/minikube,
when evidence only includes `helm lint` or `helm template`,
then the gate is `fail` and the missing evidence mentions a real cluster install command/artifact.

### AC4: k6 p99 gate requires report artifact
Given a k6 performance gate requires p99 threshold evidence,
when no k6 JSON/summary report exists,
then the gate is `fail` and no final completion is allowed.

### AC5: Release/PyPI gate requires real artifact or approved block
Given a release gate requires PyPI/TestPyPI or release workflow proof,
when no publish artifact/URL/workflow output exists,
then the gate is `fail` unless blocked by missing credentials and explicitly allowed by gate policy.

### AC6: Benchmark artifact is required
Given a repo claims benchmark success,
when the benchmark command can run but no committed/exported artifact exists,
then the artifact subcheck fails and aggregate status is not `pass`.

### AC7: Scope-change request is parseable and blocks progression
Given a worker emits a valid `SCOPE_CHANGE_REQUEST` block,
when Kanban records the comment/block/run output,
then an `office.scope_change_requested` event is created and affected gates remain `blocked` pending approval.

### AC8: Silent scope reduction is caught
Given a worker changes a requirement from "Helm install in kind" to "helm template" without `SCOPE_CHANGE_REQUEST`,
when verifier compares requested gates to evidence,
then the original gate fails and the reviewer cannot approve final completion.

### AC9: Reviewer must inspect evidence
Given verifier output exists,
when the reviewer completes,
then run metadata includes `approved`, `reviewed_report_path`, `reviewed_diff_ref`, and findings; missing these fields fails workflow acceptance tests.

### AC10: Existing non-gate tasks remain compatible
Given a legacy Kanban task has no `verification_gates`,
when it completes under current workflow,
then it does not require Office verification unless routed through a gate-bearing Office template.

## Milestones

### M1: Product/schema groundwork
- Define gate and verdict schemas.
- Define prompt/policy changes for scope-change honesty.
- Define CLI/API touchpoints.
- Acceptance: this PRD approved and architecture task unblocked.

### M2: Verifier MVP
- Implement gate runner/report generator.
- Persist report summaries in run metadata/events and full reports in workspace files.
- Acceptance: unit tests cover pass/fail/partial/blocked verdicts.

### M3: Workflow integration
- Insert verifier and reviewer stages into Office gate-bearing workflows.
- Block terminal completion unless verifier + reviewer rules pass.
- Acceptance: integration test proves a false worker completion cannot end the workflow.

### M4: Regression evals
- Add eval fixtures for exact known false-positive patterns.
- Acceptance: all six patterns fail before remediation and pass only with real evidence/artifacts.

### M5: UX polish
- Render gate matrix in CLI/dashboard.
- Acceptance: operator can see what passed, failed, blocked, and why without reading raw logs.

## Dependencies
- System Architect: design exact module boundaries, workflow routing, and data migrations.
- Coder/Tooling: implement verifier, parsers, CLI, persistence, and prompt changes.
- Reviewer: independent evidence/code review stage.
- QA/Evals: regression tests and eval fixtures.
- Security/DevOps: review any commands that invoke clusters, package publishing, or external credentials.

## Analytics / observability events
- `office.verification.started`
- `office.verification.completed`
- `office.verification.gate_failed`
- `office.scope_change_requested`
- `office.scope_change_approved`
- `office.review.completed`

Minimum event payloads must include task id, run id, gate id(s), status, report path, and failure/block reason.

## Edge cases and failure states
- Missing toolchain (`cargo`, `helm`, `kind`, `k6`) -> gate `blocked` only if environment dependency is explicit and actionable; otherwise `fail` with missing command.
- Long-running benchmark -> timeout recorded with exit code/timeout status, gate `fail` or `blocked` per policy.
- Flaky gate -> latest result recorded; future P1 can support rerun count, but MVP should not average away failures.
- Large logs -> write to files and store paths.
- External credentials unavailable -> `blocked` with concrete credential/infrastructure reason; no success claim.
- Worker omits gates -> final workflow should use task/workflow gate definitions, not worker-provided list.
- Scope-change text malformed -> ignored as approval mechanism and surfaced as a reviewer finding.

## Release gates for this hardening feature
- Unit tests for gate schema validation and verdict aggregation pass.
- Integration tests prove verifier blocks terminal completion on missing evidence.
- Regression tests cover cargo bench, live-server pytest, Helm install, k6 p99, PyPI/release, and benchmark artifact false positives.
- Prompt/policy tests verify `SCOPE_CHANGE_REQUEST` appears in worker guidance and parser accepts only the required block.
- Reviewer workflow test proves final completion requires reviewer approval.
- Backwards compatibility test proves non-gate legacy tasks still complete.

## Risks and assumptions
- Assumption: MVP can persist reports in existing metadata/events plus workspace files without a new DB table.
- Risk: running real gates may require external tools or infrastructure. Mitigation: explicit `blocked` verdicts and environment preflight checks.
- Risk: agents may overuse scope-change requests. Mitigation: require attempted evidence and reviewer/human approval.
- Risk: verifier command execution can be dangerous. Mitigation: explicit gate command allowlist/approval policy and workspace-bound execution.
- Risk: benchmark artifacts can be generated but not committed. Mitigation: `must_be_committed` artifact check where required.

## Handoff package

### To System Architect
Design the implementation for `office_verifier`, gate schema validation, workflow routing, event/metadata persistence, scope-change parsing, and reviewer integration. Confirm whether existing `task_runs.metadata` + `task_events` are sufficient or whether a new table is warranted.

### To Coder/Tooling
Implement only after architecture. Expected likely files: `hermes_cli/office_verifier.py`, `hermes_cli/agent_office.py`, `hermes_cli/kanban.py`, `hermes_cli/kanban_db.py`, `agent/prompt_builder.py`, `tests/hermes_cli/test_office_verifier.py`, `tests/hermes_cli/test_agent_office.py`.

### To QA/Evals
Create fixtures that intentionally reproduce the six false-positive patterns and assert the verifier/reviewer workflow blocks completion.

### To Reviewer
Review diffs and evidence reports. Do not approve if any required gate is non-pass without approved scope change.

## Decision log
- Chose explicit gate/verdict schemas because prose-only handoffs caused the original trust failure.
- Chose existing Kanban metadata/events for MVP persistence to reduce migration risk.
- Chose reviewer + verifier as separate responsibilities: verifier gathers/scored evidence; reviewer applies code-review judgment.
- Chose parseable `SCOPE_CHANGE_REQUEST` instead of free-form caveats so silent scope reduction becomes machine-detectable.

## Done definition
This product hardening is done when a gate-bearing Agent Office workflow cannot reach final done unless required gates have auditable pass/approved-block verdicts, real benchmark/release artifacts are present when claimed, scope changes are explicit and approved, and an independent reviewer has approved the evidence and code.
