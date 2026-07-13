# Hermes OS Task Backlog

Generated: 2026-06-15

This backlog continues the Hermes OS v3 task series after `task-113`. It focuses on making the newly native Architect/Plan CLI, dashboard, template loading, SQLite persistence, scheduled reviews, and runtime policies production-ready.

## Phase 35 - Native Command Hardening

- `task-114`: Add end-to-end CLI tests for `hermes architect review` through the installed console entrypoint.
- `task-115`: Add end-to-end CLI tests for `hermes plan` through the installed console entrypoint.
- `task-116`: Add help text and docs examples for `hermes architect review --persist --db`.
- `task-117`: Add help text and docs examples for `hermes plan --template --persist --db`.
- `task-118`: Add command error envelopes for invalid projects, invalid templates, and persistence failures.

## Phase 36 - Task Generation Engine

- `task-119`: Add a `TaskDefinition` schema with id, title, phase, dependencies, acceptance criteria, risk, and status.
- `task-120`: Add an architecture-review-to-task generator that converts missing docs, schemas, dashboards, and approvals into task definitions.
- `task-121`: Add a work-graph-to-task generator that converts blocked nodes and findings into implementation tasks.
- `task-122`: Add stable task id allocation that continues from the highest existing `task-NNN`.
- `task-123`: Add task artifact writers for `TASKS.md` and `.hermes/tasks.json`.

## Phase 37 - Dashboard Task UI

- `task-124`: Add task summary data to `/api/hermes-os/summary`.
- `task-125`: Add a task backlog panel to the Hermes OS dashboard page.
- `task-126`: Add blocked task and approval-required task filters.
- `task-127`: Add task dependency visualization for work graph-derived tasks.
- `task-128`: Add dashboard actions to regenerate tasks in dry-run mode.

## Phase 38 - Persistence Migrations

- `task-129`: Add SQLite schema versioning for Hermes OS records.
- `task-130`: Add migration runner for future Hermes OS persistence changes.
- `task-131`: Add import from local JSON records into SQLite.
- `task-132`: Add export from SQLite records into portable JSON bundles.
- `task-133`: Add repository integrity checks for malformed payloads and duplicate ids.

## Phase 39 - Scheduled Review Operations

- `task-134`: Add scheduled review job registration through `hermes cron`.
- `task-135`: Add scheduled review dry-run preview output.
- `task-136`: Persist scheduled review run summaries and score deltas.
- `task-137`: Add approval gates for scheduled review writes.
- `task-138`: Add dashboard status for last review, next review, and review failures.

## Phase 40 - Runtime Policy Enforcement

- `task-139`: Wire runtime policy decisions into delegation execution before launching a worker.
- `task-140`: Persist runtime policy audit records.
- `task-141`: Add retry backoff policy and retry exhaustion reporting.
- `task-142`: Add cost budget aggregation per project and per work graph.
- `task-143`: Add approval prompts for policy-blocked high-risk runtime actions.

## Phase 41 - Template Registry

- `task-144`: Add a template registry location under `.hermes/templates`.
- `task-145`: Add template discovery from project, workspace, and user template paths.
- `task-146`: Add template validation diagnostics with line/file context where available.
- `task-147`: Add template version metadata and compatibility checks.
- `task-148`: Add dashboard panels for loaded templates and compile failures.

## Phase 42 - Execution Readiness

- `task-149`: Add a graph execution planner that emits executable batches from dependencies.
- `task-150`: Add dry-run execution reports showing commands, policies, approvals, and expected artifacts.
- `task-151`: Add artifact ingestion from completed runtime tasks into SQLite persistence.
- `task-152`: Add validation result updates after artifact ingestion.
- `task-153`: Add rollout tests that run Architect -> Plan -> Tasks -> Dry-run Execution end to end.

## Phase 43 - Workspace & Project Runtime MVP

- `task-154`: Add project definition schema for `.hermes/projects/<project>/project.yaml`.
- `task-155`: Add workspace project registry loader and validator.
- `task-156`: Add `hermes projects` status output across registered projects.
- `task-157`: Add `hermes switch <project>` command contract and dry-run implementation.
- `task-158`: Add project memory file scaffold for architecture, decisions, progress, experiments, lessons, backlog, and agents.
- `task-159`: Add project memory loader for switch/status context.
- `task-160`: Add project-scoped task registry summary.
- `task-161`: Add project status CLI output for memory, tasks, dashboards, agents, and infrastructure.

## Phase 44 - Workspace Snapshot & Restore

- `task-162`: Add workspace snapshot schema for open files, browser URLs, terminals, services, branch, and open tasks.
- `task-163`: Add `hermes snapshot save` project workspace snapshot command.
- `task-164`: Add `hermes snapshot restore` project workspace restore command.
- `task-165`: Add VS Code workspace restore contract.
- `task-166`: Add browser URL restore contract.
- `task-167`: Add running service restore contract.

## Phase 45 - Project Runtime Manager

- `task-168`: Add runtime service definition schema for project startup commands.
- `task-169`: Add `hermes start <project>` runtime startup command.
- `task-170`: Add runtime process status tracking.
- `task-171`: Add runtime failure and partial-start reporting.
- `task-172`: Add runtime dashboard URL opening contract.
- `task-173`: Add cost and service health hooks for project runtime status.

## Phase 46 - Agent Messaging & Trace Visibility

- `task-174`: Add project-scoped agent registry from project definitions.
- `task-175`: Add project agent message bus record schema.
- `task-176`: Add agent trace record schema with sender, receiver, timestamp, type, content, and correlation ID.
- `task-177`: Add agent trace viewer data contract for dashboard timelines.
- `task-178`: Add agent health summary for dashboard modules.
- `task-179`: Add tests for project-scoped agent messaging isolation.

## Phase 47 - Infrastructure Registry & Unified Dashboard

- `task-180`: Add infrastructure registry schema for project-owned external systems.
- `task-181`: Add vector database registry fields per project.
- `task-182`: Add unified dashboard modules for projects, agent health, costs, experiments, tasks, alerts, infrastructure, queues, and activity feed.
- `task-183`: Add end-to-end workspace runtime MVP test for project switch status within 30 seconds.

## Phase 48 - Real Command Surface Completion

- `task-184`: Audit installed Hermes OS command routing for architect, plan, workspace, projects, switch, start, and snapshot.
- `task-185`: Normalize JSON output envelopes across Hermes OS native commands.
- `task-186`: Normalize non-JSON human output across Hermes OS native commands.
- `task-187`: Add shared command error codes for missing project, invalid registry, unsafe action, and persistence failure.
- `task-188`: Add command-level smoke tests for `hermes workspace` through the installed entrypoint.
- `task-189`: Add command-level smoke tests for `hermes projects` through the installed entrypoint.
- `task-190`: Add command-level smoke tests for `hermes switch` through the installed entrypoint.
- `task-191`: Add command-level smoke tests for `hermes start` through the installed entrypoint.
- `task-192`: Add command-level smoke tests for `hermes snapshot` through the installed entrypoint.
- `task-193`: Add command docs for architecture-to-runtime operating flow.
- `task-194`: Add shell-completion metadata for Hermes OS command arguments.
- `task-195`: Add backward-compatible module CLI redirects for old Hermes OS invocation paths.

## Phase 49 - Guarded Live Runtime Execution

- `task-196`: Add live runtime enablement flag to project definitions.
- `task-197`: Add per-project runtime budget fields for cost and token ceilings.
- `task-198`: Add runtime command allowlist validation before launching services or workers.
- `task-199`: Add human approval requirement for write-capable live runtime actions.
- `task-200`: Add live worker launch adapter around the Official Hermes Agent wrapper.
- `task-201`: Add runtime timeout enforcement for live worker execution.
- `task-202`: Add retry classification for transient runtime failures.
- `task-203`: Add retry exhaustion records for failed live runtime actions.
- `task-204`: Add runtime artifact quarantine before validation and ingestion.
- `task-205`: Add rollback report generation for failed live runtime operations.
- `task-206`: Add live runtime dry-run parity tests.
- `task-207`: Add live runtime opt-in integration test with safe no-op worker command.

## Phase 50 - Workspace Restore Integrations

- `task-208`: Add platform detection for workspace restore launchers.
- `task-209`: Add VS Code workspace reopen implementation behind dry-run/live mode.
- `task-210`: Add editor fallback contract for non-VS Code environments.
- `task-211`: Add browser URL reopen implementation behind dry-run/live mode.
- `task-212`: Add terminal session restore plan for captured terminal commands.
- `task-213`: Add service restart plan for project runtime services.
- `task-214`: Add git branch restore validation before checkout actions.
- `task-215`: Add active task context restore into project memory and dashboard state.
- `task-216`: Add partial restore result schema with skipped, failed, and completed steps.
- `task-217`: Add restore conflict detection for dirty worktrees and running services.
- `task-218`: Add restore approval gate for branch changes and service restarts.
- `task-219`: Add cross-platform restore tests for dry-run contracts.

## Phase 51 - Runtime Dashboard UI

- `task-220`: Add Hermes OS project list panel to the dashboard UI.
- `task-221`: Add project runtime service health panel to the dashboard UI.
- `task-222`: Add workspace snapshot history panel to the dashboard UI.
- `task-223`: Add snapshot restore preview panel to the dashboard UI.
- `task-224`: Add agent trace timeline panel to the dashboard UI.
- `task-225`: Add agent message detail drawer to the dashboard UI.
- `task-226`: Add runtime cost and budget panel to the dashboard UI.
- `task-227`: Add approval queue panel for blocked runtime actions.
- `task-228`: Add infrastructure registry panel to the dashboard UI.
- `task-229`: Add vector database registry panel to the dashboard UI.
- `task-230`: Add task backlog and dependency panel refresh from project runtime state.
- `task-231`: Add dashboard empty, loading, and error states for Hermes OS panels.

## Phase 52 - Durable Project Runtime Persistence

- `task-232`: Add SQLite schema for project definitions and registry cache.
- `task-233`: Add SQLite schema for workspace snapshots.
- `task-234`: Add SQLite schema for snapshot restore attempts.
- `task-235`: Add SQLite schema for project runtime service status.
- `task-236`: Add SQLite schema for agent messages and traces.
- `task-237`: Add SQLite schema for runtime approvals and decisions.
- `task-238`: Add SQLite schema for runtime cost records.
- `task-239`: Add SQLite schema for infrastructure and vector registry records.
- `task-240`: Add persistence repository methods for project runtime reads and writes.
- `task-241`: Add JSON import and export for project runtime persistence records.
- `task-242`: Add integrity checks for orphaned traces, snapshots, approvals, and runtime statuses.
- `task-243`: Add migration tests for project runtime persistence schemas.

## Phase 53 - External Template Packs

- `task-244`: Define external template pack manifest schema.
- `task-245`: Add template pack discovery from workspace and user-level directories.
- `task-246`: Add template pack compatibility checks against Hermes OS schema version.
- `task-247`: Add template pack dependency declaration support.
- `task-248`: Add template pack dry-run install command contract.
- `task-249`: Add template pack live install command with approval gate.
- `task-250`: Add template pack update dry-run and diff output.
- `task-251`: Add template pack uninstall safety checks.
- `task-252`: Add template pack validation diagnostics with file and line context.
- `task-253`: Add template pack dashboard visibility for installed and failing packs.
- `task-254`: Add template pack tests with valid, invalid, incompatible, and missing dependency cases.
- `task-255`: Document template pack authoring conventions and boundary rules.

## Phase 54 - Continuous Workspace Operations

- `task-256`: Add recurring workspace health check scheduler contract.
- `task-257`: Add architecture drift detection against latest project docs and decisions.
- `task-258`: Add stale task detection using task age, blockers, and dependency status.
- `task-259`: Add stale snapshot detection for projects with outdated workspace state.
- `task-260`: Add runtime service drift detection against project startup definitions.
- `task-261`: Add approval queue aging and escalation summary.
- `task-262`: Add cost budget drift detection across runtime records.
- `task-263`: Add agent failure-rate monitoring across project traces.
- `task-264`: Add infrastructure availability check contracts for project-owned systems.
- `task-265`: Add workspace health score calculation and trend history.
- `task-266`: Add dashboard activity feed records for continuous operations.
- `task-267`: Add end-to-end dry-run test for scheduled workspace operations.

## Phase 55 - Production Live Runtime

- `task-268`: Add live runtime execution state machine for queued, running, validating, completed, failed, canceled, and rolled back states.
- `task-269`: Add live runtime process supervisor with pid, exit code, stdout, stderr, and duration capture.
- `task-270`: Add live runtime cancellation command and cancellation audit record.
- `task-271`: Add live runtime resume command for interrupted executions with prior context.
- `task-272`: Add live runtime log tailing contract for dashboard and CLI consumers.
- `task-273`: Add live runtime artifact manifest schema with checksums and validation status.
- `task-274`: Add live runtime validation gate before artifact ingestion into persistence.
- `task-275`: Add live runtime rollback executor for reversible project-runtime actions.
- `task-276`: Add live runtime sandbox profile selection per project and action type.
- `task-277`: Add live runtime environment variable allowlist and redaction policy.
- `task-278`: Add live runtime execution history query API.
- `task-279`: Add production live-runtime integration test using a safe local no-op worker.

## Phase 56 - Approval UX & Governance

- `task-280`: Add approval request schema with requester, reviewer, scope, risk, expiry, and decision fields.
- `task-281`: Add approval queue persistence methods and list/get/update APIs.
- `task-282`: Add `hermes approvals list` command contract.
- `task-283`: Add `hermes approvals approve` command contract with required reason capture.
- `task-284`: Add `hermes approvals reject` command contract with required reason capture.
- `task-285`: Add approval expiry worker contract for stale pending approvals.
- `task-286`: Add policy override record schema for exceptional runtime actions.
- `task-287`: Add approval dashboard actions for approve, reject, and request-more-context flows.
- `task-288`: Add approval notification event records for dashboard activity feed.
- `task-289`: Add approval risk scoring from action type, project, cost, and target resources.
- `task-290`: Add approval audit export for compliance review.
- `task-291`: Add governance tests for approval expiry, override, and audit immutability.

## Phase 57 - Workspace Runtime Automation

- `task-292`: Add project automation workflow schema for switch, restore, start, validate, and notify steps.
- `task-293`: Add `hermes project run <workflow>` command contract.
- `task-294`: Add project switch automation that chains memory load, snapshot restore, service start, and dashboard open.
- `task-295`: Add project shutdown automation for services, snapshots, and final status records.
- `task-296`: Add automation preflight checks for dirty worktree, unavailable tools, blocked approvals, and missing config.
- `task-297`: Add automation step dependency model with skip, retry, and fallback behavior.
- `task-298`: Add automation dry-run diff between current workspace state and target project state.
- `task-299`: Add automation execution history and replay contract.
- `task-300`: Add automation dashboard panel for active and recent project workflows.
- `task-301`: Add automation schedule integration for recurring project warmup flows.
- `task-302`: Add automation failure report with next-action recommendations.
- `task-303`: Add end-to-end dry-run automation test for switch -> restore -> start.

## Phase 58 - Multi-Project Orchestration

- `task-304`: Add cross-project dependency schema with source project, target project, reason, and status.
- `task-305`: Add workspace-level dependency resolver across registered projects.
- `task-306`: Add cross-project blocker registry and blocker aging fields.
- `task-307`: Add shared approval request support for actions spanning multiple projects.
- `task-308`: Add multi-project execution queue planner with per-project isolation.
- `task-309`: Add queue balancing policy for project priority, risk, and readiness.
- `task-310`: Add shared infrastructure dependency mapping across projects.
- `task-311`: Add multi-project dashboard graph panel contract.
- `task-312`: Add cross-project stale dependency detection.
- `task-313`: Add orchestration report export for workspace review.
- `task-314`: Add conflict detection for simultaneous actions against shared resources.
- `task-315`: Add multi-project orchestration tests with dependency and blocker scenarios.

## Phase 59 - Agent Fleet Management

- `task-316`: Add agent fleet registry schema with capability, model, tools, cost, and availability fields.
- `task-317`: Add agent heartbeat record schema and stale-heartbeat detection.
- `task-318`: Add agent capability matching score for task routing.
- `task-319`: Add agent health score from success rate, failure rate, latency, and retry count.
- `task-320`: Add agent cost score from token and runtime usage records.
- `task-321`: Add agent latency score from trace and runtime records.
- `task-322`: Add agent quarantine policy for repeated failures or policy violations.
- `task-323`: Add agent fleet dashboard panels for health, cost, latency, and routing confidence.
- `task-324`: Add agent routing explanation record for every assignment.
- `task-325`: Add agent fallback selection when preferred specialist is unavailable.
- `task-326`: Add agent fleet export for operations review.
- `task-327`: Add fleet routing tests for healthy, degraded, unavailable, and quarantined agents.

## Phase 60 - Observability & Telemetry

- `task-328`: Add Hermes OS event envelope schema with project, phase, severity, source, and correlation id.
- `task-329`: Add structured event writer for command, runtime, approval, restore, and dashboard events.
- `task-330`: Add trace correlation across command invocation, runtime action, agent message, and artifact ingestion.
- `task-331`: Add metric rollup records for daily project health, cost, runtime, and approval activity.
- `task-332`: Add telemetry redaction policy for prompts, env vars, URLs, and credentials.
- `task-333`: Add observability dashboard trend panels for health, cost, approvals, runtime failures, and agent routing.
- `task-334`: Add diagnostics bundle export for a project runtime incident.
- `task-335`: Add event retention and pruning policy.
- `task-336`: Add telemetry import/export compatibility checks.
- `task-337`: Add slow command and slow runtime action detection.
- `task-338`: Add telemetry integrity checks for missing correlation chains.
- `task-339`: Add observability tests for event redaction, rollups, exports, and correlation.

## Phase 61 - Plugin & Connector Boundary

- `task-340`: Add connector manifest schema with permissions, resources, commands, and risk profile.
- `task-341`: Add connector registry discovery from project, workspace, and user paths.
- `task-342`: Add connector permission evaluator for read, write, deploy, purchase, and destructive actions.
- `task-343`: Add connector dry-run contract for external writes.
- `task-344`: Add connector audit record schema with external target and normalized action type.
- `task-345`: Add connector output normalization contract for dashboard and persistence ingestion.
- `task-346`: Add connector health check contract and status dashboard panel.
- `task-347`: Add connector secret reference policy that avoids storing raw secrets.
- `task-348`: Add connector install/update/remove lifecycle contracts.
- `task-349`: Add connector compatibility checks against Hermes OS and project definitions.
- `task-350`: Add connector sandbox tests with fake external systems.
- `task-351`: Document connector boundary rules for domain systems and source-of-truth ownership.

## Phase 62 - Evaluation & Quality Gates

- `task-352`: Add evaluation schema for architecture reviews, work graphs, templates, runtime artifacts, and dashboard contracts.
- `task-353`: Add deterministic validation checks for required fields, dependencies, approvals, and traceability.
- `task-354`: Add rubric-based evaluation contract for quality-sensitive generated documents.
- `task-355`: Add self-review/reflection workflow for generated specs and implementation plans.
- `task-356`: Add evaluation evidence persistence for pass, fail, warning, and waived outcomes.
- `task-357`: Add quality gate runner that blocks execution when required evaluations fail.
- `task-358`: Add human waiver record for failed gates with reason and expiry.
- `task-359`: Add dashboard panel for gate status and recent evaluation failures.
- `task-360`: Add regression evaluation set for Hermes OS planning outputs.
- `task-361`: Add cost-aware evaluation routing for cheap deterministic checks before expensive model review.
- `task-362`: Add evaluation export for release and compliance review.
- `task-363`: Add quality gate tests for pass, fail, warning, waiver, and blocked execution paths.

## Phase 63 - Project Memory Intelligence

- `task-364`: Add project memory index schema with source path, record id, timestamp, and summary fields.
- `task-365`: Add decision retrieval API scoped by project, topic, recency, and confidence.
- `task-366`: Add lessons-learned extraction from completed tasks, failures, and approvals.
- `task-367`: Add project memory summarizer for architecture, decisions, progress, experiments, backlog, and agents.
- `task-368`: Add memory traceability links from summary bullets back to source artifacts.
- `task-369`: Add memory drift detector for stale decisions and conflicting project documents.
- `task-370`: Add memory refresh scheduler contract for long-running projects.
- `task-371`: Add memory query dashboard panel for decisions, lessons, and drift warnings.
- `task-372`: Add memory export/import bundle for project handoff.
- `task-373`: Add guardrail that prevents agent runtime memory from overwriting source-of-truth memory records.
- `task-374`: Add project memory compaction policy for old records and summaries.
- `task-375`: Add memory intelligence tests for retrieval, summarization, drift, and source traceability.

## Phase 64 - Release Hardening

- `task-376`: Add full Hermes OS release checklist covering CLI, dashboard, runtime, persistence, templates, connectors, and docs.
- `task-377`: Add migration compatibility tests from task registry versions 183, 267, and current.
- `task-378`: Add backward compatibility tests for legacy module CLI invocations.
- `task-379`: Add dashboard build verification for Hermes OS runtime pages.
- `task-380`: Add packaging verification for new Hermes OS modules and subcommands.
- `task-381`: Add documentation pass for command examples, safety policies, restore flows, and live runtime rollout.
- `task-382`: Add failure drill for unavailable runtime worker.
- `task-383`: Add failure drill for corrupted project registry.
- `task-384`: Add failure drill for interrupted live execution.
- `task-385`: Add failure drill for failed migration or partial persistence write.
- `task-386`: Add release notes generator from completed Hermes OS task records.
- `task-387`: Add full Hermes OS integration suite target for CI or local release verification.

## Phase 65 - Conversational Operating Layer Foundation

- `task-388`: Define Conversational Operating Layer contracts and ownership boundaries.
- `task-389`: Add conversational session schema for user, project, goal, initiative, and transcript metadata.
- `task-390`: Add chat request and response envelope contracts for CLI, API, and UI consumers.
- `task-391`: Add persisted conversation transcript model with source-of-truth references.
- `task-392`: Add COL configuration and feature flags for staged rollout.
- `task-393`: Document COL safety boundaries for commands, agents, workflows, and artifacts.
- `task-394`: Add Chief of Staff context packing contract for conversational turns.
- `task-395`: Add COL audit event records for intent, routing, delegation, approval, and artifact generation.
- `task-396`: Add command routing discovery for chat-capable Hermes OS commands.
- `task-397`: Add foundation tests for COL schemas, config, and audit records.

## Phase 66 - Chat CLI Surface

- `task-398`: Add `hermes chat` command contract and non-executing prototype.
- `task-399`: Add `hermes ask` one-shot command for quoted natural-language requests.
- `task-400`: Add interactive default `hermes` shell mode behind a feature flag.
- `task-401`: Add slash-command parser for conversational shortcuts such as `/grill-me` and `/plan`.
- `task-402`: Add streaming status update contract for long-running conversational workflows.
- `task-403`: Add conversation transcript save and load behavior for CLI sessions.
- `task-404`: Add project switch support from conversational CLI requests.
- `task-405`: Add structured chat command error envelopes for unknown intent, unsafe action, and missing project.
- `task-406`: Add CLI help text and docs examples for chat-first workflows.
- `task-407`: Add CLI smoke tests for `hermes chat`, `hermes ask`, and interactive startup.

## Phase 67 - Chief of Staff Agent

- `task-408`: Add Chief of Staff role schema with responsibilities, prohibitions, and output contracts.
- `task-409`: Add orchestration policy that prevents Chief of Staff from directly performing worker tasks.
- `task-410`: Add Chief of Staff decision loop for understand, route, plan, delegate, track, and report.
- `task-411`: Add action plan output schema with selected workflow, agents, approvals, and expected artifacts.
- `task-412`: Add delegation record schema linking user intent to agents, skills, workflows, and artifacts.
- `task-413`: Add conversational status reporter for active workflow progress and blocked steps.
- `task-414`: Add recommendation generator for next best project, task, review, or research action.
- `task-415`: Add clarification and approval prompt contracts for ambiguous or high-risk requests.
- `task-416`: Add Chief of Staff memory handoff into project memory and session memory.
- `task-417`: Add Chief of Staff tests for routing, delegation, status, and direct-work prohibition.

## Phase 68 - Intent Routing Engine

- `task-418`: Add intent schema with confidence, alternatives, required context, and selected route.
- `task-419`: Add deterministic phrase and command-pattern classifier for common Hermes OS intents.
- `task-420`: Add model-assisted intent fallback contract for uncertain requests.
- `task-421`: Add new-project route for build, create, start, and launch requests.
- `task-422`: Add existing-project route for continue, update, fix, and improve requests.
- `task-423`: Add research route for research, analyze, compare, and investigate requests.
- `task-424`: Add architecture route for review architecture, refactor system, and design workflow requests.
- `task-425`: Add task and review route for implement task, review task, approve, and reject requests.
- `task-426`: Add ambiguity handling with clarification questions and safe default routes.
- `task-427`: Add intent routing evaluation set with representative conversational examples.

## Phase 69 - Conversational Workflow Engine

- `task-428`: Add conversational workflow registry for named workflows and route bindings.
- `task-429`: Add workflow state machine for planned, running, waiting, delegated, validating, completed, failed, and canceled states.
- `task-430`: Add workflow checkpoint persistence for each conversational step and artifact boundary.
- `task-431`: Add new-project workflow chain for grill-me, PRD, architecture, plan, tasks, agents, tracker, dashboard, and workspace open.
- `task-432`: Add existing-project workflow chain for context load, route selection, task planning, execution, and review.
- `task-433`: Add research workflow chain for research agents, evidence collection, synthesis, and report artifact generation.
- `task-434`: Add architecture workflow chain for architect review, refactor planning, and work graph generation.
- `task-435`: Add task-work workflow chain for task selection, execution delegation, validation, and review handoff.
- `task-436`: Add workflow dry-run preview showing steps, agents, skills, approvals, and artifacts before execution.
- `task-437`: Add workflow resume, cancel, and failure-recovery tests.

## Phase 70 - Project And Session Memory Layer

- `task-438`: Add project identity memory contract for selected project and workspace path.
- `task-439`: Add core project memory loader for project, architecture, decisions, progress, tracker, backlog, and agents documents.
- `task-440`: Add working context builder that condenses project memory for Chief of Staff and delegated agents.
- `task-441`: Add decision and progress retrieval scoped by topic, recency, and confidence.
- `task-442`: Add rolling conversation memory for recent user decisions, preferences, and unresolved questions.
- `task-443`: Add active project, active goal, and active initiative session store.
- `task-444`: Add memory freshness and drift warnings during conversational context loading.
- `task-445`: Add memory redaction and source-link policy for context passed to agents.
- `task-446`: Add agent context package builder with project memory, task context, and workflow checkpoint data.
- `task-447`: Add memory-layer tests for loading, summarization, redaction, source links, and drift warnings.

## Phase 71 - Agent Hierarchy And Delegation

- `task-448`: Add conversational agent role registry for management and worker layers.
- `task-449`: Add management-layer agent definitions for Planner, Architect, Research Lead, Engineering Lead, and Product Lead.
- `task-450`: Add worker-layer agent definitions for Engineer, Reviewer, Researcher, Analyst, QA, and Writer.
- `task-451`: Add delegation protocol schema for assignment, input artifacts, output artifacts, and completion evidence.
- `task-452`: Add agent assignment planner using intent, workflow step, capability, cost, availability, and risk.
- `task-453`: Add artifact handoff validation between agents and workflow steps.
- `task-454`: Add multi-agent trace timeline records for conversational workflows.
- `task-455`: Add reviewer approval flow for delegated artifacts before promotion to source-of-truth state.
- `task-456`: Add agent failure fallback and escalation behavior for unavailable, low-confidence, or policy-blocked agents.
- `task-457`: Add delegation tests for single-agent, chained-agent, parallel-agent, review, and fallback scenarios.

## Phase 72 - Hermes Chat UI And Dashboard

- `task-458`: Add Chief of Staff API endpoints for chat turns, workflow status, recommendations, and active context.
- `task-459`: Add Open WebUI integration contract for chat requests, session identity, and streamed responses.
- `task-460`: Add Hermes chat page with persistent conversation and workflow status.
- `task-461`: Add project switcher UI tied to session memory and project context loading.
- `task-462`: Add task backlog view filtered by active project and conversational workflow.
- `task-463`: Add review queue view for approving, rejecting, or requesting more context on artifacts.
- `task-464`: Add recommendation panel for next best tasks, reviews, research, and project actions.
- `task-465`: Add active agent activity view with assignments, status, traces, and artifacts.
- `task-466`: Add active project, active initiative, open tasks, pending reviews, and dashboard status widgets.
- `task-467`: Add UI tests for chat, project switching, task viewing, review actions, and status refresh.

## Phase 73 - Dynamic Commands And Launch Success

- `task-468`: Add skill manifest index for conversational command generation.
- `task-469`: Add slash-command aliases generated from registered Hermes skills and workflows.
- `task-470`: Add natural-language skill invocation for common skill goals without explicit command names.
- `task-471`: Add contextual command recommendation from active project, selected workflow, and recent conversation.
- `task-472`: Add permission and approval checks for generated conversational commands before execution.
- `task-473`: Add Chief of Staff dashboard panels for active project, initiative, tasks, reviews, agents, and recommendations.
- `task-474`: Add end-to-end success scenario for "Build a CRM for wholesalers" through project workspace creation.
- `task-475`: Add user guide for conversational Hermes OS workflows and operator expectations.
- `task-476`: Add rollout and migration guide from command-first Hermes to chat-first Hermes.
- `task-477`: Add regression release tests for dynamic commands, launch workflow, dashboard panels, and documentation links.
