# Hermes Super-Agent Evolution Plan (Reviewed)

## Purpose

This reviewed version keeps the original seven-phase roadmap intact while making it more implementation-oriented for the current Hermes codebase.

Review goals:
- Preserve the original intent, ordering, and north-star outcomes
- Translate each phase into concrete changes against current Hermes modules
- Add measurable exit criteria with metric definitions and sample sizes
- Add explicit phase gates, no-go triggers, and updated risk controls

Planning constraints preserved:
- No week-based planning
- Milestone and exit-criteria driven
- Emphasis on reliability, memory quality, eval rigor, and safe autonomy

---

## Strategic North Star

Build an agent that is:
1. Reliable under real tool-heavy workloads
2. Memory-compounding across sessions
3. Self-correcting on failures
4. Measurably improving through benchmarked evals
5. Cost-aware and routing-optimized
6. Safe, auditable, and user-aligned

---

## Current-Architecture Constraints To Preserve

These are not optional. The reviewed plan assumes all implementation work respects them.

- `run_agent.py` remains the single source of truth for the synchronous conversation loop and tool-call execution order.
- Prompt-caching invariants must hold: the assembled system prompt stays stable within a session and is rebuilt only on compression boundaries governed by `agent/prompt_caching.py` and `run_agent.py`.
- Agent-loop tools intercepted in `run_agent.py` (`todo`, `memory`, `session_search`, `delegate_task`) must not be silently rerouted through generic dispatch.
- Delegated children in `tools/delegate_tool.py` must keep isolated context, bounded depth, and restricted tool access.
- Persistent memory changes must not mutate the in-session prompt snapshot in `tools/memory_tool.py`.
- Tests must continue to honor the isolated `HERMES_HOME` behavior in `tests/conftest.py`.

---

## Shared Instrumentation Required Before Any Phase Can Exit

This is not a new phase. It is a cross-phase prerequisite for evidence-based gating.

### Minimum instrumentation substrate

Implement a small, durable event layer before declaring any phase complete. The simplest acceptable path is extending `hermes_state.py` with append-only event tables or writing versioned JSONL artifacts beside existing trajectories.

Required event families:

| Event family | Minimum fields | Primary code surfaces |
|---|---|---|
| Tool execution | `tool_name`, `toolset`, `success`, `error_type`, `retryable`, `latency_ms`, `session_id`, `task_id` | `run_agent.py`, `model_tools.py`, `tools/registry.py` |
| Context/compression | `tokens_before`, `tokens_after`, `reason`, `summary_model`, `sanitizer_repairs` | `agent/context_compressor.py`, `run_agent.py` |
| Memory | `candidate`, `target`, `write_decision`, `confidence`, `reconciliation_action`, `source` | `tools/memory_tool.py`, `run_agent.py` |
| Delegation/routing | `route_selected`, `route_shadow`, `child_count`, `budget_used`, `model_selected` | `tools/delegate_tool.py`, `run_agent.py` |
| Safety/audit | `risk_class`, `approval_required`, `approval_result`, `side_effect_type`, `rollback_artifact` | `tools/approval.py`, `tools/terminal_tool.py`, `tools/mcp_tool.py` |
| Eval runs | `suite`, `task_id`, `commit`, `score`, `cost`, `duration`, `baseline_ref` | new `evals/` package or `batch_runner.py` extension |

### Shared rules

- No phase can claim success from anecdotal CLI usage alone.
- All exit criteria below must be computed from versioned artifacts, not ad hoc notes.
- Every phase needs a baseline snapshot before implementation starts.
- Shadow-mode logging is preferred before behavior-changing rollouts.

---

## Operating Principles

### 1) Stability before sophistication
Do not add advanced autonomy on top of brittle orchestration.

### 2) Measured progress only
No feature promotions without eval deltas.

### 3) Memory quality over memory quantity
Store durable, actionable, validated information only.

### 4) Cost-performance balancing
Use model/tool routing based on task complexity and risk.

### 5) Human override for high-impact actions
Any destructive, sensitive, or irreversible action requires explicit approval.

---

## Phase 1: Core Stability and Context Discipline

### Objective
Harden core execution loops so Hermes is dependable under sustained use.

### Why this phase matters
Most agent failures still come from tool orchestration drift, weak error contracts, or context contamination rather than raw model capability.

### Primary implementation surfaces

- `tools/registry.py`
- `model_tools.py`
- `run_agent.py`
- `agent/prompt_builder.py`
- `agent/prompt_caching.py`
- `agent/context_compressor.py`
- `tools/web_tools.py`
- `tests/test_model_tools.py`
- `tests/test_413_compression.py`
- `tests/tools/`

### Implementation tracks

- Add a canonical tool result envelope in `tools/registry.py` for all tool handlers: `success`, `error`, `error_type`, `retryable`, `data`, `metrics`.
- Add strict post-dispatch validation at the registry boundary so malformed tool outputs fail in tests instead of surfacing as ambiguous runtime strings.
- Normalize pre-dispatch and post-dispatch error handling in `model_tools.py` and `run_agent.py::_execute_tool_calls()`.
- Record tool latency, retryability, and error-class metrics at the agent-loop boundary rather than inside each tool ad hoc.
- Tighten context hygiene in `agent/context_compressor.py`: dedupe repeated tool content, cap oversized tool results before reinjection, and count sanitizer repairs explicitly.
- Convert Firecrawl reliability work in `tools/web_tools.py` into named presets with clear retry/backoff matrices and normalized extraction metadata.

### Deliverables

- Tool Contract Spec v1
- Canonical Tool Result Helper v1
- Context Hygiene Ruleset v1
- Firecrawl Preset Profiles v1
- Stability Regression Suite v1

### Measurable exit criteria

- Tool-call success rate is at least `95%` across a representative suite of at least `300` core tool calls, excluding explicit user denials and unavailable toolsets.
- At least `100%` of registered tools return the canonical result envelope in contract tests.
- Context hard-failure incidents are reduced by at least `50%` from baseline across at least `50` long-running sessions.
- Orphaned tool-call repair events are at or below `1%` of compression events.
- Firecrawl requests requiring at least one retry are below `10%`, and terminal failures are below `3%`, on a fixed extraction corpus.

### Phase gate

- Baseline report exists with pre-change metrics for tool success, compression failures, and Firecrawl retries.
- Contract tests cover every registered tool discovered by `model_tools._discover_tools()`.
- Long-session regression suite passes with no prompt-cache invariant violations.
- Go only if the canonical tool envelope is fully adopted; no-go if envelope adoption is partial or if compression failures rise versus baseline.

---

## Phase 2: Procedural Memory That Compounds Capability

### Objective
Transform memory from passive storage into operational leverage.

### Why this phase matters
Compounding performance requires strong retrieval, contradiction handling, and high-signal persistence, not just more stored text.

### Primary implementation surfaces

- `tools/memory_tool.py`
- `tools/honcho_tools.py`
- `tools/session_search_tool.py`
- `run_agent.py` memory and Honcho integration paths
- `hermes_state.py`
- `agent/prompt_builder.py`
- `agent/skill_commands.py`
- `tests/tools/test_memory_tool.py`
- new memory reconciliation tests

### Implementation tracks

- Formalize the current memory tiers instead of inventing new ones from scratch:
  - Working memory: active messages, todo state, and task-local execution context in `run_agent.py`
  - Episodic memory: session outcomes and searchable history in `hermes_state.py`
  - Procedural memory: curated `MEMORY.md`, `USER.md`, Honcho-backed durable user context, and reusable skills
- Add a write-gate scorer before persistence with explicit reasons for accept, reject, or defer.
- Implement contradiction detection and reconciliation as a first-class operation instead of append-only growth.
- Add one retrieval arbitration policy spanning curated memory, Honcho user modeling, and session search so the agent has a single precedence model.
- Preserve prompt caching by routing mid-session retrieval updates through tool outputs or explicit session messages, never silent prompt mutation.
- Add a skill-synthesis proposal flow that generates draft skills or runbooks for review instead of auto-promoting into `~/.hermes/skills/`.
- Add pruning and consolidation routines with metrics for stale-entry removal, merge decisions, and contradiction cleanup.

### Deliverables

- Memory Governance Spec v1
- Retrieval and Reconciliation Engine v1
- Memory Event Log Schema v1
- Skill Synthesis Proposal Workflow v1
- Memory Pruning and Consolidation Routine v1

### Measurable exit criteria

- On a repeated-task corpus of at least `30` paired tasks across at least `10` task types, median time-to-success improves by at least `30%`.
- Explicit user correction rate on repeated task types decreases by at least `25%` from baseline.
- Memory write precision is at least `90%` on an audited sample of at least `100` candidate writes.
- Contradiction resolution accuracy is at least `90%` on a seeded reconciliation set of at least `50` conflict cases.
- Procedural memory or skill reuse appears in at least `40%` of eligible repeated workflows without increasing prompt-cache invalidation incidents above zero.

### Phase gate

- Reconciliation tests cover duplicate, contradictory, stale, and ambiguous entries.
- A before-and-after audit exists for memory growth, pruning rate, and correction rate.
- Prompt-cache stability is explicitly verified across sessions with memory writes.
- Go only if memory helps repeated tasks without increasing drift or mid-session prompt mutation; no-go if write precision or conflict resolution miss the thresholds above.

---

## Phase 3: Self-Correction Loops (Reflexion + Self-Refine Pattern)

### Objective
Enable Hermes to improve output quality within a session without retraining.

### Why this phase matters
Iterative critique and revision can improve outcomes materially, but only if retries are classified, bounded, and measured.

### Primary implementation surfaces

- `run_agent.py`
- `agent/trajectory.py`
- new `agent/self_correction.py` or equivalent helper module
- `tests/test_run_agent.py`
- new retry and critique tests

### Implementation tracks

- Add a structured failure taxonomy covering reasoning failure, tool misuse, missing context, schema mismatch, provider failure, and unsafe-action block.
- Start in shadow mode: record critique recommendations and likely retry actions without changing live responses.
- Promote to bounded live retries only for low-risk and medium-risk classes with explicit budgets and confidence thresholds.
- Log failure signatures and effective correction patterns into trajectories or dedicated event records.
- Add tool-specific retry policies so transient provider/network failures are handled differently from logical errors.
- Ensure high-risk or side-effecting actions never enter autonomous retry without renewed approval.

### Deliverables

- Self-Correction Loop v1
- Failure Taxonomy v1
- Retry Budgeting Policy v1
- Critique Prompt Pack v1
- Retry Outcome Report v1

### Measurable exit criteria

- On a target workflow set of at least `50` tasks, bounded retry increases final success rate by at least `10` percentage points versus first-attempt-only baseline.
- Cost-per-success increases by no more than `20%` while achieving the success gain above.
- Repeated failure-signature recurrence declines by at least `30%` across the same workflow set.
- Hallucination-induced tool misuse is below `2%` of tool calls in the evaluated set.
- High-risk tasks show `100%` compliance with approval-before-retry policy.

### Phase gate

- Shadow-mode report shows the critique signal is useful before live retries are enabled.
- Retry policies are covered by tests for low-risk, medium-risk, and high-risk paths.
- Live rollout stays behind a feature flag until the success-lift and cost ceilings are met.
- Go only if retries improve success without materially increasing unsafe behavior or runaway cost; no-go if cost or tool misuse exceeds thresholds.

---

## Phase 4: Evaluation Harness and Evidence-Driven Improvement

### Objective
Establish rigorous, repeatable measurements for agent capability growth.

### Why this phase matters
Without eval discipline, improvements are anecdotal, regressions hide in noise, and later phases become ungovernable.

### Primary implementation surfaces

- `batch_runner.py`
- `environments/hermes_base_env.py`
- `environments/agent_loop.py`
- `agent/trajectory.py`
- `hermes_state.py`
- `hermes_cli/main.py` or a dedicated eval CLI entrypoint
- `tests/test_provider_parity.py`
- new `evals/` package for tasks, graders, suites, and reports

### Implementation tracks

- Define a versioned eval format for tasks, setup, expected outputs, graders, and cost/time budgets.
- Use `batch_runner.py` for lightweight dataset sweeps and the existing `environments/` framework for scored agent-loop evaluations instead of building two parallel systems.
- Build three lanes from the original plan: coding, terminal, and agentic multi-step tasks.
- Prefer deterministic graders first; use model-based grading only for outputs that cannot be checked mechanically.
- Create dev, regression, and holdout splits to reduce benchmark overfitting.
- Persist eval artifacts per run: config, commit SHA, raw outputs, scores, latency, token usage, and failure signatures.
- Add smoke-suite and full-suite entrypoints so local development can stay fast while release gates stay meaningful.

### Deliverables

- Hermes Eval Harness v1
- Benchmark Task Suite v1
- Grader Library v1
- Metrics Dashboard Spec v1
- Regression Policy v1

### Measurable exit criteria

- Each eval lane has at least `25` versioned tasks with at least one locked holdout split.
- At least `70%` of tasks across the full suite use deterministic grading.
- Smoke suite completes unattended in under `30` minutes; full suite completes unattended in a documented, repeatable path.
- Flake rate is below `5%` across two consecutive suite runs on the same commit.
- Every major merge touching `run_agent.py`, `model_tools.py`, `tools/`, or `agent/` attaches a pre/post eval diff before promotion.

### Phase gate

- Baseline scorecards exist for all three lanes and are checked into versioned artifacts.
- Holdout tasks are separated from day-to-day tuning tasks.
- Regression thresholds are defined before the suite is used as a gate.
- Go only when the eval harness is trusted enough to reject regressions; no-go if flakiness or grading ambiguity obscures signal.

---

## Phase 5: Scalable Orchestration and Dynamic Routing

### Objective
Increase throughput and quality by decomposition, specialization, and model/tool routing.

### Why this phase matters
Complex tasks benefit from manager-worker structures and specialized execution paths, but only when routing is auditable and state isolation is real.

### Primary implementation surfaces

- `tools/delegate_tool.py`
- `cli.py`
- `run_agent.py`
- `model_tools.py`
- `toolsets.py`
- `agent/model_metadata.py`
- `tools/mcp_tool.py`
- `tools/mixture_of_agents_tool.py`
- routing and delegation stress tests

### Implementation tracks

- Fix known shared-state hazards before scaling delegation, especially the process-global `_last_resolved_tool_names` path noted in `model_tools.py`.
- Add explicit delegation budgets: child count, iteration budget share, latency ceiling, and return-summary size budget.
- Introduce routing in shadow mode first: log recommended model/tool path before allowing policy-driven execution.
- Add one central routing policy layer above the current CLI, agent-loop, delegation, and multi-model entry points so behavior stays explainable.
- Classify tasks by complexity, risk, and determinism so routing is explainable rather than heuristic sprawl.
- Keep parent context clean by enforcing summary-only child returns and size caps for delegated outputs.
- Expand MCP integrations behind a checklist covering auth scope, side-effect classification, and audit support before tool exposure.

### Deliverables

- Orchestration Topology v1
- Routing Policy Engine v1
- Cost and Latency Decision Table v1
- Delegation Budget Policy v1
- MCP Integration Checklist v1

### Measurable exit criteria

- On a complex-task corpus of at least `40` tasks, throughput improves by at least `40%` or median time-to-success improves by at least `25%`, while quality is non-inferior on eval scores.
- Cost per successful task declines by at least `15%` relative to the non-routed baseline.
- Delegated-task summaries keep parent-context growth within `10%` of the non-delegated baseline for comparable tasks.
- Orchestration or routing failures do not increase above baseline error rates from Phase 1.
- Cross-session or cross-agent toolset leakage incidents are reduced to zero in stress tests.

### Phase gate

- Shadow-mode routing report shows policy quality before live routing is enabled.
- Delegation stress tests cover concurrency, interrupts, failures, and budget exhaustion.
- The `_last_resolved_tool_names` global-state risk is retired or fully contained before enabling broader delegation.
- Go only if routing improves cost or throughput without hurting reliability; no-go if orchestration failures or context pollution rise.

---

## Phase 6: Safety, Governance, and Auditability

### Objective
Ensure capability growth remains aligned with user trust and operational safety.

### Why this phase matters
More autonomy without stronger approvals, provenance, and rollback support creates compounding operational risk.

### Primary implementation surfaces

- `tools/approval.py`
- `tools/terminal_tool.py`
- `tools/file_operations.py`
- `tools/skills_guard.py`
- `tools/mcp_tool.py`
- `tools/memory_tool.py`
- `tools/checkpoint_manager.py`
- `agent/redact.py`
- `hermes_state.py`
- safety and adversarial test suites

### Implementation tracks

- Convert safety policy into an explicit action taxonomy: read-only, reversible side effect, irreversible side effect, sensitive data access, and policy-sensitive content.
- Expand approval gating beyond shell danger patterns so MCP actions, browser actions, file mutations, and external side effects share one risk model.
- Add provenance tagging for high-impact outputs: which tool ran, what was changed, what approval existed, and how to undo it.
- Add secret and PII detection checks before memory persistence, external transmission, and artifact logging.
- Store audit records in a queryable form rather than only console output.
- Require rollback artifacts for automated file edits, external writes, or stateful integrations whenever rollback is possible.

### Deliverables

- Safety Policy Spec v1
- Unified High-Risk Approval Gate v1
- Data Governance Controls v1
- Audit Trail Schema v1
- Rollback Playbook v1

### Measurable exit criteria

- Zero severity-1 unsafe autonomous actions on an adversarial validation set of at least `100` tasks.
- `100%` of destructive, sensitive, or irreversible actions pass through approval or explicit policy denial paths.
- `100%` of high-impact actions emit audit records with actor, rationale, tool path, and result.
- Rollback drills succeed for `100%` of sampled reversible workflows in the validation set.
- Memory and audit persistence show zero seeded-secret or seeded-PII leakage incidents in tests.

### Phase gate

- Adversarial and red-team report exists with severity rubric and reproducible cases.
- Approval, audit, and rollback paths are exercised in automated tests, not only manual demos.
- High-impact action coverage is measured by event logs, not inferred.
- Go only if safety coverage is complete for the currently enabled capability surface; no-go if any high-impact path lacks auditability or rollback expectations.

---

## Phase 7: Continuous Evolution Loop (Session-Based Cadence)

### Objective
Create a repeatable improvement loop that compounds with each cycle.

### Why this phase matters
The end state is not a feature set. It is a disciplined operating system for improving Hermes without drifting on quality, cost, or safety.

### Cycle template

1. Run smoke and targeted eval suites
2. Identify the top three bottlenecks by impact on success, correction burden, or cost
3. Select one to three focused changes only
4. Ship behind flags or shadow mode where appropriate
5. Re-run evals and compare against baseline and holdout
6. Promote only positive-net changes
7. Archive lessons into skills, memory rules, routing policy, or test cases

### Primary implementation surfaces

- `evals/` results and reports
- `batch_runner.py`
- `agent/trajectory.py`
- `agent/skill_commands.py`
- `hermes_state.py`
- versioned phase-gate and risk-review artifacts

### Deliverables

- Evolution Cycle SOP v1
- Bottleneck Prioritization Rubric v1
- Promotion Checklist v1
- Knowledge Capture Template v1
- Risk Review Template v1

### Measurable exit criteria

- Three consecutive evolution cycles show non-regressing holdout scores and at least one improved primary metric per cycle.
- Cost per successful outcome is stable or decreasing across the same three cycles.
- Manual intervention rate decreases across the same three cycles on the tracked workflow set.
- Every promoted change has a linked eval diff and risk-review record.

### Phase gate

- Each cycle produces a single decision record: promote, hold, or revert.
- Unresolved high-severity risks cannot remain open for more than two cycles without an explicit exception record.
- Knowledge captured from one cycle must be traceable to later behavior changes, tests, or skill updates.
- Go only if the loop keeps producing measurable gains; no-go if changes are being promoted without durable evidence.

---

## Phase Gate Framework

Use the same evidence package for every phase so gate decisions stay comparable.

### Required gate packet

- Scope declaration with exact files or modules touched
- Baseline metrics snapshot taken before implementation
- Implementation checklist with all required deliverables
- Test report with named suites and pass/fail status
- Eval diff report with pre/post metrics
- Cost impact summary
- Risk review with newly introduced and retired risks
- Go or no-go decision record with rollback note

### Default no-go triggers

- Metrics improve on a dev subset but regress on holdout
- Reliability, safety, or prompt-cache invariants regress
- Success gains depend on hidden cost growth that exceeds stated thresholds
- Feature behavior is enabled without shadow-mode evidence where shadow mode was feasible

---

## Cross-Phase Metric Stack

### Reliability

- Tool success rate
- End-to-end task completion rate
- Retry-to-success ratio
- Compression hard-failure rate

### Quality

- Benchmark pass rates by lane
- Human-rated output quality on sampled tasks
- User correction frequency
- Holdout non-regression score

### Efficiency

- Time-to-success
- Cost-per-success
- Tokens-per-successful-task
- Delegation overhead per successful task

### Safety

- Unsafe action incidents
- Approval gate bypass incidents
- Audit completeness rate
- Secret and PII handling violations

### Memory effectiveness

- Retrieval relevance score
- Contradiction resolution accuracy
- Memory write precision
- Skill reuse and impact score

---

## Risk Register (Updated)

| ID | Risk | Likelihood | Impact | Early warning signal | Mitigation / contingency | Primary surfaces |
|---|---|---|---|---|---|---|
| R1 | Prompt-cache invalidation from mid-session prompt mutation | Medium | High | System prompt hash changes outside compression events | Assert prompt hash stability, route live updates through tool outputs, block silent prompt rebuilds | `run_agent.py`, `agent/prompt_builder.py`, `agent/prompt_caching.py`, `tools/memory_tool.py` |
| R2 | Tool contract drift creates ambiguous runtime failures | High | High | Tool results fail JSON or envelope validation | Canonical result envelope, registry validation, contract tests for all tools | `tools/registry.py`, `model_tools.py`, `tests/tools/` |
| R3 | Context bloat or tool-pair corruption degrades reasoning | High | High | Compression hard failures, orphaned tool repair count rises | Dedupe tool output, cap reinjected content, track sanitizer repairs, strengthen compression tests | `agent/context_compressor.py`, `run_agent.py` |
| R4 | Memory drift, contradiction buildup, or over-persistence | High | High | User corrections rise on repeat tasks, contradiction count grows, memory size grows faster than reuse | Write-gate scoring, reconciliation, pruning, source-precedence rules | `tools/memory_tool.py`, `run_agent.py`, `hermes_state.py` |
| R5 | Local memory and Honcho state diverge | Medium | High | Pending sync queue grows, conflicting facts appear between stores | Add sync-health metrics, write-ahead event log, clear precedence rules, repair job for drift | `run_agent.py` Honcho paths, `tools/memory_tool.py`, `tools/honcho_tools.py` |
| R6 | Eval contamination or benchmark overfitting | Medium | High | Dev-suite gains do not reproduce on holdout | Locked holdout tasks, rotated eval subsets, regression gates based on holdout | new `evals/`, `batch_runner.py` |
| R7 | Cost creep from retries, delegation, browser, or MCP overuse | High | Medium | Cost-per-success and average retries trend upward without quality gain | Shadow-mode routing, retry budgets, route-level cost dashboards, revert policies | `run_agent.py`, `tools/delegate_tool.py`, `tools/mcp_tool.py` |
| R8 | Shared-state orchestration bugs under delegation | Medium | High | Cross-agent tool leakage, execute-code import failures, inconsistent enabled-tools sets | Remove or contain process-global state, add concurrency stress tests, per-agent routing state | `model_tools.py`, `tools/delegate_tool.py`, `run_agent.py` |
| R9 | Safety coverage lags behind new capability surface | Medium | High | New tools or MCP integrations appear without approvals or audit support | Deny-by-default capability rollout, unified approval taxonomy, integration checklist | `tools/approval.py`, `tools/mcp_tool.py`, `tools/terminal_tool.py` |
| R10 | Audit or rollback data is incomplete during real incidents | Medium | High | High-impact actions cannot be reconstructed after the fact | Queryable audit store, rollback artifact requirement, drill reversible workflows | `hermes_state.py`, `tools/approval.py`, `tools/terminal_tool.py`, `tools/checkpoint_manager.py` |
| R11 | Observability gaps make phase gates anecdotal | Medium | Medium | Metrics require manual reconstruction from logs or memory | Mandatory event schema, versioned reports, no gate without baseline artifact | `hermes_state.py`, `agent/trajectory.py`, new `evals/` |

---

## Immediate Execution Order (Retained, With Implementation Notes)

1. Phase 1 core stability and contract normalization
2. Phase 4 foundations early enough to establish baselines and regression visibility
3. Phase 2 memory quality and retrieval discipline
4. Phase 3 self-correction in shadow mode, then bounded live mode
5. Phase 5 scalable orchestration and dynamic routing after state-isolation risks are retired
6. Phase 6 safety hardening before broader autonomous rollout or new high-impact integrations
7. Phase 7 continuous evolution loop as the standing operating cadence

### Practical sequencing notes

- Start instrumentation work before closing Phase 1, or later gates will be weak.
- Treat Phase 4 as an enabling layer, not a late-stage reporting exercise.
- Start designing Phase 6 controls earlier than its formal gate if any new side-effecting capability is introduced in Phases 3 through 5.

---

## Source Spine (Primary References)

### Hermes docs

- https://hermes-agent.nousresearch.com/docs/
- https://hermes-agent.nousresearch.com/docs/guides/tips/
- https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- https://hermes-agent.nousresearch.com/docs/user-guide/features/skills
- https://hermes-agent.nousresearch.com/docs/developer-guide/architecture

### Agent engineering guidance

- https://www.anthropic.com/research/building-effective-agents
- https://www.anthropic.com/engineering/writing-tools-for-agents
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- https://developers.openai.com/api/docs/guides/agent-evals/

### Research foundations

- Voyager: https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651
- Toolformer: https://arxiv.org/abs/2302.04761
- GAIA: https://arxiv.org/abs/2311.12983
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Memory survey: https://arxiv.org/abs/2512.13564

### Benchmarks

- SWE-bench: https://www.swebench.com/SWE-bench/
- Terminal-Bench: https://www.tbench.ai/

### Firecrawl docs

- Scrape endpoint: https://docs.firecrawl.dev/api-reference/endpoint/scrape
- Extract endpoint: https://docs.firecrawl.dev/api-reference/endpoint/extract
- Map endpoint: https://docs.firecrawl.dev/api-reference/endpoint/map
- Extract feature: https://docs.firecrawl.dev/features/extract

---

## Definition of Success

Hermes is considered elite when it demonstrates:

- High reliability on real, tool-heavy tasks
- Persistent compounding gains from memory and skill systems
- Measurable benchmark improvement across lanes
- Controlled cost growth through routing
- Strong safety and audit guarantees
- Low user correction burden over time
- A repeatable promotion process where improvements are evidenced, gated, and reversible
