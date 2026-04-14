# Local Gemma 4 Subagent Routing Policy v1

Date: 2026-04-09
Status: active support policy
Parent references:
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `docs/ops/model_pool_policy_v1.json`
- `ops/openclaw/codex_worker_pool_runbook.md`

## Purpose
Define the practical default use of local `gemma4:26b` on the Linux box as a low-cost, always-available support subagent for work that does not require premium cloud-model authority.

This policy is intentionally support-scoped first:
- it defines when Gemma should be preferred,
- when it must not be the final authority,
- and when work should escalate to Codex Spark or heavy cloud models.

It does **not** by itself change canonical no-LLM authority gates or promote Gemma into final truth authority.

## Local runtime target
- Runtime: local Ollama
- Model: `gemma4:26b`
- OpenClaw model id: `ollama/gemma4:26b`
- Runnable agent presets:
  - `gemma4-local-support`
  - `gemma4-local-triage`
  - `gemma4-local-docs`
  - `gemma4-local-log-audit`
  - `gemma4-local-plan-draft`
- Box: Linux host with local GPU
- Intended posture: cheap local support worker, not final authority lane

## Route class
Add a practical support route beneath the current matrix:
- `LOCAL_SUPPORT` = local Gemma 4 worker via Ollama

Interpretation:
- sits below cloud `SPARK` in cost and authority
- sits above pure deterministic grep/read only when model synthesis is actually useful
- cannot override `NO_LLM` decisions

## Default use cases
Use local Gemma 4 by default for:
1. Repo scanning and first-pass file triage.
2. Summarizing logs, traces, and command output.
3. Drafting implementation plans, checklists, and work breakdowns.
4. First-pass bug localization and likely-cause ranking.
5. Documentation drafting, cleanup, and report scaffolding.
6. Structured extraction from trusted markdown, JSON, YAML, and config files.
7. Test-failure clustering and first-pass remediation hypotheses.
8. Cheap support subagents for audit, triage, compression, and note-making.
9. Broad but low-stakes reading tasks where local throughput is good enough.
10. Repetitive iterative loops where cloud cost would dominate value.

## Preferred task shapes
Gemma is preferred when the task is mainly:
- understand
- summarize
- shortlist
- compare
- classify
- draft
- inspect
- compress
- propose likely next steps

Good scope shapes:
- `single_surface`
- `multi_surface_disjoint` analysis-only work
- support-only fan-out passes that do not write canonical truth

## Do not use Gemma as final authority for
1. `NO_LLM` gate decisions.
2. Freeze-line or certification truth judgments.
3. Final blocker truth packets when precision is critical.
4. High-risk multi-file refactors.
5. Canonical doctrine or policy mutations without stronger review.
6. Safety, rollback, dispatch, lock, or lease authority.
7. Final artifact-backed closeout acceptance.
8. Nuanced architecture decisions where wrong synthesis would be expensive.

## Escalation rules
Escalate from local Gemma to cloud `SPARK` when:
- the task moves from reading to real code mutation,
- answer quality is too shallow or fuzzy,
- tool-use discipline matters more than cost,
- cross-file implementation detail needs stronger reliability,
- outputs are likely to be copied directly into production edits.

Escalate from local Gemma to `HEAVY` when:
- the task is high-risk or coupled,
- the result affects canonical truth or execution policy,
- stronger synthesis depth is needed across many moving surfaces,
- a final decision or closeout must be highly trustworthy,
- Spark failed quality or convergence gates.

Escalate down to `NO_LLM` when:
- a deterministic script or validator can decide the result,
- schema validation or authority gates are the real task,
- the action is scheduler, lock, retry, freshness, or continuity gating.

## Practical default routing rule
Use this shortcut:
- If the task is "understand, summarize, shortlist, classify, or draft", prefer local Gemma.
- If the task is "change important code correctly", use cloud Codex.
- If the task is "decide truth for the system", Gemma is not the authority.

## Suggested operational mapping
- Local Gemma 4:
  - `gemma4-local-support`: general support subagent
  - `gemma4-local-triage`: first-pass repo and incident triage
  - `gemma4-local-docs`: documentation drafting and cleanup
  - `gemma4-local-log-audit`: log compression, audit passes, and trace summarization
  - `gemma4-local-plan-draft`: plan drafting, work breakdowns, and shortlist generation
- Cloud SPARK:
  - bounded code work
  - stronger execution-oriented support tasks
  - low-to-medium risk coding and patch planning
- Cloud HEAVY:
  - hard debugging
  - high-risk implementation
  - final closeout judgment
  - architecture and coupled-system reasoning
- NO_LLM:
  - gates, validators, locks, schedulers, freshness, authority decisions

## Output expectations for Gemma lanes
Gemma outputs should be treated as one of:
- draft
- support evidence
- first-pass hypothesis
- compression/synthesis helper output

They should not be treated as authoritative unless a stronger validator lane or deterministic check confirms them.

## Review posture
Gemma should now be considered:
- approved for support-only local subagent work,
- canonically admitted as the `LOCAL_SUPPORT` route in the routing matrix and model-pool policy,
- still non-authoritative for `NO_LLM` gates, canonical truth mutation, and final high-risk implementation decisions.

## Next optional follow-ons
If desired later:
1. add a lightweight task-class router for automatic Gemma selection,
2. add validator tests that assert `LOCAL_SUPPORT` never becomes the terminal authority for gated or canonical-truth slices,
3. tune preset-specific prompts or tool caps if the named Gemma roles begin to diverge materially.
