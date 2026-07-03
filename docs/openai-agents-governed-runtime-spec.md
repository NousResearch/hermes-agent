# OpenAI Agents SDK Governed Runtime Specification

Status: implemented best-in-class baseline for current local constraints  
Owner/orchestrator: Hermes / GPT-5.5  
Worker runtime: OpenAI Agents SDK (`openai-agents==0.17.7`)

## 1. Project boundary

The Helix-on-NAS/Postgres migration is explicitly **not started** by this specification. That project is downstream and separate. The only purpose of this spec is to make the OpenAI Agents SDK lane safe, governed, testable, and ready to support that later architecture work.

## 2. Source documentation alignment

This implementation follows the OpenAI Agents SDK documentation surfaces below:

| SDK doc surface | Relevant guidance | Local implementation |
|---|---|---|
| Agents | `Agent` + `Runner` manage turns, tools, guardrails, handoffs, sessions, structured outputs | Each lane creates a bounded `Agent` and invokes `Runner.run_sync` |
| Guardrails | Input guardrails can be blocking (`run_in_parallel=False`) to prevent token/tool spend; output guardrails validate final output | `_build_agent_guardrail_kwargs()` attaches blocking input guardrails and output proof guardrails |
| Results | Final outputs should be consumed from structured result surfaces, and run metadata can support audits | `output_type=GovernedAgentOutput`; receipts persist the structured result |
| Tracing | SDK tracing exists, but sensitive trace data must be controlled; ZDR orgs may not use tracing | `tracing_disabled=True` and `trace_include_sensitive_data=False` until sanitized tracing is deliberately added |
| Sessions | SDK sessions are useful for multi-turn agent memory but should not be mixed casually with other continuation mechanisms | Not enabled yet; Hermes remains the session/memory orchestrator |
| Handoffs | Handoffs are useful for specialist delegation but should be explicit and logged | Not enabled yet; Hermes exposes review/execute/verify as explicit top-level lanes first |

## 3. Runtime roles

```text
Hermes / GPT-5.5
  = orchestrator, user intent, memory, policy, approval, verification synthesis

OpenAI Agents SDK lanes
  = bounded subordinate workers with structured proof outputs
```

The SDK is not allowed to silently become the top-level orchestrator.

## 4. Tools and authority

| Tool | Role | Authority |
|---|---|---|
| `openai_agents_review` | skeptical review of claims, plans, code, or architecture | advisory, no mutation |
| `openai_agents_execute` | bounded drafting/execution inside SDK context | limited; no Hermes filesystem/terminal/browser inheritance |
| `openai_agents_verify` | independent verification/falsification pass | advisory/no mutation, high skepticism |
| `openai_agents_run` | backward-compatible alias | maps to execute lane |
| `openai_agents_architecture` | composed architecture workflow | runs execute → review → verify lanes and emits aggregate receipt; stage failures return blocked aggregate receipts instead of unstructured tool errors |

## 5. Structured proof contract

Every SDK lane must return this Pydantic schema:

```json
{
  "status": "verified | partial | blocked",
  "summary": "concise result",
  "actions_taken": [],
  "proof": [],
  "risks": [],
  "next_required_action": null,
  "requires_human_approval": false
}
```

Postconditions:

1. `status="verified"` requires at least one proof item.
2. Review/verify lanes that report mutation-like actions are downgraded.
3. Unstructured output is coerced to `partial`.
4. High-risk tasks without explicit scope/approval are rejected before model spend.
5. Architecture workflow stage failures return a structured `blocked` aggregate with stage receipts/errors, preserving auditability instead of collapsing into an unstructured tool error.
6. Architecture workflow stages enforce a 1400-token minimum and concise-output constraint to reduce structured JSON truncation failures while keeping the run bounded.
7. Review/execute/verify lanes retry once after SDK structured-output/JSON/schema failures, writing a first-failure receipt before retrying with compact-output constraints.

## 6. Governance controls

### 6.1 Deterministic preflight

Before invoking the SDK/model, `_preflight_request()` blocks high-risk tasks unless constraints include explicit authorization or read-only/no-mutation scope.

High-risk tokens include deletion, reset, install/uninstall, restart/stop service, credentials/secrets/tokens, firewall/registry/admin/sudo, chmod/chown, and `rm -rf`.

### 6.2 Native SDK guardrails

`_build_agent_guardrail_kwargs()` attaches:

- blocking input guardrail with `run_in_parallel=False`;
- output guardrail requiring schema/proof correctness.

The deterministic preflight remains authoritative because it is cheaper and less sensitive to SDK API changes.

### 6.3 Receipts

Every successful SDK run writes a sanitized JSON receipt to:

```text
$HERMES_HOME/receipts/openai_agents/
```

Receipts include lane, model, SDK version, governance contract, result, enforcement metadata, best-effort token usage when exposed by the SDK, and SHA-256 hashes returned to Hermes for independent proof. Secret-bearing keys such as API keys/tokens/secrets/passwords/credentials are removed before persistence.

Failed SDK lane attempts also write sanitized blocked-status receipts when the bridge has enough local context to do so. Failure receipts fingerprint task/context with SHA-256 and character counts instead of persisting prompt/context text, so parsing/model errors remain auditable without durable private-content leakage.

### 6.4 Model trust policy

Blocked model/provider fragments include Alibaba, Baichuan, DeepSeek, Doubao, Kimi, MiniMax, Moonshot, Qwen, Stepfun, Tencent, Volcengine, Wenxin, Yi, and Zhipu.

## 7. Verification workflow

Plan → specification → execution → verification → test is represented as:

1. Plan: `.hermes/plans/2026-07-03_100839-openai-agents-best-in-class-runtime.md`
2. Specification: this document.
3. Execution: `plugins/openai_agents/*`.
4. Verification: live smoke receipts under `$HERMES_HOME/receipts/openai_agents/`.
5. Test: `tests/plugins/test_openai_agents_bridge.py` plus `tests/test_toolsets.py`.
6. Hard local gate: `python scripts/check_openai_agents_quality.py`.
7. Live bounded gate when needed: `python scripts/check_openai_agents_quality.py --live-smoke`.

## 8. Quality gate and eval corpus

Professional-readiness checks are encoded in:

```text
scripts/check_openai_agents_quality.py
scripts/generate_openai_agents_proof_bundle.py
schemas/openai-agents-receipt.schema.json
evals/openai_agents/governance_cases.json
docs/openai-agents-project-tracking.json
docs/openai-agents-source-manifest.json
docs/openai-agents-git-workflow.md
```

The quality gate verifies:

- Python compile of SDK plugin and gate script;
- targeted pytest suite;
- `git diff --check` over scoped files;
- static scan for common secret, shell-injection, eval/exec, pickle, and SQL string-format patterns;
- plugin registration and toolset resolution;
- deterministic governance eval corpus;
- project tracking manifest referential integrity across roadmap items, local commit refs, receipt groups, and next actions;
- native git workflow fields: current branch, origin remote, local commit refs, explicit-scope external actions, and allowed untracked paths;
- OpenAI Agents SDK source freshness manifest for official docs, package identity, local assumptions, and recheck triggers;
- proof bundle generator compile/readiness;
- recent current-shape receipt schema validation;
- optional bounded live SDK smoke with `--live-smoke`.

## 9. Readiness criteria for the future Helix/NAS/Postgres project

The future project may start only after:

- governed SDK tool schemas are loaded after gateway restart;
- targeted tests pass;
- live review/execute/verify smoke tests return receipts;
- no unresolved SDK runtime blockers remain for the intended workflow.

## 10. Known non-blocking gaps

| Gap | Why not blocking now | Future upgrade |
|---|---|---|
| Sanitized OpenAI trace export | Cloud traces may expose sensitive content; tracing is intentionally disabled | Add opt-in custom trace processor/redaction policy |
| SDK sessions | Hermes already owns session/memory context; SDK sessions could duplicate/confuse state | Add only for isolated multi-turn SDK workflows |
| SDK handoff graph | The explicit Hermes lane architecture is safer first | Add explicit architect→critic→verifier handoff workflow once lane evals remain stable |
| SDK tools/filesystem | Workers intentionally lack Hermes side-effect tools | Add narrowly scoped SDK tools only behind tool guardrails/approval |

## 11. Current status

This is now a strong best-in-class baseline for a solo advanced developer under Scott's constraints: bounded authority, proof-required outputs, deterministic gates, SDK guardrails, receipts, and regression tests.
