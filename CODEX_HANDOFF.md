# Codex Handoff — Hermes-based Engineering Agent

## 1. Mission

Build a model-agnostic **Software Engineering Agent / Agent Development Harness** on top of the open-source **NousResearch/Hermes Agent** runtime.

The product is not a Codex wrapper, not a Spec Kit fork, and not a starter-template repository. Its purpose is to make different capable models and delegated coding agents work through a stable, verifiable software-engineering execution system.

Long-term principle:

> Models are replaceable engines. The engineering harness owns project context, task workflow, permissions, verification, review, domain engineering, and completion gates.

## 2. Core Architecture Decision

Use Hermes Agent as an **upstream runtime dependency / kernel candidate**, not as a fork in V1.

Hermes provides reusable infrastructure such as:

- agent loop
- model/provider abstraction
- tool registry and dispatch
- plugin surface
- skills
- context/session facilities
- approvals / lifecycle hooks
- optional delegation to coding tools

Our product owns the software-engineering-specific layer:

```text
Engineering Agent
├── EngineeringOrchestrator
├── VerificationEngine
├── EngineeringPolicy
├── ProjectInspector
├── ProjectContext / AGENTS.md
├── Engineering Skills
├── Domain Engineering
│   ├── payment
│   ├── database
│   └── API integrations
└── Hermes Engineering Plugin
        ↓
    Hermes Runtime
        ↓
    Model Providers / Tools
```

## 3. Runtime Boundary

### Hermes responsibility

- LLM/provider integration
- generic agent execution
- generic tool calling
- plugin discovery
- skills loading
- generic context/session/memory facilities

### Our responsibility

- repository understanding
- project bootstrap
- architecture discovery
- task classification
- engineering workflow states
- deterministic verification
- completion gating
- build/test/lint/verify contract
- code review gate
- database safety
- payment safety
- production guardrails
- engineering-specific evals

## 4. Completion Must Be Controlled by the Harness

Do not rely on a system prompt saying “remember to test”.

Target state machine:

```text
UNDERSTAND
   ↓
EXPLORE
   ↓
PLAN (when required)
   ↓
IMPLEMENT
   ↓
VERIFY
   ├── FAIL → FIX → VERIFY
   └── PASS
         ↓
       REVIEW
         ├── FAIL → FIX → VERIFY
         └── PASS
               ↓
              DONE
```

The intelligent agent works inside stages. The harness decides whether the task may advance.

## 5. Verification Principle

An agent saying “done” is not sufficient evidence.

A task is complete only after applicable deterministic checks have produced fresh evidence, such as:

- git diff inspected
- build succeeds
- relevant tests succeed
- lint/format checks succeed when configured
- no obvious secret leakage
- API compatibility evaluated when relevant
- DB compatibility/migration safety evaluated when relevant
- domain-specific checks completed
- review gate passed

## 6. Existing Engineering Blueprint

The original design blueprint is kept separately as `agent-engineering-blueprint.md`.

Important concepts from that blueprint that remain valid:

```text
Global Rules        → how the agent should behave
Project Context     → facts about this repository
Skills              → how a class of task should be performed
ExecPlan            → plan for a specific complex task
Tools               → real data and deterministic operations
Task/Scripts        → stable build/test/verify entry points
Hooks/CI            → hard quality gates
Subagents           → specialization / separation of concerns
Docs/ADR/Runbooks   → durable engineering knowledge
```

## 7. Role of Codex

Codex Desktop is the development environment used to build this project now.

Codex is **not** the architectural foundation of the product.

Later Codex may also become one delegated coding backend:

```text
Engineering Agent
├── Native Hermes model execution
└── Delegated execution
    ├── Codex
    ├── Claude Code
    └── OpenCode
```

Do not hard-code current product architecture around Codex-specific behavior.

## 8. Role of Spec Kit

Spec Kit is not the runtime foundation.

Treat it as a future optional engineering integration/capability for:

- specification-driven feature development
- reusable workflows
- presets/bundles where useful

Do not introduce Spec Kit into the V1 core until the Hermes plugin/runtime boundary has been validated.

## 9. Role of Superpowers

Use Superpowers as upstream methodology/reference for skill design, especially:

- systematic debugging
- planning
- verification before completion
- code review
- subagent development patterns

Do not make the V1 runtime depend on Superpowers.

## 10. V1 Non-Goals

Do NOT build these yet:

- Hermes fork
- own model provider framework
- own generic tool-call protocol
- own generic plugin framework
- complex MCP gateway
- web console
- SaaS control plane
- fleet management
- automatic production deployment
- autonomous self-modification of production skills
- broad multi-agent orchestration
- Spec Kit package manager clone

## 11. Recommended Local Development Layout

Keep upstream and product repositories separate:

```text
engineering-agent-lab/
├── hermes-agent/          # upstream NousResearch repository; avoid product edits
└── hermes-engineering/    # our product repository
    ├── AGENTS.md
    ├── CODEX_HANDOFF.md
    ├── docs/
    ├── src/
    ├── skills/
    ├── tests/
    └── pyproject.toml
```

Do not commit product work into `hermes-agent/` during V1 discovery.

## 12. Proposed Product Package Shape

Do not generate all implementations immediately. This is the target shape to validate:

```text
hermes-engineering/
├── AGENTS.md
├── CODEX_HANDOFF.md
├── pyproject.toml
├── docs/
│   ├── architecture/
│   ├── decisions/
│   └── research/
├── src/
│   └── hermes_engineering/
│       ├── orchestrator/
│       ├── verification/
│       ├── policy/
│       ├── inspector/
│       ├── context/
│       ├── runtime/
│       └── plugin/
├── skills/
│   ├── project-bootstrap/
│   ├── bugfix/
│   ├── verification/
│   └── review/
└── tests/
```

## 13. First Technical Milestone

Before implementing the full product, perform a **Hermes source architecture reconnaissance**.

Questions that must be answered from source code, not guesses:

1. What is the actual main agent-loop entry point?
2. How are providers resolved and invoked?
3. How are tools registered, authorized, dispatched, and reported back to the model?
4. What plugin hooks exist before/after model and tool calls?
5. Can a plugin veto a tool call?
6. Can a plugin inject context before an LLM call?
7. How are skills discovered and loaded?
8. How are project context files discovered?
9. How are sessions represented and persisted?
10. Where is task completion determined?
11. Is there a hook or wrapper point that can prevent final completion until verification passes?
12. Can an outer orchestrator invoke the Hermes agent for one workflow stage at a time?
13. Which public Python APIs are stable enough for a downstream project?
14. What minimum Hermes dependencies are needed for an engineering-only runtime profile?
15. Can Codex/Claude Code/OpenCode delegation be invoked programmatically or only through skills/shell?

Create an evidence-based report with file paths and symbols for each answer.

## 14. First POC

After reconnaissance, build the smallest independent plugin/extension POC possible.

POC goal:

> Add one engineering-specific lifecycle capability without modifying Hermes core.

Suggested first POC:

`engineering_guard` plugin

Behavior:

- observe tool calls
- block one explicitly configured dangerous shell pattern
- record the blocked event
- inject a short engineering-policy context message where supported
- include automated tests

This POC answers whether the Hermes plugin surface is sufficient for our engineering guardrail layer.

Do not build VerificationEngine until this boundary is proven.

## 15. Decision Gate After POC

After the plugin POC, produce one architecture decision:

### Option A — Plugin-only
Use if all required interception and lifecycle controls can be expressed through plugins/hooks.

### Option B — Plugin + Outer Orchestrator
Preferred if plugins handle tools/context but cannot reliably control engineering workflow completion.

### Option C — Small upstream contribution
Use when one narrowly scoped Hermes lifecycle extension would make the architecture clean. Prefer an upstream PR over a fork.

### Option D — Fork
Only if a critical product requirement fundamentally conflicts with Hermes architecture and cannot reasonably be solved through A–C.

## 16. Development Rules

- Inspect before modifying.
- Distinguish source-backed facts from design hypotheses.
- Do not refactor Hermes while researching it.
- Keep upstream changes separate from product code.
- Prefer the smallest proof that validates an architectural boundary.
- Add tests for deterministic behavior.
- Never weaken sandbox/approval behavior merely to make the POC easier.
- Do not claim a lifecycle capability exists unless verified in source or executable tests.
- Record important architectural decisions in ADRs.

## 17. Immediate Codex Task

Start with source reconnaissance only.

Deliver:

1. `docs/research/hermes-runtime-reconnaissance.md`
2. architecture diagram of the actual Hermes execution path
3. table of useful extension points
4. list of blockers/gaps
5. recommendation: plugin-only vs plugin+outer-orchestrator
6. proposed minimal POC file layout

Do not implement the product architecture yet unless required to verify an extension point.
