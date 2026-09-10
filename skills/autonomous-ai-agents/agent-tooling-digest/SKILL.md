---
name: agent-tooling-digest
description: "Curated reference of open-source agent tooling relevant to Hermes Agent — persistent memory layers (Mem0, Graphiti, Cognee), multi-agent coordination frameworks (AG2, Google ADK, AWS Strands), and agent safety/interoperability tooling (Microsoft Agent Governance Toolkit, RAMPART, A2A protocol). Use when choosing a memory backend, wiring Hermes into a multi-agent pipeline, adding observability or safety governance to tool-use, or deciding between A2A and MCP."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [agents, memory, multi-agent, coordination, safety, a2a, mcp, observability, reference]
    category: autonomous-ai-agents
    related_skills: [hermes-agent, llm-wiki]
---

# Agent Tooling Digest

A curated, point-in-time reference of high-signal open-source projects for extending
Hermes Agent, across three areas: **persistent memory/retrieval**, **multi-agent
coordination**, and **agent safety + interoperability**. Each entry states what the
tool does and the concrete integration rationale for Hermes.

> **Snapshot date: 2026-06-10.** Star counts, releases, and version numbers below
> were current at compile time. Before recommending or integrating, re-check the
> linked repo for the latest release, license, and API — fast-moving projects drift.
> Verify claims against the source rather than treating this file as ground truth.

## When This Skill Activates

Use this skill when the user:
- Asks which **memory layer** to give an agent persistent cross-session recall, or how to
  retain user preferences / prior outcomes / factual corrections.
- Wants Hermes to **interoperate** with other agents or orchestrators (AG2, LangGraph,
  Google ADK, enterprise platforms), or asks about **A2A vs MCP**.
- Needs **multi-agent coordination** — parallel tool calls, agent swarms, DAG/sequential
  workflows, or wrapping Hermes as a callable sub-agent / service.
- Wants **observability** (tracing, latency breakdown) over multi-step tool-use chains.
- Needs **safety governance** — secrets redaction, sandboxing, human-in-the-loop gating,
  or a repeatable agent **red-team / safety regression** suite.

For general agent orchestration patterns within Hermes itself, see [[hermes-agent]].
For building a persistent local knowledge base by hand, see [[llm-wiki]].

## Quick Decision Guide

| If the user wants… | Reach for | Why |
|--------------------|-----------|-----|
| Drop-in cross-session memory, minimal setup | **Mem0** | Most-adopted memory layer; atomic-fact extraction, flexible vector backends |
| Memory where facts change over time | **Graphiti** | Time-stamped triples with validity windows; temporal queries |
| Ingest a whole codebase/corpus into a queryable graph | **Cognee** | Local-first ontology builder; "second brain" over docs |
| Heterogeneous multi-agent pipeline, no framework lock-in | **AG2** + **A2A** | Native A2A server/client; cross-framework calls |
| Parallel tool calls / hierarchical delegation | **Google ADK** | ParallelAgent/SequentialAgent/LoopAgent patterns |
| Production observability over tool-use chains | **AWS Strands** | First-class OpenTelemetry tracing out of the box |
| Stop secrets leaking into context/logs; gate risky actions | **MS Agent Governance Toolkit** | Declarative policy, secrets redaction, HITL gating |
| A CI safety-regression suite across model checkpoints | **RAMPART** | pytest-native adversarial scenarios |
| Make Hermes a discoverable, callable service | **A2A protocol** | Agent cards + standard HTTP messaging |

---

## 1. Persistent Memory & Knowledge Retrieval

Hermes's tool-use and multi-turn chains have no persistent cross-session memory by
default. These layers add it without changing the core model.

### 1.1 Mem0 — Universal Agent Memory Layer
- **Repo:** https://github.com/mem0ai/mem0
- **What it does:** A "universal memory layer." Converts conversation history into atomic
  memory facts via an extraction pipeline, stored across a hybrid backend — a vector DB
  (Qdrant, Chroma, Milvus, pgvector, or Redis) for semantic retrieval, plus an optional
  knowledge-graph layer (Neo4j, Kuzu, Amazon Neptune). ~55k+ stars at snapshot; v1.0
  ships a production REST API and managed cloud tier alongside self-hosting.
- **For Hermes:** Drop in as a memory middleware layer so Hermes retains user preferences,
  prior task outcomes, and factual corrections across sessions. Atomic-fact extraction
  strips conversational noise before storage, keeping retrieval precision high. Vector-
  backend flexibility means it pairs with existing infra (e.g. a self-hosted Qdrant).

### 1.2 Graphiti — Temporal Knowledge Graph (part of Zep)
- **Repo:** https://github.com/getzep/graphiti
- **What it does:** Indexes facts as time-stamped relationship triples; every assertion
  carries a validity window (e.g. "User works at Acme — valid from 2025-01-01"). Answers
  "what did we know *before* X changed?" instead of overwriting state. Apache-2.0,
  self-hostable, Neo4j backend. Zep's benchmarks show it beating pure vector search on
  temporal/relational reasoning.
- **For Hermes:** For long-running tasks where facts change (project status, evolving
  architecture, shifting requirements). Lets Hermes distinguish stale from current context
  without manual prompt engineering. Integration: supplement the context window with
  Graphiti-backed long-term storage, querying with time-scoped lookups at inference time.

### 1.3 Cognee — Self-Hosted Knowledge Graph Engine
- **Repo:** https://github.com/topoteretes/cognee
- **What it does:** Combines vector embeddings, graph reasoning, and ontology generation
  to build a persistent KB from unstructured docs and conversation history. Runs fully
  locally; one-click deploy with Neo4j, FalkorDB, or KuzuDB. Unlike Mem0, it doesn't just
  store facts — it builds a structured ontology, suitable for ingesting whole codebases or
  documentation corpora.
- **For Hermes:** An offline, self-hosted "second brain" for coding/research tasks. Ingest
  a codebase once, then query the resulting graph at inference time instead of re-reading
  files every turn — eliminating repeated large file-read tool calls. Local-first design
  fits enterprise deployments with data-privacy requirements.

**Memory layer pick:** Mem0 for fast drop-in conversational memory; Graphiti when facts
have a time dimension; Cognee for deep, local, structured ingestion of large corpora.

---

## 2. Multi-Agent Coordination Frameworks

All three ship native **A2A** support (see §3.3), the practical path to Hermes joining
heterogeneous pipelines without per-framework adapters.

### 2.1 AG2 (formerly AutoGen) — Open-Source AgentOS
- **Repo:** https://github.com/ag2ai/ag2 · https://www.ag2.ai
- **What it does:** The v0.4+ rewrite of Microsoft's AutoGen, rearchitected around an
  event-driven, async-first execution model. Its **GroupChat** pattern lets specialized
  agents share a conversation with dynamic turn routing. Ships native A2A:
  `A2aAgentServer` exposes AG2 agents as A2A services; `A2aRemoteAgent` calls any external
  A2A-compliant agent — interop with Google ADK, LangGraph, and OpenAI agents in one pipeline.
- **For Hermes:** Wrap a Hermes instance as an A2A service callable by orchestrators in
  other frameworks, or have Hermes call out to specialized sub-agents (code-execution,
  search) built in AG2/LangGraph/ADK. The path to multi-agent pipelines without lock-in.

### 2.2 Google Agent Development Kit (ADK)
- **Docs/Repo:** https://google.github.io/adk-docs/ · https://github.com/google/adk-python
- **What it does:** Structures agent systems as hierarchical trees (root delegates to
  sub-agents recursively). Three built-in orchestration patterns — **SequentialAgent**,
  **ParallelAgent**, **LoopAgent** — plus LLM-driven dynamic routing. Pluggable session
  state (in-memory, Firestore, Redis). Native A2A; tight Vertex AI integration.
- **For Hermes:** **ParallelAgent** is immediately relevant — for tasks needing multiple
  sources at once (web search + file read + code execution), wrap each tool-use as a
  sub-agent and run them in parallel to cut wall-clock latency vs serial tool calls. ADK can
  also be the orchestration layer above Hermes, with Hermes as the core reasoning agent.

### 2.3 AWS Strands Agents SDK
- **Repo:** https://github.com/strands-agents/sdk-python
- **What it does:** Model-driven, framework-agnostic SDK (open-sourced by AWS, May 2025;
  14M+ downloads). Supports Bedrock, Anthropic, OpenAI, Ollama, any LiteLLM provider.
  Multi-agent patterns: **GraphAgent** (DAG workflows), **SwarmAgent** (emergent peer
  coordination), **WorkflowAgent** (linear pipelines). First-class **OpenTelemetry** tracing
  out of the box; native A2A.
- **For Hermes:** Solves the observability gap — Hermes lacks structured production tracing
  over multi-step tool-use. Wrapped in a Strands workflow, every tool call, LLM invocation,
  and memory lookup emits structured traces ingestible by Grafana/Jaeger/Datadog — enabling
  regression detection, cost tracking, and debugging of complex agentic failures.

---

## 3. Agent Safety & API Interoperability

Hermes's tool-use (filesystem, API calls, code execution) is its most powerful and most
dangerous surface. These tools harden and standardize it.

### 3.1 Microsoft Agent Governance Toolkit
- **Repo:** https://github.com/microsoft/agent-governance-toolkit
- **What it does:** Runtime security governance for autonomous agents (April 2026, MIT).
  Addresses all 10 OWASP Agentic AI Top 10 risks with deterministic, sub-millisecond policy
  enforcement: zero-trust identity for agent-to-tool calls, execution sandboxing, **secrets
  redaction** (models never see raw credentials in tool responses), and human-in-the-loop
  gating for destructive/irreversible actions. Declarative YAML/JSON policies, framework-
  agnostic (any agent that makes tool calls).
- **For Hermes:** The secrets-redaction middleware is immediately deployable as a wrapper
  around Hermes's tool-use pipeline — preventing API keys, tokens, or PII from leaking into
  context or logs. HITL policies provide a configurable safety layer for enterprise
  deployments where Hermes has elevated system access.

### 3.2 Microsoft RAMPART — Agent Red-Teaming Framework
- **Announcement:** https://www.microsoft.com/en-us/security/blog/2026/05/20/introducing-rampart-and-clarity-open-source-tools-to-bring-safety-into-agent-development-workflow/
- **What it does:** Risk Assessment and Measurement Platform for Agentic Red Teaming — a
  **pytest-native** framework for encoding adversarial and benign agent scenarios as
  repeatable, version-controlled tests. Red-team findings (prompt injection, tool-misuse
  exploits, data-exfiltration scenarios) are written as standard Python tests runnable in
  CI/CD. Released May 2026.
- **For Hermes:** Nous ships regular Hermes updates; without a safety regression suite it's
  hard to ensure a new checkpoint doesn't introduce new tool-misuse surfaces. RAMPART
  scaffolds a Hermes-specific safety test suite — known jailbreaks, tool-call injection, and
  filesystem-boundary violations as permanent CI tests that run on every model/prompt change.
  Especially valuable given Hermes's strong function-calling, a high-value adversarial target.

### 3.3 A2A (Agent2Agent) Protocol
- **Site/Repo:** https://a2a-protocol.org · https://github.com/google-a2a/A2A
- **What it does:** Open standard (Linux Foundation governance) for how agents advertise
  capabilities and communicate via structured, signed messages. Each agent publishes an
  **agent card** — a machine-readable manifest of capabilities, I/O schemas, and trust
  metadata. Others discover and invoke it over a standard HTTP messaging layer. **A2A is
  complementary to MCP:** MCP governs agent-to-tool (vertical); A2A governs agent-to-agent
  (horizontal). AG2, Google ADK, and AWS Strands all ship native A2A.
- **For Hermes:** Implementing A2A turns Hermes from a standalone model into a composable
  service any A2A orchestrator can discover and invoke without custom glue. A Hermes agent
  could publish as an A2A "reasoning agent," callable by an AG2-managed research pipeline
  with no Nous-specific adapter. This is the interoperability standard the ecosystem is
  converging on — early adoption reduces future integration debt.

### A2A vs MCP — don't confuse the axes

| | **MCP** (Model Context Protocol) | **A2A** (Agent2Agent) |
|---|---|---|
| Axis | Vertical: agent → **tools/data** | Horizontal: agent ↔ **agent** |
| Question | "How does this agent call a tool?" | "How do two agents discover & talk?" |
| Hermes today | Already speaks MCP (e.g. registered MCP servers) | Candidate to add for cross-framework interop |

They are complementary, not competing — a fully composable Hermes speaks **both**.

---

## How to Use This Reference

1. **Match the need** to the Quick Decision Guide above.
2. **Open the linked repo** and confirm the latest release/license/API — this file is a
   2026-06-10 snapshot and details drift.
3. **Prefer self-hostable, permissively-licensed options** for enterprise/privacy-sensitive
   Hermes deployments (Graphiti/Cognee local-first; Governance Toolkit MIT; AG2/Strands OSS).
4. **Combine layers deliberately** — e.g. Mem0 for memory + Strands for tracing + Governance
   Toolkit for secrets redaction is a coherent production stack; A2A ties Hermes into a
   larger multi-agent system on top.
5. **Validate before integrating** — for memory and coordination claims (latency, benchmark
   wins), reproduce on your own workload rather than trusting vendor benchmarks.
