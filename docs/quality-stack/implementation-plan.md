# Hermes Quality Stack Implementation Plan

Goal: add measurable quality, observability, safe Obsidian access, security gates, and optional advanced-stack pilots without replacing the Hermes core agent loop.

Architecture:

- Observability stays in the existing opt-in Langfuse plugin.
- Quality evaluation is deterministic first, with optional LLM-as-judge tools layered later.
- Obsidian access is an opt-in plugin with read safeguards, review-queue-only writes, and audit logging.
- Security gates are file-based and CI-friendly without adding runtime dependencies.
- Advanced stacks are tracked as isolated pilots, not core migrations.

Rollback:

- Disable plugins through `hermes plugins disable observability/langfuse` and `hermes plugins disable obsidian-safe-bridge`.
- Remove the optional security configs `.semgrep/hermes-security.yml` and `renovate.json` if CI noise is unacceptable.
- The quality eval harness is standalone; removing `agent/quality_eval.py`, `scripts/hermes_quality_eval.py`, and related tests reverts it.

Pilot decisions:

- LangGraph: pilot only for `trend-discovery-v2` or approval workflows.
- Pydantic AI: pilot only for typed tools and structured outputs.
- GraphRAG: read-only Obsidian subset pilot only.
- OpenAI Agents SDK: reference architecture unless a scoped feature needs it.
