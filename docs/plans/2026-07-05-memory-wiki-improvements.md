# Memory Wiki Improvements Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Improve Hermes memory/wiki continuity by adding inspectable, budgeted, source-grounded wiki-memory artifacts over existing memories, sessions, skills, and future summaries.

**Architecture:** Keep Hermes' built-in `MEMORY.md` / `USER.md`, `state.db` FTS5, skills, curator, and `/journey` as the source of truth. Add an incremental, local-first wiki-memory layer that exports compact JSON/Markdown indexes, supports agentic retrieval-style `find/read/grep/retrieve`, and only injects retrieved context through budgeted, fenced, provenance-aware paths.

**Tech Stack:** Python stdlib first, SQLite FTS5 already in `hermes_state.py`, existing dashboard/FastAPI only for later UI, existing cron/curator for scheduling.

---

## Source Findings

### `kiranklabs/hermes-memory-wiki`

Source: https://github.com/kiranklabs/hermes-memory-wiki

Useful patterns:
- Session archive as inspectable files plus a web UI.
- `facts.json`, `decisions.json`, `wiki-index.json` generated from Hermes `state.db`.
- Facts are overwritten/superseded, not appended forever.
- Decisions keep supersedence trails.
- Cron sessions are classified separately from human work.
- Context injection should be compact and relevance-gated.

Cautions:
- README says macOS-first launch agent setup; Yahya's host is Linux/systemd.
- Uses Next.js frontend and LLM summaries; do not copy as core dependency.
- Repo is small/new: 12 stars, no open issues, low evidence of robustness.

### Memory OS article

Source: https://www.marktechpost.com/2026/06/01/meet-memory-os-a-6-layer-open-source-memory-stack-built-on-top-of-hermes-agent/

Useful patterns:
- Layered memory: workspace files, sessions, structured facts, fabric/tool recall, vectors, LLM wiki.
- Relevance gates before context injection.
- Dedup per session.
- Fallback cascade: hybrid → dense → lexical → SQLite.
- Decay/dedup maintenance.

Cautions:
- Heavy infra: Docker, Qdrant, Redis, ARQ worker.
- No public recall/latency/token benchmarks cited.
- Should be optional plugin/profile feature, not Hermes core default.

### Context vs Memory Engineering

Source: https://machinelearningmastery.com/context-vs-memory-engineering-in-agentic-ai-systems/

Useful patterns:
- Explicit write policy: triggers, eligible data, schema, trust, provenance, conflict handling, TTL.
- Retrieval-aware context budget: retrieval must accept a max token/char budget.
- Placement matters: retrieved memory should be near current task, not buried in middle.
- Maintenance: TTL, decay, dedup, stale fact pruning.

Cautions:
- Avoid unbounded automatic writes; they poison retrieval quality.

### LlamaIndex Legal KB / Retrieval Harness

Source: https://www.marktechpost.com/2026/07/05/llamaindex-legal-kb-agentic-retrieval-over-index-v2-with-retrieve-find-read-and-grep-tools/

Useful patterns:
- Expose knowledge base as filesystem-like agent tools: `find`, `read`, `grep`, `retrieve`.
- Enforce order: establish inventory first, then retrieve, then confirm exact wording before citing.
- Version files and cite exact source spans.

Cautions:
- Visual citation/bbox layer is unnecessary for Hermes memory at first.
- Do not add LlamaCloud dependency; replicate the tool semantics locally.

### DeepWiki / LangChain Wiki Memory

Sources:
- https://cognition.com/blog/deepwiki
- https://www.langchain.com/blog/wiki-memory

Useful patterns:
- Wiki memory is precomputed synthesis over raw sources, not raw RAG chunks.
- Files are a good substrate: inspectable, editable, versionable, agent-readable.
- Code/wiki docs reduce repeated rediscovery for both humans and agents.

Cautions:
- Wiki memory is not all memory; keep preferences/facts/procedures distinct.

---

## Local Hermes Extension Points

| Area | Existing files | Reuse |
|---|---|---|
| Built-in durable memory | `tools/memory_tool.py`, `agent/memory_manager.py` | Source of truth for profile/user facts; strict threat scanning; frozen prompt snapshot.
| Session recall | `tools/session_search_tool.py`, `hermes_state.py` | FTS5 discovery/read/scroll; demotes cron sessions already.
| Journey/memory graph | `agent/learning_graph.py`, `hermes_cli/journey.py` | Existing visual substrate for memories + skills.
| Skills/procedural memory | `tools/skills_tool.py`, `tools/skill_manager_tool.py`, `tools/skill_usage.py`, `agent/curator.py` | Procedural memory and lifecycle telemetry.
| Verification evidence | `agent/verification_evidence.py` | Provenance/evidence model for accepting learned facts or skill changes.
| Dashboard | `hermes_cli/web_server.py` | Later Memory Wiki UI/API endpoint.
| Cron | `cron/`, `cronjob` | Incremental indexing, backups, decay, summaries.

---

## Prioritized Implementation

### Phase 1 — Safe local wiki-memory index

**Objective:** Export existing `MEMORY.md` / `USER.md` into structured, inspectable JSON entries with source, category, stable id, keywords, timestamps, and budgeted retrieval.

Files:
- Create: `agent/memory_wiki.py`
- Create: `tests/agent/test_memory_wiki.py`

Features:
- Parse entries split by `§` without changing memory files.
- Stable IDs from source/index/content hash.
- Heuristic categories: identity, preference, environment, constraint, procedural, decision, fact.
- Keyword extraction and source provenance.
- `select_memory_context(query, max_chars)` for retrieval-aware budgeting.
- No LLM, no vectors, no writes to user memory.

Verification:
- `PYTHONWARNINGS=error python -m pytest tests/agent/test_memory_wiki.py -q`
- Related: `tests/agent/test_learning_graph.py tests/tools/test_memory_tool.py`

### Phase 2 — CLI export command

**Objective:** Add `hermes memory wiki-index --out ~/.hermes/memory-wiki/wiki-index.json`.

Files:
- Modify: `hermes_cli/subcommands/memory.py`
- Modify: `hermes_cli/main.py` or route through `hermes_cli/memory_setup.py` if refactoring allows.
- Add tests under `tests/hermes_cli/`.

Output:
- JSON by default.
- Optional Markdown page with facts/preferences/constraints/decisions.

### Phase 3 — Agentic local retrieval harness

**Objective:** Add local knowledge tools over generated wiki files and session summaries:
- `wiki_find`
- `wiki_read`
- `wiki_grep`
- `wiki_retrieve`

Use exact source confirmation before citation. Keep disabled by default until mature.

### Phase 4 — Session summaries + decisions

**Objective:** Incrementally summarize sessions into `~/.hermes/memory-wiki/sessions/*.json`, `facts.json`, `decisions.json`.

Rules:
- Human sessions first; cron/subagent/tool sessions separate.
- No auto-write into `MEMORY.md`; generate candidate facts for review.
- Supersedence trail for decisions.

### Phase 5 — Dashboard/Journey UI

**Objective:** Add Memory Wiki panel to dashboard and enrich `/journey` with category counts, decision trails, and links to session provenance.

### Phase 6 — Optional advanced local memory backend

**Objective:** Evaluate Qdrant/BM25 hybrid retrieval as an optional plugin, not core.

Acceptance criteria:
- Benchmarked recall wins over FTS5/session_search.
- Measured latency/token savings.
- Docker-free fallback remains available.

---

## First Safe Slice Acceptance Criteria

- Pure stdlib.
- Read-only over memory files.
- Deterministic tests.
- No gateway restart required.
- Does not mutate `MEMORY.md` / `USER.md` or inject into prompt.
- Produces a reusable substrate for CLI/dashboard/cron later.
