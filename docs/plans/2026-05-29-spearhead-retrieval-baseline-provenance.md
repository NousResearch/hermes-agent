# Spearhead retrieval baseline — provenance & non-adoption note

**Date:** 2026-05-29
**Origin:** Source-spike parent `t_f555a6e8` inspected
[`jamwithai/production-agentic-rag-course`](https://github.com/jamwithai/production-agentic-rag-course)
at commit `424a0eb99edf841994f2a9a053912b489d2a94ff`. Verdict: **ADOPT
SELECTIVELY — retrieval patterns only; no direct code import; no
OpenSearch/Jina/LangGraph/Langfuse adoption without Filip approval.**
Follow-up `t_73dd3561` (this note) records the existing baseline so future
workers do not re-import an external retrieval stack that Hermes/Mempalace
already covers with a safer, dependency-light design.

> Status: durable design/provenance note. No production code change. The
> repo working tree was already dirty from other lanes when this was added;
> this note is a single new untracked doc and was not committed/pushed.

## TL;DR for future workers

Before reaching for OpenSearch, Jina embeddings/rerankers, LangGraph, or
Langfuse to "add retrieval," know that Hermes already has two retrieval
baselines, each verified below with exact citations:

1. **Conversation recall** — deterministic SQLite **FTS5** full-text search
   over the session message store, exposed as the `session_search` tool.
   Zero LLM calls.
2. **Corpus recall** — Mempalace **hybrid BM25 + vector** candidate union
   with verbatim drawer content, neighbor-chunk expansion, and virtual
   line-number provenance for citation.

Neither requires a new external service or paid API.

## Baseline 1 — Conversation recall (deterministic FTS / `session_search`)

- **Tool:** `tools/session_search_tool.py`. Single-shape tool with three
  modes inferred from args — discovery (`query`), scroll
  (`session_id` + `around_message_id`), browse (no args). The module
  docstring is explicit: *"All three modes operate on the SQLite session DB
  via the FTS5 index … No LLM calls anywhere — every shape returns actual
  messages from the DB"* (`tools/session_search_tool.py:21-23`). Dispatch is
  in `session_search()` (`tools/session_search_tool.py:378-450`); the
  discovery path runs FTS5 then anchors a window + bookends per hit
  (`_discover`, `tools/session_search_tool.py:277-375`, calling
  `db.search_messages` at `:289` and `db.get_anchored_view` at `:337`).
- **Engine:** `hermes_state.py`. The FTS5 virtual table `messages_fts` and
  its insert/delete/update triggers are defined at `hermes_state.py:290-313`
  (plus a trigram table `messages_fts_trigram` for CJK/substring at
  `:319-343`). `search_messages()` (`hermes_state.py:2154`) runs
  `messages_fts MATCH ?` (`:2212`) ranked by **FTS5 BM25 relevance by
  default** (`ORDER BY rank`, `hermes_state.py:2209`; docstring at `:2177`),
  with optional `newest`/`oldest` temporal ordering. User input is
  defended via `_sanitize_fts5_query()` (`hermes_state.py:2071`).
- **Provenance for recall:** `get_anchored_view()`
  (`hermes_state.py:1766`) returns the ±window around the FTS5 hit plus
  session bookends (first/last user+assistant turns), so a hit anywhere in a
  long session yields goal → match → resolution on one call without loading
  the whole transcript. Built on the `get_messages_around()` primitive
  (`hermes_state.py:1688`).

This path is deterministic, local, and free — no embeddings, no network, no
LLM.

## Baseline 2 — Corpus recall (Mempalace BM25/vector union + provenance)

File: `/home/filip/hermes_research/mempalace/mempalace/searcher.py` (branch
`develop`, inspected at `6957c7e`).

- **Hybrid retrieval:** module docstring — *"Hybrid search: BM25 keyword
  matching + vector semantic similarity"* (`searcher.py:2-10`). Okapi-BM25
  scoring is implemented in-process in `_bm25_scores()`
  (`searcher.py:63-90`); the plain vector path is `search()`
  (`searcher.py:294`), which returns **verbatim drawer content**.
- **Candidate union:** `search_memories(..., candidate_strategy="union")`
  widens the rerank pool's *source* by appending BM25-only sqlite
  candidates to the vector hits via `_merge_bm25_union_candidates()`
  (`searcher.py:637-703`; registered in `_CANDIDATE_MERGERS` at `:709-712`;
  entry point `search_memories` at `:748`). BM25 is a ranking signal that
  can only help, never gate — vector-only selection would otherwise skip
  docs with strong keyword signal but distant embeddings.
- **Neighbor provenance:** `_expand_with_neighbors()`
  (`searcher.py:194-243`) fetches the ±radius sibling chunks of a matched
  drawer so a chunk boundary clipping mid-thought still returns enough
  context, falling back to the matched drawer alone on any failure.
- **Virtual line / citation provenance:** read-time line-number grid
  (`searcher.py:1023-1077`) — `render_with_line_numbers()` (`:1037`) and
  `extract_line_range()` (`:1057`) resolve closet pointers like
  `→2026-01-18:L55-L72` against verbatim drawers without rewriting the
  corpus. Source drawer text is never mutated.

## Explicit non-adoption statement

Adopting the production-agentic-rag course is **pattern-only**. Do **not**
provision, import, or add a dependency on **OpenSearch, Jina
(embeddings/rerankers), LangGraph, or Langfuse** — nor any other external
retrieval service, paid API, or credential — without explicit Filip
approval. The existing FTS5 + Mempalace baselines above already cover
conversation and corpus recall locally and deterministically. Any proposal
to introduce such infrastructure must first justify why the baseline is
insufficient and pass a **NEEDS FILIP APPROVAL** gate (no install, no
provisioning, no credential changes in the meantime).

## Citations (quick index)

| Capability | Location |
| --- | --- |
| FTS5 session tool, no-LLM contract | `tools/session_search_tool.py:21-23`, dispatch `:378-450`, discovery `:277-375` |
| FTS5 virtual table + triggers | `hermes_state.py:290-313` (trigram `:319-343`) |
| FTS5 BM25 search query | `hermes_state.py:2154` (`MATCH` `:2212`, `ORDER BY rank` `:2209`) |
| Anchored window + bookends | `hermes_state.py:1766` (primitive `:1688`) |
| Mempalace hybrid BM25+vector | `searcher.py:2-10`, BM25 `:63-90`, vector `search()` `:294` |
| BM25/vector candidate union | `searcher.py:637-703` (entry `:748`) |
| Neighbor-chunk expansion | `searcher.py:194-243` |
| Virtual line-number provenance | `searcher.py:1023-1077` |

## Note on parent artifacts

The parent scratch workspace
`/home/filip/.hermes/kanban/workspaces/t_f555a6e8/artifacts/`
(`source-spike.md`, `closure-summary.md`) was already cleaned up at the time
this note was written, so the artifacts could not be re-read from disk. The
parent's verdict and metadata (commit, source URL, ADOPT-SELECTIVELY
verdict, follow-up id) are preserved in the `t_73dd3561` task record and are
reproduced above; all retrieval citations below were verified directly
against live source, not against the parent artifacts.
