# Nora Pinecone Memory Architecture

Tracking issue: [WAT-1368](https://linear.app/watsiai/issue/WAT-1368/nora-pinecone-memory-cloud-semantic-recall-layer).
This document is the contract for that work — code lands in follow-up
PRs and must conform to the scope, retrieval order, and trust model
locked in here.

## Purpose

Nora's long-term context is currently bottlenecked by local-only stores:
the curated `memory` (`MEMORY.md` / `USER.md` files written by
`tools/memory_tool.py`), SQLite session search via FTS5
(`tools/session_search_tool.py`), and the file-backed skills index
assembled in `agent/prompt_builder.py::build_skills_system_prompt`.
Each of those is high-trust, locally durable, and intentionally
small — `MemoryStore` defaults to a 2,200-char `MEMORY.md` budget
and a 1,375-char `USER.md` budget (see `tools/memory_tool.py:118`).

Pinecone adds a **cloud-backed semantic recall layer** that augments
those stores with embedding-similarity retrieval over older session
summaries, curated topic notes, and artifact summaries. It does **not**
replace any existing surface. The existing `MemoryStore` snapshot,
`session_search` semantics, and skill index remain authoritative;
Pinecone is consulted last and contributes a small, capped, provenance-
tagged snippet block.

The integration is implemented as a memory-provider plugin under
`plugins/memory/pinecone/`, conforming to the `MemoryProvider` ABC in
`agent/memory_provider.py`. It is therefore subject to the
"only one external provider at a time" rule enforced in
`agent/memory_manager.py::MemoryManager.add_provider` (see
`agent/memory_manager.py:204`); operators select it via
`memory.provider: pinecone` in `config.yaml`.

## v1 Non-goals

The following are explicitly out of scope for v1. Reviewers should
reject PRs that drift into them:

1. **Replacing exact retrieval for code/files.** File reads, Glob,
   Grep, and `read_file` remain the only path for source-of-truth
   code lookups. Pinecone never serves code chunks.
2. **Auto-saving every tool result.** Ingestion is opt-in and trigger-
   gated (see "Ingestion Triggers"). Tool outputs are not embedded.
3. **Mutating existing `memory` semantics.** The schema, char limits,
   `§`-delimited entry format, frozen-snapshot pattern, and
   threat-pattern scan in `tools/memory_tool.py` are unchanged.
   The Pinecone tool does not write to `MEMORY.md` or `USER.md`.
4. **Ingesting raw transcripts.** Only post-hoc summaries (session
   summary, artifact summary, curated topic notes) are embedded.
   Raw conversation messages stay in SQLite, where `session_search`
   can FTS5-rank them and the auxiliary model can summarize on
   demand.
5. **Replacing `session_search`.** `session_search` continues to be
   the recall surface for "what did we do last time?" queries and
   keeps its current FTS5 + auxiliary-summarizer flow
   (`tools/session_search_tool.py:325`). Pinecone surfaces older
   compressed _summaries_; `session_search` still returns
   freshly summarized transcripts ranked by FTS5.
6. **Replacing skills.** Procedural workflows still belong in
   `~/.hermes/skills/` and are indexed via
   `build_skills_system_prompt`. Pinecone does not store skill
   bodies.

## Memory Classes

Pinecone records carry a `class` tag that drives ingestion rules,
TTL, and retrieval weighting. v1 supports exactly five:

| Class              | Source                                                                                                                                                      | Mutability                                                    | Default TTL                  | Notes                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `profile`          | Curated user/operator facts mirrored from the local `USER.md` snapshot at session start                                                                     | Append-only mirror; deletes propagate from local              | None (until removed locally) | The local store is authoritative; Pinecone only copies entries to make them recall-searchable. Edits go through `memory_tool` first. |
| `project_context`  | Curated topic / domain notes the operator has explicitly placed in the configured ingestion directory                                                       | Operator-edited; re-indexed on content-hash change            | None (manual revoke)         | Equivalent of a small wiki. Files are markdown; chunked at heading boundaries.                                                       |
| `session_summary`  | Auto-generated summary of a closed session (one record per summary, not per message)                                                                        | Immutable once written                                        | 180 days                     | Produced by the existing summarization pipeline used by `session_search`; Pinecone stores the summary text + session metadata.       |
| `artifact_summary` | Summaries of substantial deliverables (a written doc, a generated dataset description, a long PR description)                                               | Immutable; new versions get a new record + supersedes pointer | 365 days                     | Captures "what did we build" without ingesting the artifact body.                                                                    |
| `ephemeral`        | Short-lived recall hints written by an explicit tool action (e.g. a planning scratch note the user wants the agent to remember for "the next few sessions") | Operator-revocable                                            | 14 days                      | Lowest trust; aggressively decayed in ranking and dropped on TTL.                                                                    |

Class names are persisted as Pinecone metadata (`class`) and used
as filters at query time. Adding a new class is a v2+ concern and
must update this document first.

## Ingestion Triggers

Ingestion is **trigger-driven, not blanket**. The plugin runs no
embedder unless one of these events fires:

1. **Session close → `session_summary`.** When a session is finalized
   (the same point where SQLite has the full transcript and the
   `session_search` summarizer is available), a single summary record
   is generated and embedded. Trigger fires from the existing
   `on_session_end` hook on `MemoryProvider`
   (`agent/memory_provider.py:153`). If summarization fails, no record
   is written — Pinecone never stores a partial or zero-content
   summary.

2. **Manual `memory_pinecone_ingest` tool call → `project_context`
   or `artifact_summary`.** The agent (or the operator via a slash
   command) explicitly nominates a file or paste to ingest with a
   class tag. The tool refuses to embed unrecognized classes and
   refuses to embed source files (`.py`, `.ts`, etc.) — code stays
   out of Pinecone.

3. **Background topic-folder sync → `project_context`.** A configured
   directory (e.g. `~/.hermes/nora/topics/`) is scanned periodically.
   Files are content-hashed; unchanged files are **skipped** so we
   never re-embed identical content (this is one of the parent
   ticket's acceptance criteria). Hashes are stored as Pinecone
   metadata.

4. **Profile mirror on session start → `profile`.** When the local
   `MemoryStore` loads (`tools/memory_tool.py::load_from_disk`),
   the plugin diffs the snapshot against what Pinecone has tagged
   `class=profile` for this user and reconciles: new local entries
   are upserted, locally-removed entries are deleted in Pinecone.
   The local file remains the source of truth.

5. **`memory_pinecone_note` tool call → `ephemeral`.** Explicit
   short-lived note. Off by default; gated on a config flag so it
   can be disabled in stricter deployments.

Triggers that do **not** ingest:

- Per-turn writes (would inflate the index and leak transcript
  content).
- Tool results (out of scope per non-goal #2).
- `memory_tool` writes other than the profile mirror above
  (the local store is the source of truth; we mirror, we do not
  duplicate semantics).

## Retrieval Order

The agent assembles per-turn context in **four ordered layers**.
Pinecone is always last. Earlier layers are never displaced or
truncated to make room for Pinecone results.

```
Layer 1: profile memory          (MEMORY.md + USER.md frozen snapshot)
Layer 2: session_search           (on-demand, agent-invoked tool call)
Layer 3: skills                   (skill index + skill_view loads)
Layer 4: Pinecone semantic recall (auto-prefetch, capped, last in prompt)
```

Why this order:

- **Layer 1 — profile memory.** Highest trust. Curated by the user
  and the agent's `memory_tool` writes. Injected as a frozen snapshot
  by `prompt_builder` (`run_agent.py:5009`) to keep the prefix cache
  stable. Pinecone never edits these files.
- **Layer 2 — `session_search`.** Agent-invoked when the user
  references prior conversations. Returns auxiliary-model summaries
  of FTS5-matched sessions (`tools/session_search_tool.py:325`).
  Pinecone may _complement_ this by surfacing older summaries the
  agent did not explicitly query for, but it does not replace
  the search semantics: the tool, its schema, its ranking, and its
  three-result default all stay as they are.
- **Layer 3 — skills.** Procedural knowledge with explicit
  load-on-relevance semantics, indexed in
  `build_skills_system_prompt` (`agent/prompt_builder.py:718`).
  Skills carry imperative instructions; Pinecone records do not.
- **Layer 4 — Pinecone.** Soft, similarity-ranked, capped at
  **3–4 snippets** in the prompt. Each snippet is wrapped in
  `<memory-context>` fencing per the existing convention in
  `agent/memory_manager.py::build_memory_context_block`
  (`agent/memory_manager.py:173`) so the model treats it as
  background reference, not user input.

If a higher layer already contains the answer, Pinecone results
are **not** added — the orchestrator deduplicates against text
already present in earlier layers.

## Provenance and Staleness Rules

Every Pinecone record carries metadata sufficient to rebuild the
provenance line shown to the model:

- `class` — one of the five classes above.
- `source_path` — original file or session ID (no raw transcript
  content; just an identifier).
- `created_at` — ISO timestamp.
- `last_seen_at` — last time the source was confirmed live (re-ingest
  bumps it; missing files mark stale).
- `content_hash` — SHA-256 of the embedded text. Used to skip
  re-embedding unchanged sources.
- `supersedes` — optional pointer to a prior record this replaces.
- `ttl_days` — class default unless overridden.

Retrieval format (rendered into the prompt before fencing):

```
[pinecone | <class> | <source_path> | <age in days>]
<snippet text>
```

Staleness rules — applied in this order:

1. **Hard TTL.** Records older than `ttl_days` are filtered out at
   query time and deleted by the background sync.
2. **Conflict with local canonical sources.** If Pinecone returns a
   `profile` snippet whose content is not present in the current
   local `USER.md` snapshot, it is **dropped**, not surfaced. Local
   wins. The reconciliation pass on session start should make this
   rare; the runtime check is the safety net.
3. **Conflict with newer summaries.** If two `session_summary`
   records cover overlapping time windows, the newer one ranks
   higher and the older one is filtered when scores tie.
4. **Age decay.** Similarity scores are multiplied by an age-decay
   factor so a fresh `session_summary` can outrank a stale, slightly
   more similar `project_context` note. The exact factor is a v1
   config knob, not a constant in this doc.
5. **`supersedes` follow-through.** If the top hit has been
   superseded, the orchestrator follows the pointer and surfaces
   the successor instead.

The contract is: **stale memories never outrank fresher local or
canonical sources** (parent-ticket acceptance criterion).

## Fail-Open Behavior

Pinecone is a soft layer. Hermes must remain fully functional when
the integration is disabled, mis-configured, rate-limited, or down.

- **Disabled by config.** If `memory.provider` is not `pinecone`,
  the plugin is never loaded and no tools are exposed. Layers 1–3
  behave exactly as today.
- **Init failure.** If `is_available()` returns False or
  `initialize()` raises, the plugin logs a warning and is unregistered.
  `MemoryManager` swallows per-provider failures already (see the
  try/except wrappers in `prefetch_all` / `sync_all` /
  `build_system_prompt`); we conform to that contract.
- **Query failure or timeout.** `prefetch()` returns an empty string
  on any error. The agent proceeds with layers 1–3 only. There is
  no retry-storm; we log at debug and move on. This matches the
  existing "failures in one provider don't block others" comment in
  `agent/memory_manager.py::prefetch_all` (`agent/memory_manager.py:285`).
- **Embedding failure during ingestion.** Records are not written;
  the trigger logs and gives up for that item. The next trigger
  fires normally.
- **Empty / pre-wrapped responses.** The output goes through
  `sanitize_context` and `build_memory_context_block` like every
  other provider, so a misbehaving response cannot inject
  `<memory-context>` fences or system notes.
- **Tool-schema unavailability.** If credentials are missing, the
  plugin does **not** register `memory_pinecone_ingest` or
  `memory_pinecone_note`. The schema surface stays minimal, matching
  the existing "lazy registration" pattern in other tools.

The user-visible failure mode is "no Pinecone hits this turn,"
never a hard error.

## Cross-Check Against Existing Hermes Surfaces

Reviewed for regression risk against the four files named in the
ticket scope:

- **`AGENTS.md`** — describes the memory surface only at a high
  level (`memory` toolset, plugin discovery system, the
  one-external-provider rule). This architecture conforms: Pinecone
  ships as `plugins/memory/pinecone/`, registers via the existing
  `MemoryProvider` ABC, and respects the
  `plugins/memory/<name>/cli.py` command-registration contract.
  No edits to core files (`run_agent.py`, `cli.py`, etc.) are
  required, satisfying the Teknium May-2026 plugin rule recorded
  at `AGENTS.md:509`.
- **`agent/prompt_builder.py`** — system-prompt assembly is
  untouched. Pinecone does not contribute a static
  `system_prompt_block()`; its content arrives via `prefetch()` and
  is fenced. The frozen `MemoryStore` snapshot still composes the
  cached prefix.
- **`tools/memory_tool.py`** — schema, char limits, atomic write
  semantics, frozen-snapshot pattern, and `_scan_memory_content`
  all unchanged. The profile mirror is read-only against the local
  store; Pinecone never calls into `MemoryStore.add` /
  `replace` / `remove`.
- **`tools/session_search_tool.py`** — schema, FTS5 query, auxiliary
  summarization model, parent-session resolution, and the
  hidden-source filter are unchanged. Pinecone cannot be selected
  via `session_search`; the only overlap is that
  `session_summary` Pinecone records are produced from the same
  summarization pipeline (post-session, not per-turn).

No regressions to memory compactness (the local stores keep their
existing char budgets) and no semantic shift in `session_search`.

## Open Questions (for v2+)

- Whether the embedder abstraction should support multiple backends
  on day one (OpenAI text-embedding-3-large, a local model). v1
  should ship with one but keep the embedder behind an interface
  so this is a config swap later.
- Per-workspace partitioning vs. per-user partitioning of the
  Pinecone namespace. v1 will partition per `agent_workspace`
  passed to `initialize()` (`agent/memory_provider.py:78`).
- UI affordances for showing the user _which_ Pinecone records
  influenced a turn. Out of scope for v1; provenance is in metadata
  and visible via the admin tool.

## Phased Rollout

Tracked in WAT-1368; documented here for reference so reviewers
know which phase any given PR belongs to:

1. **Dark launch.** Plugin registers, ingestion runs, but
   `prefetch()` returns empty. Verify ingestion correctness and
   index health.
2. **Debug retrieval.** `prefetch()` returns content, but only
   logged — not injected into the prompt. Verify ranking, decay,
   provenance.
3. **Limited prompt recall.** Inject capped (≤3) snippets for
   opted-in profiles only. Watch for prompt-cache regressions.
4. **Broad enablement.** Default-on for users with credentials.
   Cap remains 3–4 snippets.

Each phase gate requires a green full test suite (parent-ticket
acceptance criterion) before advancing.
