# Phase 4 — NOVA Knowledge

Declared corpora, indexed by NOVA, searched by an agent through a scoped tool.
**Core patches: still 1** (the `AGENTS.md` routing row). No Hermes file was modified.

This follows the shape committed to in
[`KNOWLEDGE_CAPABILITY_AUDIT.md`](KNOWLEDGE_CAPABILITY_AUDIT.md) §8, with one boundary
resolved better than the audit proposed — see [Extraction](#extraction-a-borrowed-capability).

---

## What was built

```
nova/knowledge/                    platform-owned, runtime-agnostic
  sources.py     what a corpus is: root, globs, size cap, classification — and the walk
  chunk.py       splitting, with provenance preserved on every chunk
  index.py       the FTS5 corpus index, in NOVA's own database
  ingest.py      orchestration; extraction delegated to the runtime contract
  query.py       question -> safe FTS5 expression. COPIED VERBATIM into the plugin

nova/runtime/hermes/
  extract.py             borrows the runtime's document extractors, lazily
  knowledge_tool.py      the payload: register(ctx) + knowledge_search
  knowledge_manifest.yaml
```

One tenant file (`knowledge.yaml`), one grant per agent (`knowledge.sources`), one index
per deployment, one tool per granted agent.

---

## The three properties that matter

### 1. Scope is a SQL predicate, never an instruction

An agent's readable corpora are resolved at materialization time into
`<profile>/nova-knowledge.json` and applied as `AND source_id IN (?, ?)`. Nothing in the
tool's arguments can widen them.

A model asking for a corpus it was not granted is **not refused** — it is told which
corpora it does have. A refusal invites negotiation; a filter does not.

Two rules in `nova/knowledge/query.py` carry most of the weight:

- **Empty scope returns nothing, never everything.** Treating "no corpora granted" as "all
  corpora" is the single failure that turns a scoping bug into a disclosure incident.
- **The question never reaches the FTS5 parser intact.** Everything a model types is
  tokenized into words and quoted phrases and the expression is rebuilt from parts the
  module chose. `NEAR(a b)`, `source_id:*`, `^refund`, a stray apostrophe: all become
  ordinary search terms.

### 2. Retrieved text is untrusted, and the runtime will not treat it as such for us

The runtime's untrusted-tool-result wrapping is keyed to a hard-coded list of tool names
(`web_extract`, `web_search`, `browser_*`, `mcp_*` — `agent/tool_dispatch_helpers.py:436`).
A plugin tool is not on it and gets **neither the threat scan nor the delimiters**.

Naming the tool `mcp_knowledge` to inherit that treatment would be a lie about what it is,
so the payload applies the wrapping itself — same delimiter, same warning text, so the
model meets one consistent boundary rather than a second dialect of the same idea. It also:

- defangs embedded `untrusted_tool_result` tokens, so a document cannot close the boundary
  early by containing the closing tag;
- scans every passage with the runtime's own `threat_patterns` and **annotates rather than
  drops**. A customer's security handbook legitimately contains the phrase "ignore all
  previous instructions", and deleting the document that explains the attack is worse than
  labelling it.

### 3. Model-visible means logged

Two new kinds in `MODEL_VISIBLE_KINDS`, both write-ahead:

| Kind | When | Written by |
|---|---|---|
| `knowledge.indexed` | a corpus's contents change | `nova/knowledge/ingest.py` |
| `knowledge.granted` | an agent's readable corpora change | materialization |
| `knowledge.search` | an agent retrieves | the installed plugin |

The search record carries **citations, not passages**. Where the agent read is the
governance question and is cheap to store; copying customer documents into a second file,
with different permissions and a different retention story, is a liability rather than an
improvement. A search naming an ungranted corpus is recorded under phase `refused` — either
a stale prompt or something probing for what else exists, and both are worth seeing.

---

## Extraction: a borrowed capability

The audit flagged one open boundary: NOVA must not import the runtime, but the runtime owns
extractors for PDF, Office and OpenDocument that would be weeks of work and a permanent
liability to reimplement.

Resolved by putting extraction **on the contract** rather than granting the boundary test
an exemption:

- `AgentRuntime.extract_text()` — defaults to UTF-8 text, so a runtime with no extractor
  still ingests plain text and markdown.
- `nova/runtime/hermes/extract.py` — a **lazy, function-level** import of
  `tools.read_extract`. Inside that function the runtime is, by definition, installed.

`tests/platform/test_boundaries.py` now enforces exactly that distinction:

| Rule | Scope |
|---|---|
| No module-level runtime import | everywhere, adapters included |
| Lazy runtime import allowed | only inside `nova/runtime/<adapter>/` |
| Adapter still imports with no runtime present | asserted in a subprocess |

The third is what keeps the second honest.

---

## Retrieval: BM25, and why it stops there

FTS5 with the `porter` stemmer, ranked by `bm25`. No embeddings anywhere. That is a
deliberate stopping point, not an unfinished one:

- BM25 over a few thousand well-chunked enterprise documents is genuinely good.
- It is exact and explainable when a customer asks why a document ranked where it did.
- It adds no model provider, no vector store, no dimension-compatibility problem, and no
  re-embedding cost on every ingest.

Semantic retrieval is a real improvement for some corpora and belongs behind the same
interface later. Shipping it first would have meant shipping a dependency stack before
shipping retrieval. NOVA still depends only on the standard library and PyYAML.

**One caveat worth stating plainly:** BM25 scores are corpus-relative. They are not a
percentage, not a confidence, and not comparable across corpora — nothing should threshold
on them. On a small corpus every term is near-universal, so real scores land around `1e-06`;
they are reported to four *significant* figures for that reason.

---

## Safety at the walk

`iter_documents` is the security boundary — everything downstream assumes the file it is
handed is one the tenant meant to publish. It refuses rather than guesses:

| Refused | Why |
|---|---|
| Symlinks | A link planted in a documents folder is the cheapest way to read a private key into a searchable corpus |
| Paths resolving outside the root | Same, by a different route |
| Dot-directories, `node_modules`, `.venv`, build output | Never customer knowledge; each can contribute tens of thousands of files |
| Files over `max_file_bytes` (default 8 MiB) | A stray database dump fails loudly instead of filling the index |

Every refusal is reported in the ingest report with its reason. Nothing is skipped silently.

`**/*.md` is expanded before matching: `fnmatch` has no `**`, so the unexpanded pattern
behaves as `*/*.md` and silently misses every file at the corpus root. The only symptom
would have been a smaller number in the ingest report than expected.

---

## Lifecycle

Ingestion is incremental by content hash and **prunes in the same pass**. A document that
has disappeared from disk is dropped from the index immediately: a corpus that keeps
serving a withdrawn document is a disclosure nobody authorised.

A corpus left in the index after being undeclared stays searchable by any agent whose grant
was not re-applied. The control plane surfaces that as `undeclared_in_index` and the
dashboard raises it as a problem banner rather than filtering it out.

---

## Surfaces

```
nova knowledge ingest <bundle> [--source ID] [--force] [--dry-run]
nova knowledge status <bundle> [--json]
nova knowledge search <bundle> "question" [--as-agent ID] [--source ID] [--limit N]
```

`--as-agent` rehearses one agent's real scope — the check worth running before a rollout.
It *intersects* with `--source` rather than replacing it, so it can never reach past the
grant it is rehearsing.

`GET /platform/v1/knowledge` returns declared corpora, index counts, and which agents can
read each. The dashboard's Knowledge panel puts "readable by" in a column of its own: it is
the fact a reviewer came for, and `nobody` is stated explicitly, because a corpus that is
indexed and unread is a cost with no benefit.

---

## Known limitations

- **No semantic search.** Keyword retrieval only; a question sharing no vocabulary with the
  document will not find it.
- **One index per deployment.** Agents are separated by the query-time scope filter, not by
  separate databases. That is the right trade for one tenant per home; it is not
  multi-tenancy, which NOVA still does not claim.
- **`known_plugin_toolsets`.** Plugin toolsets are on by default
  (`hermes_cli/tools_config.py:535`), which is why no `platform_toolsets` write is needed.
  But that key is written on every config save: if a customer saves their toolset
  configuration while the plugin is loaded, `nova_knowledge` becomes "known", and being
  absent from the saved list would then turn it off. Documented rather than worked around.
- **Extraction quality is the runtime's.** An image-only PDF needs OCR; the ingest report
  says so per file rather than producing an empty document.
- **No access control below the corpus.** A grant is per corpus, not per document. Split
  the corpus if a subset needs different readers.

---

## Tests

`tests/platform/test_knowledge.py` (42) — declaration, the walk's refusals, chunk
provenance, hostile queries, incremental ingest, pruning, the audit invariant.

`tests/platform/test_knowledge_tool.py` (19) — the plugin **as installed**, loaded from a
materialized profile rather than the source tree, including: the manifest parsing under the
runtime's own loader, scope that cannot be widened through any argument, the trust boundary
that cannot be closed early, and the copied modules staying byte-identical to their
originals.
