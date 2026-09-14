# Knowledge Capability Audit

**What the runtime already provides for knowledge, retrieval, memory, documents,
embeddings, search and tool extension — which of it an agent can actually call — and where
the NOVA boundary falls.**

No core modification. The boundary is proposed here and proven before anything is built.

The test applied throughout is not *"does code exist"* but *"is it registered in the tool
registry and reachable by a model"*. Every claim is a file and line read at HEAD `bdc6202`.

---

## 0. The headline

**Every agent-callable search in this runtime searches the operator's own material — their
files, their past conversations, the web. None searches a business corpus.**

The registry's whole knowledge-adjacent surface is:

| Tool | What it searches | Useful as business knowledge? |
|---|---|---|
| `search_files` | The filesystem, grep-shaped | No — not a corpus |
| `session_search` | Past conversations (FTS5 + BM25) | No — and it *hides* worker sessions |
| `read_file` | One file, with document extraction | Partly — the extractor is reusable |
| `memory` | Curated `MEMORY.md` / `USER.md` | No — personal notes, char-budgeted |
| `web_search`, `web_extract`, `x_search` | The public web | No |
| `skills_list`, `skill_view` | Procedural skills | Partly — wrong shape for reference docs |

There is **no ingestion, no chunking, no embedding pipeline and no corpus index** anywhere
in core. That part of the Phase 0 audit still holds.

What *has* changed since Phase 0 is that the reusable pieces are now identified precisely,
and one of them removes most of the risk from building the rest.

---

## 1. Search infrastructure

### `session_search` — the closest thing, and the wrong scope

`tools/session_search_tool.py`: FTS5 over the session database, BM25-ranked, deduped by
lineage, with four modes inferred from its arguments and **no LLM calls** — every shape
returns real rows.

Two details make it unusable as business knowledge, and both are deliberate:

- It **hides** `kanban`, `subagent` and `tool` sessions: *"not the user's history"*. A NOVA
  agent's own work is invisible to it by design.
- It is scoped to conversations, not documents.

**Reuse the technique, not the tool.**

### The FTS5 layer is genuinely good, and genuinely session-bound

`hermes_state_fts.py` builds `messages_fts_cjk` — a virtual table over messages, with
triggers, a tokenizer-capability fallback, and corruption detection. The schema is specific
to messages; the *patterns* are what transfer.

One asset is portable as-is: **`native/fts5_cjk/fts5_cjk.c`**, a ~250-line loadable SQLite
FTS5 tokenizer with no dependencies, shipped with a build script. It re-emits CJK runs as
overlapping bigrams because the stock trigram tokenizer needs three characters and starves
1–2 character Korean and Chinese terms. A customer with CJK documents would otherwise get
quietly poor recall.

It must be compiled, and the runtime already degrades gracefully when it cannot load.

### `tool_search` — not reusable

`tools/tool_search.py` implements BM25 with Snowball stemming, but over *tool definitions*
for progressive disclosure. It is coupled to tool schemas and token budgets. Reading it is
worthwhile; importing it is not.

---

## 2. Documents

**`tools/read_extract.py` is the single best reuse candidate in this audit.**

A clean module with a real `__all__` — `EXTRACTABLE_EXTENSIONS`, `ExtractionError`,
`extract_document_bytes` — and no coupling to the tool that calls it.

| Always available (stdlib) | Via `firecrawl-anydoc` |
|---|---|
| `.ipynb`, `.docx`, `.xlsx` | `.doc`, `.xls`, `.xlsm`, `.xlsb`, `.odt`, `.ods`, `.odp`, `.rtf`, `.epub`, `.pdf` |

`firecrawl-anydoc` is a **core dependency**, not optional — the maintainers moved it in
because "PDF reads are a common first-session action". So in any real deployment the full
format set is present.

Malformed documents raise a typed `ExtractionError` rather than returning junk, which is
what an ingestion pipeline needs.

**This is the piece NOVA should not rewrite.** Re-implementing PDF and legacy-Office
extraction is weeks of work and a permanent maintenance liability.

---

## 3. Memory

Three layers, unchanged from Phase 0, and **none of them is a knowledge store**.

`tools/memory_tool_store.py` is file-backed `MEMORY.md` / `USER.md` with character budgets
and injection scanning. Deliberately small, deliberately personal.

### The memory provider interface is the wrong seam — this is the key finding

`agent/memory_provider.py` is a real extension point with eight shipped backends, a
declarative config schema, and a generic settings UI. It can even expose agent-callable
tools through `get_tool_schemas()`.

NOVA knowledge should still **not** be a memory provider. Four reasons, each read from the
contract:

1. **"ONE external provider at a time."** Registering knowledge as the memory provider
   would displace whatever the customer chose for memory. Knowledge and memory are not
   alternatives.
2. **`prefetch` runs every turn.** A business corpus should be queried when an agent needs
   it, not injected into every request.
3. **`sync_turn` writes the conversation back.** Authoritative business documents must be
   read-only; a handbook that absorbs chat is no longer a handbook.
4. **No per-agent source scoping.** Phase 0 required that not every agent sees every
   document. The provider interface has no place to express it.

Worth noting what the interface gets right, because NOVA must respect the same rule:
`system_prompt_block` is explicitly **static**, with recall routed through `prefetch`
instead. That is the prompt-caching invariant the root `AGENTS.md` calls *"sacred"* —
anything that rebuilds the system prompt mid-conversation multiplies the customer's cost.
**A knowledge tool must return results into the tool-result channel, never into the system
prompt.**

---

## 4. Embeddings

**Absent from core.** No `sentence_transformers`, no `embeddings.create`, no vector store.

One useful discovery: the bundled proxy allowlists `/embeddings` for its provider adapters
(`hermes_cli/proxy/adapters/nous_portal.py:29`, `xai.py:19`). So a deployment that already
routes inference through the proxy can reach an embedding endpoint **without NOVA adding a
provider dependency or a second credential path**.

That is a route for a future vector phase, not a reason to start with one.

---

## 5. Tool extension — the seam NOVA should build on

A plugin declaring `provides_tools` in `plugin.yaml`, with a `tools.py` beside it, registers
**agent-callable tools** through `PluginContext.register_tool()`
(`hermes_cli/plugins.py:449` → `registry.register()` at `:476`).

The contract is strict in exactly the right places:

- Opting in is explicit: the manifest names the tools; a stray `tools.py` does not opt a
  plugin in by accident.
- Registering a **new** tool name needs no special permission.
- **Overriding a built-in** requires an operator opt-in
  (`plugins.entries.<id>.allow_tool_override: true`), so no plugin can silently replace
  `write_file`.
- A declared tool with no `tools.py` logs a warning rather than vanishing quietly.

Combined with what Phase 2 already proved — plugin discovery scans `<home>/plugins`, and a
worker's home *is* its profile — this gives NOVA:

- an agent-callable `knowledge_search` tool,
- installed per agent,
- with per-agent corpus scoping for free,
- and **zero core patches**.

It is the same mechanism as the policy plugin, with `provides_tools` instead of `hooks`.

### MCP — the other route, and the one to prefer where it fits

Per-profile `mcp.json` already resolves secrets from the active profile's scope. Sixty-five
integrations ship, including `notion`, `algolia` and `context7`.

**Where a customer's knowledge already lives behind an MCP server, NOVA should connect to
it rather than ingest a copy.** A synchronised duplicate of a customer's wiki is a
liability: it goes stale, and it doubles the places their data sits.

### Skills — procedural knowledge, not reference

58 bundled skills with progressive disclosure and a curator. Skills are *loaded into
context*, which is right for "how to run a refund" and wrong for a 200-page handbook.
NOVA should keep skills for procedure and not force reference documents through them.

---

## 6. The reuse / build boundary

### Reuse from the runtime

| Asset | Why |
|---|---|
| `tools/read_extract.py` | Weeks of format handling, already a core dependency, typed errors |
| `native/fts5_cjk` tokenizer | Real multilingual recall; degrades gracefully when absent |
| The plugin `provides_tools` seam | Agent-callable tools, per-agent, zero core patches |
| Per-profile `mcp.json` | Connect to knowledge in place rather than copying it |
| SQLite + FTS5 patterns | The storage layer is already there and already backed up |

### Build in NOVA

| Component | Why it cannot be reused |
|---|---|
| Ingestion pipeline | Nothing in core ingests |
| Chunking | Nothing in core chunks |
| Corpus index and retrieval | The only FTS index is over messages |
| Per-agent source scoping | No existing interface expresses it |
| Provenance and citation | Grounding needs a document identity the runtime has no concept of |
| Injection scanning on retrieved text | Must reuse `threat_patterns`, but the retrieval path is NOVA's |

### Do not build

Embeddings or a vector store in a first phase. FTS5 plus the CJK tokenizer covers keyword
retrieval over business documents, and it ships, backs up and restores with the storage
already in place. Add vectors when measured recall proves keyword search insufficient — not
before.

---

## 7. The one boundary that is not yet proven

NOVA's ingester needs `tools/read_extract.py`, and **`nova/**` may not import a runtime
module** — a rule enforced by `tests/platform/test_boundaries.py`.

Three ways to resolve it, honestly stated:

1. **Extraction runs inside the runtime, in the installed plugin payload.** The payload is
   an artifact NOVA *ships into* the runtime, not NOVA code, so it may use the plugin
   contract — the same category as Phase 2's `_decide.py`. Requires a narrow, documented
   exemption in the boundary test for payload files.
2. **NOVA shells out** to the runtime for extraction. Keeps the import rule pristine; costs
   a subprocess per document and a fragile interface.
3. **NOVA extracts only text and Markdown itself** in the first phase, deferring binary
   formats. Cleanest boundary, smallest capability.

**Recommendation: (1), with the exemption written as a rule rather than an exception** —
*payload files shipped into the runtime may use the runtime's plugin contract; nothing else
under `nova/` may.* That keeps the dependency arrow intact for the platform proper while
letting the artifact that runs inside the runtime behave like what it is.

This needs a decision before the knowledge phase starts. It is the only place where the
existing boundary does not already answer the question.

---

## 8. Recommended shape for the knowledge phase

```
nova/knowledge/                     platform-owned, runtime-agnostic
  sources.py       declaration: what a corpus is, per tenant and per agent
  chunk.py         splitting, with provenance preserved
  index.py         FTS5 corpus index (its own database, not the runtime's)
  ingest.py        orchestration; extraction delegated per §7

nova/runtime/hermes/
  knowledge_tool.py   the payload: plugin.yaml provides_tools + tools.py,
                      installed per agent, scoped to that agent's sources
```

Retrieval returns into the tool-result channel with citations, never into the system
prompt. Every retrieved chunk passes `threat_patterns` before it reaches a model — customer
documents are the same injection vector as curated memory, and the runtime already treats
that seriously.

**No Hermes core file is modified.** The patch budget stays at one.
