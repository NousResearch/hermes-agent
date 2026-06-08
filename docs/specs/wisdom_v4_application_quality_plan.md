# Hermes Wisdom v4 Application Quality Plan

## Objective

Wisdom v4 improves the most important product step in the simple PKM loop:

```text
capture exact words -> retrieve -> review -> apply
```

The v4 goal is better, domain-aware application output when the user asks what
to do with an idea. It should make saved fragments more reusable without turning
Wisdom into a notes app, graph database, dashboard, scheduler, or integration
platform.

## 1. What v1/v2/v3 Already Provide

v1 provides the local Wisdom kernel:

- exact original preservation in `raw_events.original_text`
- SQLite storage owned by `wisdom/`
- deterministic capture, classification, retrieval, interpretation, and
  application proposals
- secret-like capture blocking
- `/wisdom` commands
- fail-open gateway behavior
- no writes to `~/.hermes/productivity/productivity.db`

v2 provides Hermes-native model tools:

- `tools/wisdom_tool.py`
- shared command/tool service layer in `wisdom/service.py`
- the `wisdom` toolset and default tool exposure
- natural-language affordances through tool descriptions rather than a regex
  router

v3 provides the manual quality loop:

- review queues
- deterministic quality scores
- related captures without embeddings
- accept, dismiss, archive, and applied statuses
- improved but still generic deterministic application templates

## 2. Why v4 Focuses on Application Quality

More capture sources would increase volume before Wisdom proves that saved ideas
can change decisions. Voice capture, imports, cron review, dashboards,
embeddings, and external integrations are all lower leverage than making
`wisdom_apply` useful.

The product bottleneck is now:

```text
"I saved this exact thought. What should I do with it?"
```

v4 therefore focuses on high-quality deterministic transformations for the four
existing categories:

- business/x10x
- investing
- health
- life

## 3. What Will Change in `wisdom_apply` / Service Layer

Keep the existing application table and application types. Do not add a new
table or ontology.

Change `wisdom/apply.py` so proposals are more specific by category:

- business: client language, operating principle, report/process task proposal
- investing: investment rule, checklist, risk heuristic or decision rule
- health: health experiment, personal rule, tracking question or decision
  boundary
- life: principle, reflection prompt or writing seed, decision rule

The service API can continue to return all proposals or filter by
`application_type`. The `context` parameter remains guidance for the model's
final response; durable proposals stay deterministic and are never written by a
live model.

## 4. LLM-Assisted Generation Decision

Initial v4 implementation used deterministic templates because Wisdom did not
yet have a safe service boundary for model-written proposals.

Follow-up implementation adds that boundary:

- `wisdom.application.mode: deterministic | llm`
- `auxiliary.wisdom_apply` as the configurable model slot
- LLM output must be JSON with validated application types and bounded bodies
- output is stored only as internal application proposals
- invalid output, model errors, timeouts, or missing credentials fall back to
  deterministic templates
- tests mock the LLM call and never require live model access

The model still never writes SQL or modifies the exact original.

## 5. Source/Context Metadata Handling

Use existing metadata fields only:

- `raw_events.metadata_json`
- `captures.metadata_json`

Preserve exact original text regardless of metadata extraction.

Support lightweight patterns without making capture syntax rigid:

- `Source: Acquired - Costco episode`
- `Context: trust, low-price positioning`
- `Podcast note: Acquired - Costco episode.`
- `Book note: Poor Charlie's Almanack.`

The parser should inspect only the first few lines, extract short non-secret
source/context values when obvious, and otherwise do nothing. It may infer
`source_type` for simple podcast/book/article/meeting forms, but user/tool
provided explicit `source_type` still wins. No complex ontology, source table,
or grammar is added.

## 6. Tool Descriptions To Improve

Improve `tools/wisdom_tool.py` descriptions so Hermes/Codex naturally chooses:

- `wisdom_capture` for "remember this", "save this", "podcast note",
  "book note", "investing thought", "business idea", and source-aware capture
- `wisdom_search` for "find that idea", "what have I said about", and saved
  idea recall
- `wisdom_original` for "show exact wording" and exact original retrieval
- `wisdom_apply` for "turn that into client language", "make this a checklist",
  "make this an investment rule", "turn this into a health experiment", "make
  this a decision rule", and "apply this to x10x"
- `wisdom_review` for "what should I review"
- `wisdom_accept` and `wisdom_dismiss` for "accept that" and "dismiss that"

Do not add a regex router.

## 7. Tests / Golden Evals

Add deterministic application-quality fixtures under:

```text
tests/wisdom/fixtures/application_quality_cases.yaml
```

Add tests proving:

- exact original preservation
- expected category/source type
- source/context metadata extraction where present
- expected application types
- domain-specific phrases in application output
- low-value notes are not over-promoted
- tool descriptions contain the natural trigger phrases v4 depends on
- no old productivity DB writes
- no live model calls

Existing v1/v2/v3 Wisdom tests must continue to pass.

## 8. Explicitly Out of Scope

Do not implement:

- voice capture
- scheduled reviews or cron
- embeddings/vector search
- Apple Notes, Readwise, Notion, Obsidian, dashboard, export, or cloud sync
- external tasks/reminders
- automatic capture of all conversations
- MCP
- theme graph or complex ontology
- new application tables or accepted-output models
- live Telegram sends in tests
- writes to the old productivity database

## 9. Deviations Required By Repo Reality

- `toolsets.py` already has unrelated local productivity changes in this
  worktree. v4 should not edit that file unless a Wisdom registration bug is
  discovered.
- `docs/wisdom_kernel.md` already contains some v3-era "What Wisdom Is Now"
  wording. v4 should update it in place instead of replacing the whole document.
- Because the running gateway imports tool modules at process start, deployed
  tool-description/code changes require a gateway restart to affect a live
  launchd gateway, but this implementation should not restart Hermes.
