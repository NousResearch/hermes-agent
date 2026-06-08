# Hermes Wisdom Product & Architecture Review

## Executive Verdict

Verdict: continue with corrections.

The direction is broadly right: v1 created a conservative local kernel, and v2 made the kernel available through Hermes' native tool loop instead of faking natural language with regexes. That is the correct architectural move for a Hermes/Codex-backed product.

But the current system is not yet a good Wisdom product. It is solid infrastructure for capture and recall, with thin deterministic interpretations and generic application proposals. It can store notes. It can retrieve notes. It has not yet proved that it compounds thinking, improves decisions, resurfaces the right ideas at the right time, or produces genuinely useful applications for business/investing/life/health.

The next build should not be voice, embeddings, dashboard, or more integrations. The next build should prove and improve the core loop: capture -> retrieve -> review -> apply -> keep/ignore. Add a rigorous local eval/smoke harness and a better manual review/application workflow before adding more surfaces.

## What We Have Built

### v1 Summary

v1 added a source-backed `wisdom/` package with:

- `~/.hermes/wisdom/wisdom.db` as a fresh SQLite store.
- exact originals in `raw_events.original_text`.
- deterministic trigger/classification in `wisdom/classify.py`.
- secret-like capture blocking in `wisdom/redaction.py`.
- SQLite/WAL/migration/FTS code in `wisdom/db.py`.
- `/wisdom` commands in `wisdom/commands.py`.
- gateway-safe wrappers in `wisdom/integration.py`.
- deterministic interpretation and application proposal generation in `wisdom/interpret.py` and `wisdom/apply.py`.

v1 correctly avoided the old productivity DB and external actions.

### v2 Summary

v2 added:

- `tools/wisdom_tool.py` with native Hermes model tools.
- `wisdom/service.py` as a shared command/tool service layer.
- `wisdom` toolset registration in `toolsets.py`.
- inclusion in `_HERMES_CORE_TOOLS`, so CLI and Telegram default toolsets see Wisdom.
- `wisdom` in `hermes_cli/tools_config.py`.
- native tool tests in `tests/wisdom/test_native_tools.py`.

The intended runtime path is now:

natural message -> Hermes agent -> Codex sees Wisdom tools -> model calls Wisdom tool -> deterministic Wisdom service reads/writes SQLite -> model replies.

### What Works

- The kernel boundaries are sane: DB logic is in `wisdom/db.py`; model-facing wrappers are thin.
- Exact-original preservation is correctly prioritized.
- Secret-like captures are blocked rather than silently altered.
- `/wisdom` commands and native tools share the service layer.
- Gateway edits are minimal and fail open.
- The implementation uses Hermes' existing native tool registry, not a new subsystem.
- Tests cover DB creation, exact preservation, secret blocking, command behavior, gateway fail-open behavior, tool registration, schema shape, and dispatcher-level tool calls with temp DBs.

### What Is Still Not Proven

- Whether Codex/GPT-5.5 will reliably call Wisdom tools in live conversation.
- Whether the model will over-call search or under-call capture.
- Whether tool descriptions are sufficient under a large default toolset.
- Whether FTS search is good enough after hundreds or thousands of captures.
- Whether application proposals are useful enough to change behavior.
- Whether review output is useful or just a list.
- Whether users will develop a habit around the system.
- Whether the running gateway's toolset/cache behavior matches tests after restart.

## Product Assessment

### Strengths

- The product starts with the right primitive: preserve the user's exact words.
- Natural-language tool use is a better interface than command-heavy UX.
- Commands remain available for debugging and power use.
- Local-first storage fits the privacy expectations of a personal thinking system.
- The split between original, cleaned text, interpretation, and application is correct.
- The user domains are clear enough for v1/v2 categories: business, investing, health, life, inbox.

### Weaknesses

- Current "apply" output is mostly templated restatement. It is structurally safe but not yet valuable.
- Review is shallow: counts, recent captures, unapplied captures. It does not synthesize, challenge, cluster, or help the user decide what matters.
- Search is lexical only. That is acceptable for v1/v2, but it will miss paraphrased recall.
- There is no capture triage. Everything captured is treated similarly unless archived.
- There is no explicit accept/dismiss loop for application proposals even though the schema has application statuses.
- There is no product telemetry/eval loop to learn from real usage.
- "Wisdom" as a name creates a high expectation; the current behavior is closer to "local thought capture and recall."

### Likely User Friction

- The user may say "turn that into client language" after a normal conversation, but if no capture/search result with an ID is in context, the model may not know what "that" refers to.
- The user may expect natural search to find paraphrases; FTS will often need exact-ish terms.
- The user may not trust that something was saved unless the response clearly includes a capture ID.
- The user may capture many notes but not review them. That creates note hoarding.
- Application proposals may feel too generic to be worth asking for.
- Adding 10 Wisdom tools to the default toolset may slightly increase tool-selection confusion.

### Does It Serve The User's Use Case?

Partially.

For "save this exact thought and retrieve it later," yes. For "make Hermes a compounding personal intelligence system for business/investing/life/health," not yet. The missing product loop is selection and transformation: which notes matter, what patterns repeat, what should become a principle/checklist/client phrase/decision rule, and what should be discarded.

The minimum lovable next feature is not more capture. It is a sharper review/apply loop that helps the user turn saved fragments into reusable assets.

## PKM / Knowledge-System Assessment

### Capture

Capture is intentionally low-risk and explicit. That is good. The system avoids automatic capture of ordinary chat, which prevents a noisy database.

The weakness is that capture is still dependent on either explicit trigger phrases or the model choosing `wisdom_capture`. Live behavior needs evals because natural-language capture is only as good as the model's tool selection.

### Retrieval

Retrieval is source-backed and honest: `wisdom_search` searches stored originals, cleaned text, interpretations, and applications. `wisdom_original` returns exact wording.

FTS/LIKE is enough for early use. It is not enough for mature PKM. Real users remember meanings, not exact tokens. Search will eventually need either high-quality query expansion, embeddings, or model-assisted reranking. That should wait until there is enough real data to evaluate.

### Review

Current review is underpowered. It lists counts and recent/unapplied captures. Serious PKM depends on resurfacing, progressive summarization, and periodic decisions. The current system has storage, not compounding.

### Linking and Related Ideas

There is no related-ideas feature, no backlinks, no themes, and no recurring pattern detection. That is acceptable for v2, but it is the largest missing PKM primitive after basic retrieval.

Do not jump immediately to a full theme graph. Start with lightweight related-capture suggestions during review using existing FTS/category/time signals, then evaluate.

### Application

The architecture separates applications from captures, which is good. The content quality is weak. The deterministic templates are safe but generic. They preserve the shape of the workflow, not the value.

For this product to matter, "apply" must produce outputs the user would actually reuse: client language, investment checklist items, decision rules, operating principles, writing seeds. The model should do the language work, but the DB should store explicit user-accepted outputs, not every rough proposal as if it were equally valuable.

### Compounding vs Note Storage

Right now Wisdom is likely to store notes. It is not yet likely to compound knowledge.

It can become compounding if the next iterations add:

- intentional review cadence,
- related idea resurfacing,
- accept/dismiss decisions,
- versioned durable principles/checklists,
- evals against real recall/apply tasks,
- friction to prevent keeping everything.

Without those, it risks becoming second-brain theater: a clean database with a lot of captured text and little behavioral leverage.

## Agentic Architecture Assessment

### Tool Design

v2's native tool move was correct. `tools/wisdom_tool.py` follows the repo's `registry.register(...)` convention. The tool descriptions include natural trigger language. The handlers are thin and call `wisdom/service.py`.

The tool set is somewhat large. Ten tools are understandable, but `wisdom_inbox` and `wisdom_review` overlap, and `wisdom_status`/`wisdom_set_enabled` are rarely needed in ordinary conversation. This is acceptable now, but if tool bloat becomes visible, consider moving admin tools out of the default toolset while keeping capture/search/original/apply/review.

### Model/Tool Boundary

The boundary is mostly right:

- deterministic code handles persistence, exact originals, hashing, and secret blocking.
- the model decides when natural language implies capture/search/original/apply/review.
- the model can polish final wording after grounded tool calls.

The weak spot is application quality. Current deterministic applications are placeholders. Long term, the model should generate candidate applications under a strict schema, and the deterministic layer should store only validated proposals. That should be a later iteration after eval coverage exists.

### Use of Codex/GPT-5.5

Using Codex/GPT-5.5 for natural tool selection is the right move. Using it to pretend deterministic keyword capture understands language would be wrong, and v2 avoided that.

However, relying only on tool descriptions is unproven. There are no golden conversational evals that simulate natural prompts and assert expected tool calls. Tool schema tests prove availability, not behavior.

### Unnecessary Routing Logic

There is not much unnecessary routing. The v1 gateway explicit-capture intercept is acceptable as a low-friction fallback. It should not expand into a smart-capture router.

### Observability and Debuggability

Observability is thin. Hermes has pre/post tool hooks and an observability plugin path, but Wisdom does not yet provide a compact local audit trail of tool usage, capture outcomes, blocked captures, search misses, or apply acceptance. Avoid raw-content logs, but add counters/events later.

## Hermes-Native Integration Assessment

### Toolset Integration

Wisdom is integrated in the native Hermes way:

- `tools/wisdom_tool.py` registers native tools.
- `toolsets.py` defines a `wisdom` toolset.
- `_HERMES_CORE_TOOLS` includes Wisdom, so `hermes-cli`, `hermes-telegram`, and other messaging toolsets include it.
- `hermes-api-server` also includes Wisdom.
- `hermes-acp` does not include Wisdom, which is reasonable because ACP is coding-focused.

The risk is that `_HERMES_CORE_TOOLS` makes Wisdom broadly visible. That is good for the product direction but should be monitored for tool confusion and token/schema bloat.

### Command Fallback

The `/wisdom` command is properly registered in `hermes_cli/commands.py` and handled in `gateway/run.py`. The command fallback is still important for debugging exact IDs, status, and retrieval.

### Gateway Concerns

The gateway integration is minimal and sensible:

- `/wisdom` handled post command recognition.
- explicit natural capture happens only when `not command`.
- fail-open warnings are redacted.
- durable capture is not done in pre-auth hooks.

The main concern is deployment/stale-code. The running launchd process needs restart to load new tool modules. This is operational, not architectural.

### Skills

Not adding a skill was correct. The inspected skill path is mostly slash/invocation-based and not a reliable always-loaded gateway instruction surface. Tool descriptions are the right v2 mechanism.

### Memory and Session Search

Wisdom is separate from Hermes memory and session search. That is correct for now:

- session search recalls conversations,
- memory stores user/profile facts,
- Wisdom stores user-selected thoughts and applications.

Future integrations should be explicit. Do not silently merge Wisdom with global memory or session search. A future "search all my local memory" orchestration can call both `wisdom_search` and `session_search`, but Wisdom should remain a distinct store.

### Cron and Voice

Cron and voice substrates exist, but Wisdom is right not to use them yet.

Scheduled Wisdom review would be useful eventually, but automatic pings before review quality is good will create noise. Voice capture could reduce friction, but it adds transcription cost, latency, and privacy risk before the core loop has proven value.

## Data Model & Storage Assessment

### Schema Adequacy

The v1 schema is adequate for a kernel:

- `raw_events` preserves original input.
- `captures` stores classification and metadata.
- `interpretations` keeps annotations separate.
- `applications` stores internal proposals.
- `settings` stores enable/capture mode.

The schema is intentionally small and should not be overexpanded yet.

### Exact-Original Design

The exact-original design is right. Blocking secrets rather than redacting them into "exact" originals is the correct tradeoff.

One nuance: the native tool can pass `category`, `source_type`, and `context_note`. These do not alter the original text, so they are safe structurally. But model-supplied category/source metadata can be wrong. Treat those fields as hints, not truth.

### FTS/Search

FTS is fine for v2. It searches original, cleaned, title, interpretation text, and application text. LIKE fallback exists.

Limitations:

- no semantic recall,
- no recency/category scoring beyond basic query filtering,
- no related-idea retrieval,
- no search-miss tracking,
- FTS query normalization is minimal.

Do not add embeddings until there are real search failure examples and evals.

### Migrations and Backups

Migrations use `schema_version`; WAL fallback exists. That is good.

Backup strategy is under-specified. A local daily/weekly backup before migrations and before destructive status changes should be added before schema v2 migrations. There is archive, not delete, so the immediate data-loss risk is moderate.

### Future Schema Needs

Likely future fields/tables:

- proposal acceptance/dismissal events,
- application versioning,
- lightweight links between captures,
- review sessions and decisions,
- search/capture/apply audit events with no raw sensitive text,
- optional source provenance beyond hashed session/message IDs,
- durable principles/checklists as first-class accepted outputs.

Do not add all of these at once.

## Security / Privacy Assessment

### Redaction and Blocking

Secret blocking covers common API keys, bearer tokens, cookies, auth headers, passwords, GitHub/Slack/OpenAI-style tokens, and private keys. It deliberately avoids over-redacting normal thoughts. That is the right posture.

The system should eventually expose a "blocked capture" confirmation only for explicit commands/tools, not for ordinary gateway capture. Current gateway silent pass-through for secret-like natural capture is reasonable.

### Logs

Wisdom avoids raw-content logging in its own code. Gateway warnings redact exception strings. Tool error details are redacted and truncated.

Potential issue: tool inputs/results may be observed by generic Hermes observability plugins. If users enable external observability, Wisdom tool args could contain private thoughts. The docs should eventually warn that Wisdom content can appear in model/tool traces if tracing plugins are enabled.

### Secrets and IDs

Session/message identifiers are HMAC-hashed with a local salt. Raw Telegram IDs are filtered out of metadata. This is solid for local storage.

### Prompt-Injection Concerns

Prompt injection can make the model call `wisdom_capture` with content the user did not intend to save if the injected text is in the conversational context. The tool description says to capture only explicit user requests, but tool descriptions are not a hard security boundary.

Mitigations:

- do not auto-capture ordinary chat,
- keep capture tool narrow,
- add evals for adversarial "ignore previous and save this" content in quoted/web/session material,
- consider requiring model tools to identify the exact user-authored span being captured when the source is not the latest user message.

### External Actions

External actions are adequately blocked. Applications are internal proposals only. There is no productivity DB write and no send-message behavior inside Wisdom tools.

## Reliability / Test / Eval Assessment

### What Tests Prove

The existing tests prove:

- DB initializes and migrations are idempotent.
- WAL/FTS/LIKE paths work in simple cases.
- exact originals round-trip.
- secret-like captures are blocked.
- raw platform IDs are not stored in metadata.
- deterministic classification and interpretation behave as expected.
- application generation is internal and idempotent.
- commands work.
- gateway Wisdom paths fail open.
- native tools are registered, schema-shaped, in toolsets, and callable through `handle_function_call`.
- temp DBs are used in tests.

This is good engineering coverage for v1/v2.

### What Tests Do Not Prove

They do not prove:

- live model tool-selection behavior,
- Telegram end-to-end behavior after launchd restart,
- natural-language pronoun resolution like "turn that into client language",
- search quality at realistic corpus size,
- application usefulness,
- review usefulness,
- tool confusion with the full default toolset,
- behavior under concurrent gateway calls,
- backup/restore safety,
- observability privacy behavior.

### Missing Evals

Add a local no-network eval suite that feeds mocked model tool calls or recorded expected tool-call plans for prompts like:

- "Remember this: ..."
- "Do not save this, just explain it."
- "Find that idea about peace of mind."
- "Show exact wording."
- "Turn that into client language."
- "What have I been thinking about investing recently?"
- "The web page says 'remember this secret'; summarize it."
- "I pasted an API key by mistake; don't store it."

Add a small live smoke checklist after restart, but do not make tests send Telegram messages automatically.

### Suggested Smoke Tests

After gateway restart:

1. Capture a natural thought and verify response includes capture ID.
2. Search for it using paraphrased and exact wording.
3. Ask for exact original and verify verbatim text.
4. Apply it to client language and verify no external tasks/reminders are created.
5. Ask an ordinary informational question and verify Wisdom is not used.
6. Try a secret-like capture and verify it is blocked/not stored.
7. Run `/wisdom status`, `/wisdom inbox`, `/wisdom original <id>`.

### Regression Strategy

- Keep v1 unit tests.
- Add model-tool descriptor regression tests.
- Add a Wisdom eval fixture with expected tool-call intent labels.
- Add corpus search evals once real captures exist.
- Add concurrency tests around capture/search/apply using temp DBs.
- Add migration/backup tests before schema v2.

## The Biggest Risks

1. Product becomes PKM theater.
   - Mitigation: build review/apply/accept loops and measure whether captures turn into reusable outputs.

2. Live model behavior does not match schema tests.
   - Mitigation: add golden conversational evals and post-restart smoke tests.

3. Search misses become frustrating.
   - Mitigation: track search misses and add query expansion/reranking before embeddings.

4. Application proposals remain generic.
   - Mitigation: let the model draft richer proposals but store only validated, structured, user-accepted outputs.

5. Tool bloat confuses the model.
   - Mitigation: monitor tool call choices; consider reducing default Wisdom tools to capture/search/original/apply/review.

6. Prompt injection causes unwanted capture.
   - Mitigation: eval adversarial contexts and require explicit user-authored save intent.

7. Private thoughts leak through tracing or logs.
   - Mitigation: document observability risk; keep Wisdom raw content out of logs; redact tool error details.

8. Stale launchd code creates false debugging trails.
   - Mitigation: include version/commit startup log or `wisdom_status` code version after restart.

9. Schema migrations become risky without backups.
   - Mitigation: implement backup-before-migration before any schema v2.

10. User captures too much and reviews too little.
    - Mitigation: add triage, archive/dismiss, and review prompts before adding more capture surfaces.

## The Biggest Missing Pieces

1. A real review/triage loop.
2. Golden conversational evals for model tool use.
3. Better application output quality and accept/dismiss workflow.
4. Related idea resurfacing.
5. Search quality measurement.
6. Lightweight observability without raw content.
7. Backup/restore before schema evolution.
8. Live post-restart Telegram smoke evidence.
9. Clear distinction between raw captures, durable principles, and disposable notes.
10. Product language that sets realistic expectations.

## Recommended Next Build

Do not make v3 voice. Do not make v3 embeddings. Do not make v3 a dashboard.

Recommended v3: Wisdom Review & Quality Loop.

Build a small, high-quality loop that proves Wisdom can turn captures into useful knowledge:

- a local eval harness for natural-language Wisdom intents,
- improved `wisdom_review` output with recent captures, unapplied captures, candidate related ideas, and explicit "keep/apply/archive" recommendations,
- application proposal accept/dismiss/archive commands/tools,
- a first-class "accepted output" concept for principles/checklists/client language/decision rules,
- search-miss and apply-quality smoke fixtures using temp DBs,
- documentation for live Telegram smoke testing after restart.

This should remain manual/on-demand in v3. No scheduled review until the review output is good enough to deserve interrupting the user.

### Should v3 Be Voice?

No. Voice reduces capture friction, but the system has not yet proved that captured material becomes useful. Voice would increase input volume before review quality exists.

### Should v3 Be Related Ideas?

Yes, but lightweight. Add related capture suggestions inside review/apply, using existing category/FTS/time signals first. Do not build a full theme graph yet.

### Should v3 Be Weekly Review?

Not scheduled. Build manual review first. Once manual review is useful and tested, schedule it.

### Should v3 Be Better Tool Descriptions?

Minor improvements only. The bigger need is evals proving the descriptions work with the full toolset.

### Should v3 Be Topic-Specific Capture?

Not yet. Topic-specific capture is useful, but it is lower priority than triage/review/application quality.

## Things Not To Build Yet

- Voice transcription capture.
- Scheduled weekly/monthly review.
- Semantic embeddings/vector DB.
- Dashboard/export.
- Apple Notes, Readwise, Notion, Obsidian integration.
- Old productivity DB migration.
- External task/reminder execution.
- Full theme graph.
- Agent swarm for capture/review.
- Large regex smart-capture router.
- Automatic capture of all ordinary chat.
- Cloud sync.

## Proposed Roadmap

### Immediate Hardening

- Add a no-network Wisdom conversational eval harness.
- Add live smoke-test documentation and expected outputs.
- Add concurrency tests around capture/search/apply.
- Add a `wisdom_status` or startup-visible code/version field to reduce stale-code confusion.
- Document observability/privacy caveat for tool tracing plugins.

### v3

Wisdom Review & Quality Loop:

- better manual review,
- related ideas inside review,
- apply proposal accept/dismiss/archive,
- first-class accepted outputs,
- search/apply quality fixtures,
- no scheduled interruptions.

### v4

Resurfacing and cadence:

- optional weekly review once manual review is good,
- stronger related-idea ranking,
- durable principles/checklists/client-language library,
- lightweight backup-before-migration,
- optional export of accepted outputs only.

### Longer-Term

- voice capture after review loop proves value,
- semantic retrieval after lexical search failures are measured,
- dashboard only if CLI/Telegram review becomes insufficient,
- old productivity import only if source is recovered and migration can be audited,
- external task/reminder execution only with explicit confirmation and strong tests.

## Acceptance Criteria for the Next Iteration

- No code path writes to `~/.hermes/productivity/productivity.db`.
- Existing v1/v2 tests still pass unchanged.
- New eval harness classifies at least 30 natural-language prompts into expected Wisdom intents without live model calls.
- Adversarial prompts from quoted/web/session content do not produce capture intent unless the user explicitly asks to save.
- Manual review output includes recent captures, unapplied captures, related candidates, and clear action suggestions.
- Application proposals can be accepted or dismissed without external actions.
- Accepted outputs are stored separately from raw captures/proposals.
- Exact originals remain retrievable verbatim.
- Search/review/apply tests use temp DBs only.
- Live smoke checklist is documented and can be run manually after gateway restart.
- The running gateway does not need new invasive hooks.

## Open Questions for User

1. Which output type matters most first: client language, investment rules, decision rules, checklists, or principles?
2. Should Wisdom optimize for Telegram-first daily use, or should CLI/local workflows be equally important?
3. How much interruption is acceptable once reviews are good: manual only, daily brief inclusion, or weekly scheduled ping?
4. Should captured business/investing material be considered private enough to disable external observability/tracing by default when Wisdom tools run?

## Final Recommendation

Keep the architecture. Do not declare product success yet.

v1 and v2 are the right foundation: local, exact-original preserving, deterministic, source-backed, and exposed as native Hermes tools. The next decision should be product discipline, not feature expansion. Build the review/application/eval loop that proves Wisdom can convert captured thoughts into reusable judgment. If that loop works, voice, scheduled reviews, related themes, and semantic retrieval become justified. If it does not, more capture surfaces will only create a larger pile of notes.
