# Hermes Wisdom v3 Review & Quality Loop Plan

## Objective

Wisdom v3 turns the v1/v2 foundation from capture/search infrastructure into a manual review and quality loop:

```text
capture -> review -> judge -> relate -> apply/dismiss -> compound
```

The goal is to prevent Wisdom from becoming a pile of saved notes. v3 stays local, deterministic, source-backed, and Hermes-native.

## Repo Inspection Findings

- `README.md` frames Hermes as a tool-using, messaging-native agent with memory, session search, skills, gateway, cron, and voice substrates.
- `website/docs/developer-guide/adding-tools.md` and `tools-runtime.md` confirm the built-in tool convention: add `tools/*.py`, return JSON strings, register with `tools.registry`, and wire the toolset in `toolsets.py`.
- `website/docs/user-guide/features/memory.md` distinguishes always-in-context curated memory from on-demand session search. Wisdom should remain separate: user-selected thought capture, not global profile memory.
- `website/docs/user-guide/messaging/telegram.md` and `developer-guide/gateway-internals.md` confirm Telegram/gateway flow, slash command dispatch, authorization boundaries, and the need to avoid pre-auth durable capture.
- `website/docs/user-guide/features/cron.md` confirms scheduled jobs are powerful but should not be used for Wisdom review until manual review quality is proven.
- `website/docs/user-guide/features/voice-mode.md` confirms voice is already a Hermes substrate, but v3 should not add transcription capture because review quality is the current bottleneck.
- `docs/specs/wisdom_product_architecture_review.md` recommends exactly this v3: review quality, related suggestions, accept/dismiss workflow, and deterministic evals before voice/embeddings/dashboard/scheduled review.

## 1. What v1/v2 Currently Provide

v1 provides:

- `wisdom/` package
- SQLite DB at `~/.hermes/wisdom/wisdom.db`
- exact original storage in `raw_events.original_text`
- deterministic capture/classification/search/interpret/apply
- `/wisdom` command fallback
- explicit natural capture phrases in the post-auth gateway path
- fail-open gateway behavior

v2 provides:

- `tools/wisdom_tool.py`
- `wisdom/service.py`
- native Hermes model tools
- `wisdom` toolset in the default core toolset
- model/tool-backed natural-language use
- shared command/tool service layer

## 2. Product Problem v3 Solves

v1/v2 answer: "Can Hermes remember and retrieve this?"

v3 answers: "What should I do with what Hermes remembered?"

It adds prioritization, judgment, related ideas, and explicit disposition actions so captures become reusable thinking assets instead of passive notes.

## 3. Why Review/Application Quality Comes Before More Capture Sources

More capture without review creates PKM theater. Voice, smart capture, scheduled reviews, imports, or dashboards would increase the volume of stored notes before the system can decide what matters.

v3 should improve the value of existing and future captures:

- rank high-potential ideas,
- hide low-value dismissed/archive material,
- surface accepted-but-unapplied captures,
- show related captures without embeddings,
- make application proposals more useful,
- let the user accept/dismiss/archive/apply.

## 4. Hermes-Native Patterns v3 Will Reuse

- Native tool registration in `tools/*.py` through `tools.registry`.
- Existing `wisdom` toolset in `toolsets.py`.
- Existing `/wisdom` command in `hermes_cli/commands.py` and `gateway/run.py`.
- SQLite/WAL/schema migration pattern in `wisdom/db.py`.
- Renderer separation in `wisdom/render.py`.
- Temp DB testing patterns in `tests/wisdom/conftest.py`.
- No MCP, no plugin, no new gateway router.

## 5. DB/Schema Additions

Use a minimal schema migration to version 2.

Add columns to `captures`:

- `review_status TEXT NOT NULL DEFAULT 'unreviewed'`
- `reviewed_at REAL`
- `accepted_at REAL`
- `dismissed_at REAL`
- `applied_at REAL`

Keep existing `captures.status` for active/archived visibility. `review_status='archived'` mirrors archive workflow, but the old `status='archived'` remains the hiding mechanism.

Do not add a theme graph, embeddings table, accepted-output table, or review-events table in v3. Those can come after the manual loop proves useful.

Quality scores will be computed deterministically on demand. Existing `importance_score`, `novelty_score`, and `actionability_score` columns remain available but v3 will not require backfilling them.

## 6. Service Functions To Add

Add or update service operations in `wisdom/service.py`:

- `review(...)`
- `related(capture_id, ...)`
- `accept(capture_id)`
- `dismiss(capture_id)`
- update `archive(capture_id)` to set review status too
- update `apply(capture_id, ...)` to mark captures applied

Add `wisdom/review.py` for deterministic scoring, review queue construction, related-capture suggestions, and suggested next actions.

All SQL remains in `wisdom/db.py`.

## 7. Tools To Add Or Change

Keep existing tools and update behavior:

- `wisdom_review`: return prioritized review items, high-potential counts, unapplied counts, related IDs, quality indicators, and suggested actions.
- `wisdom_apply`: create better deterministic applications and mark the capture applied.
- `wisdom_archive`: mark review status archived.

Add:

- `wisdom_related`
- `wisdom_accept`
- `wisdom_dismiss`

This adds three tools. That is acceptable because the actions are model-facing natural-language verbs, but v3 should not add more tool surfaces beyond this.

## 8. `/wisdom` Commands To Add Or Change

Keep all existing commands.

Improve:

- `/wisdom review`
- `/wisdom review <category>`
- `/wisdom review unapplied`
- `/wisdom review high-potential`
- `/wisdom apply <id>`
- `/wisdom archive <id>`

Add:

- `/wisdom related <id>`
- `/wisdom accept <id>`
- `/wisdom dismiss <id>`

Do not remove any v1/v2 command.

## 9. Natural-Language Model/Tool Usage After v3

The model should call Wisdom tools for:

- "What should I review?" -> `wisdom_review`
- "Review my recent business ideas." -> `wisdom_review(category="business")`
- "What have I captured but not applied?" -> `wisdom_review(mode="unapplied")`
- "Show me high-potential ideas." -> `wisdom_review(mode="high_potential")`
- "Show related ideas." -> `wisdom_related`
- "Accept that." -> `wisdom_accept`
- "Dismiss this one." -> `wisdom_dismiss`
- "Turn #15 into an investment rule." -> `wisdom_apply(application_type="investment_rule")`

The model should use visible capture IDs from prior tool results for "that" references. No hidden state machine is added.

## 10. Related-Capture Suggestions Without Embeddings

Use deterministic signals only:

- token overlap between original/cleaned/title text,
- same category,
- same source type,
- FTS search using top tokens from the target capture,
- recency as a weak tie-breaker.

Return IDs, category, excerpts, and concise reasons. Do not use vector embeddings.

## 11. Accept/Dismiss/Archive/Apply Workflow

- `accept`: mark a capture as worth compounding; do not create applications automatically.
- `dismiss`: mark a capture as low-value/noise; preserve original; hide/deprioritize from default review.
- `archive`: hide from normal review/search via existing archive behavior; preserve original.
- `apply`: create internal application proposals and mark the capture applied.

No delete operation. No external tasks/reminders.

## 12. Quality Scoring

Add deterministic score fields to review output:

- importance
- actionability
- novelty
- reusability
- overall

Scores are not objective. They are sorting hints.

Heuristics:

- category/source type,
- length and specificity,
- action/reuse language,
- business/investing/health/life domain terms,
- metaphor/client-language potential,
- whether applications already exist,
- whether related captures exist,
- review status.

Default review prioritizes:

- unreviewed captures,
- accepted but unapplied captures,
- high-actionability business/investing captures,
- repeated or related themes,
- older unreviewed material.

Applied and dismissed captures should not keep resurfacing in default review.

## 13. Tests/Evals

Add deterministic fixture-based review-quality tests:

- business/client reporting
- investing/position sizing
- health/sleep/decision quality
- life/systems-as-avoidance
- low-value/noisy capture

Tests should cover:

- schema migration,
- review statuses,
- accept/dismiss/archive/apply,
- related suggestions,
- quality ranking,
- improved application templates,
- native tool registration/descriptions,
- command variants,
- exact-original preservation,
- no productivity DB writes.

No live LLM calls. No Telegram sends.

## 14. Explicitly Not Included In v3

- voice transcription capture,
- scheduled weekly/monthly reviews,
- embeddings/vector DB,
- dashboard/export,
- external task/reminder execution,
- old productivity DB migration,
- Apple Notes/Readwise/Notion/Obsidian/cloud sync,
- full theme graph,
- automatic capture of ordinary chat,
- MCP.

## 15. Deviations Required By Repo Reality

- Use native Hermes tools, not MCP; this is the established built-in tool path.
- Do not add a skill; Hermes skills are not the reliable always-loaded gateway instruction layer for this product.
- Do not add gateway routing; v2 already gives the model native tools.
- Do not schedule review; cron exists, but the review output needs to prove value manually first.
