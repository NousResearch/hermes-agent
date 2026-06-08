# Hermes Wisdom Kernel v1 Specification

## 1. Product Goal

Build Hermes Wisdom Kernel v1:

A Hermes-native, Telegram-first memory kernel that lets the user naturally capture thoughts, preserves the user's exact words, optionally interprets/distills them, makes them searchable, and lets the user create internal application proposals.

This is the kernel for a future Hermes Personal Intelligence OS, but v1 must stay focused.

The kernel must do five things exceptionally well:

1. Preserve exact originals.
2. Store captures durably.
3. Classify lightly.
4. Search reliably.
5. Create application proposals without taking external actions.

## 2. Core Principle

The user's exact words are the source of truth.

Never replace the user's words with a summary.

Store these as separate layers:

- original text exactly
- optional cleaned text
- optional interpretation
- optional application proposal

Distillation/interpretation is an annotation, not a replacement.

If the model or interpretation layer fails, capture must still work.

## 3. Non-Goals for v1

Do not implement:

- full Personal Intelligence OS
- theme graph
- automatic smart capture of all messages
- scheduled reviews
- voice transcription/STT pipeline
- old productivity migration
- external task/reminder execution
- semantic embeddings/vector DB
- web dashboard
- Apple Notes integration
- Readwise integration
- Notion/Obsidian export
- full task/reminder engine
- agent swarm for every message

## 4. Context and Repo Reality

Repo:

`~/.hermes/hermes-agent`

Runtime:

- Hermes runs locally through launchd as `ai.hermes.gateway`.
- Telegram gateway exists.
- Model/provider path exists.
- Skills/session search/scheduler/voice/media substrate exists.
- Plugin LLM facilities may exist.
- Existing custom productivity system is not safe to depend on because source is missing/not importable.
- Do not rely on or write to `~/.hermes/productivity/productivity.db`.

Build Wisdom Kernel as a fresh, source-backed subsystem.

## 5. Naming and Paths

Source package:

`wisdom/`

User-facing name:

Hermes Wisdom Kernel

Default DB path:

`~/.hermes/wisdom/wisdom.db`

Test DBs must use temp dirs and must not touch the real DB.

Documentation:

`docs/wisdom_kernel.md`

Specification:

`docs/specs/wisdom_kernel_v1.md`

## 6. Required Source Layout

Create or adapt as appropriate:

```text
wisdom/
  __init__.py
  config.py
  models.py
  db.py
  router.py
  capture.py
  interpret.py
  retrieve.py
  review.py
  apply.py
  render.py
  redaction.py
  integration.py
  commands.py
  errors.py
```

Tests:

```text
tests/wisdom/
  test_db.py
  test_router.py
  test_capture.py
  test_interpret.py
  test_retrieve.py
  test_apply.py
  test_render.py
  test_redaction.py
  test_integration_fail_open.py
```

Follow existing repo conventions where appropriate, but keep the package clean and source-backed.

## 7. Database Design

Use SQLite.

Default DB:

`~/.hermes/wisdom/wisdom.db`

Requirements:

- use WAL mode
- use transactions
- use migrations with `PRAGMA user_version` or an explicit `schema_migrations` table
- use temp DBs in tests
- do not write to the old productivity DB
- no model-generated SQL
- no SQL outside DB layer except tests if absolutely necessary

## 8. Minimum Tables

### 8.1 raw_events

Stores original inbound material exactly.

Fields:

- id
- created_at
- channel
- source_kind
- session_key_hash
- message_ref_hash
- original_text
- metadata_json
- processing_state

Notes:

- original_text is exact preserved text.
- Do not store raw chat IDs.
- Hash session/message identifiers.
- Redact obvious secrets before storage/logging when needed, while preserving ordinary user wording.
- Avoid over-redacting normal thoughts.

### 8.2 captures

Stores recognized knowledge captures.

Fields:

- id
- raw_event_id
- created_at
- updated_at
- title
- original_text
- cleaned_text
- category
- source_type
- status
- confidence
- importance_score
- novelty_score
- actionability_score
- metadata_json

Categories:

- business
- investing
- life
- health
- inbox

Source types:

- thought
- voice
- podcast
- book
- article
- meeting
- quote
- conversation
- other

Statuses for v1:

- raw
- interpreted
- applied
- archived

### 8.3 interpretations

Stores Hermes/AI interpretation separately.

Fields:

- id
- capture_id
- created_at
- summary
- insight
- why_it_matters
- possible_application
- counterpoint
- confidence
- model_used
- metadata_json

### 8.4 applications

Stores application proposals only. No external action execution in v1.

Fields:

- id
- capture_id
- created_at
- application_type
- title
- body
- status
- metadata_json

Application types:

- task_proposal
- reminder_proposal
- principle
- checklist
- client_language
- investment_rule
- health_experiment
- writing_idea
- decision_rule

Statuses:

- proposed
- accepted
- dismissed
- archived

### 8.5 settings

Fields:

- key
- value
- updated_at

Settings should include:

- enabled
- capture_mode

Capture modes:

- off
- explicit
- smart

Default:

`explicit`

For v1, implement explicit well. smart can exist as a config value, but it should be conservative or disabled unless safely tested.

## 9. FTS / Search

If FTS5 is available, create FTS indexes over:

- raw original text
- capture original text
- capture cleaned text
- title
- interpretation summary
- interpretation insight
- application title/body

If FTS5 is unavailable, gracefully fall back to safe LIKE-based search.

Search must find both:

- original user wording
- interpretation/application wording

## 10. Exact Original Preservation

When the user sends:

> Remember this: clients don't buy alpha. They buy the feeling that someone sensible is watching the road.

Store the exact original text.

Hermes may also store:

- cleaned text
- title
- category
- interpretation

But the original must remain recoverable exactly.

The command:

`/wisdom original <id>`

must return the exact stored original.

## 11. Explicit Capture Triggers

Capture when a message begins with or clearly contains:

- remember this
- save this
- save this thought
- note this
- business idea
- investing thought
- health note
- life thought
- book note
- podcast idea
- `/wisdom capture`

Do not capture ordinary chat unless explicitly commanded.

Examples that should capture:

- Remember this: clients buy peace of mind, not alpha.
- Podcast idea: cadence beats strategy for x10x.
- Investing thought: position sizing matters more than thesis confidence.
- Health note: poor sleep changes my decision quality.
- `/wisdom capture Clients need windshields, not rear-view mirrors.`

Examples that should pass through:

- What do you think about this?
- Help me debug Hermes.
- `/todo now call Yash`
- `/today`
- `/remind me tomorrow`

Existing non-wisdom slash commands must not be captured.

## 12. Classification

Use deterministic heuristics first.

Category heuristics:

- business: client, x10x, PMS, AIF, report, sales, team, prospect, business, meeting, ops
- investing: stock, market, option, portfolio, risk, macro, thesis, PMS, allocation, trade, sizing
- health: sleep, food, energy, exercise, gym, decision quality, health, lunch, cognition
- life: family, relationship, happiness, philosophy, courage, fear, meaning, habit
- inbox: fallback

If a term could belong to multiple categories, choose the most contextually likely category. If uncertain, use inbox.

Source type heuristics:

- podcast idea -> podcast
- book note -> book
- health note -> thought unless voice context is supplied
- investing thought -> thought
- quote -> quote
- voice transcript context -> voice
- otherwise -> thought or other

## 13. Interpretation

For v1:

- Interpretation may be deterministic/lightweight if no safe LLM context is available.
- If the repo's plugin LLM context is safely available in gateway integration, use structured outputs for interpretation with strict schema validation and hard timeout.
- If interpretation fails, still save raw event and capture.
- No model failure should block capture.

Interpretation should produce, when possible:

- summary
- insight
- why it matters
- possible application
- counterpoint
- confidence

Counterpoint matters. Avoid turning the system into a self-confirmation machine.

The interpretation layer must never overwrite original text.

## 14. Commands

Implement `/wisdom` command handling.

Required commands:

- `/wisdom status`
- `/wisdom capture <text>`
- `/wisdom inbox`
- `/wisdom search <query>`
- `/wisdom original <id>`
- `/wisdom interpret <id>`
- `/wisdom apply <id>`
- `/wisdom archive <id>`
- `/wisdom review`
- `/wisdom on`
- `/wisdom off`

Optional, if easy and safe:

- `/wisdom help`

Unknown `/wisdom` subcommands should return concise help.

## 15. Command Behavior

### 15.1 /wisdom status

Show:

- enabled/disabled
- capture mode
- DB path
- counts: captures, interpretations, applications
- FTS availability
- last capture timestamp if any

### 15.2 /wisdom capture <text>

Capture the supplied text.

Return concise confirmation:

```text
Captured #42 · Business · Thought
Original saved exactly.
Optional read: Wealth management sells decision confidence, not just performance.
```

If interpretation is not created, do not pretend it was.

### 15.3 /wisdom inbox

Show recent non-archived captures, especially raw/uninterpreted.

Include:

- ID
- date
- category
- title
- short original excerpt

### 15.4 /wisdom search <query>

Search originals, captures, interpretations, and applications.

Results should show:

- capture ID
- date
- category
- source type
- title
- short original excerpt
- optional insight

### 15.5 /wisdom original <id>

Return exact stored original text.

### 15.6 /wisdom interpret <id>

If interpretation exists, show it.

If not:

- create one if safe
- otherwise provide deterministic interpretation if possible
- otherwise say no interpretation exists yet

### 15.7 /wisdom apply <id>

Create one or more internal application proposals.

Examples:

- business captures may produce client_language, task_proposal, principle
- investing captures may produce investment_rule, checklist
- health captures may produce health_experiment, decision_rule
- life captures may produce principle, writing_idea, decision_rule

No external task/reminder should be created.

Example output:

```text
Application proposals for #42:
1. Client language: "Our job is to help you make fewer regretful decisions over time."
2. Principle: Wealth management is regret minimization.
3. Task proposal: Add "What this means now" to x10x client reports.
```

### 15.8 /wisdom archive <id>

Mark capture archived.

### 15.9 /wisdom review

Manual v1 review summary:

- count by category
- recent best captures
- unapplied captures
- suggested application candidates

Do not schedule automatic weekly reviews in v1.

### 15.10 /wisdom on and /wisdom off

Persist enabled setting.

When off:

- natural capture should not occur
- `/wisdom` commands should still allow status/help/on

## 16. Gateway Integration

Integrate with Hermes Telegram/gateway only if it can be done safely and minimally.

Audit found:

- Telegram handlers in `gateway/platforms/telegram.py`
- Gateway core in `gateway/run.py`
- Post-auth/pre-agent region is the preferred extension zone
- `pre_gateway_dispatch` is pre-auth and should not durably capture unless it rechecks auth

Integration principles:

- Do not capture pre-auth.
- Do not break existing slash commands.
- Existing Hermes commands must win unless command is explicitly `/wisdom`.
- Wisdom should have a short timeout and fail open.
- Wisdom core should not directly send Telegram messages.
- Integration layer may return a rendered response to the gateway if the gateway architecture supports command responses.
- If integration is too risky, implement the Wisdom package and command handlers cleanly, document the exact safe hook location, and add tests/stubs. But make a best effort to wire it end-to-end safely.

Desired flow:

```text
authorized Telegram text
-> existing command parsing
-> if /wisdom: handle Wisdom command
-> else if capture_mode explicit and message has explicit capture trigger: capture, optionally render short confirmation
-> else pass through to normal Hermes agent unchanged
```

If a capture confirmation is sent, keep it short.

## 17. Fail-Open Requirement

Wisdom must fail open.

If any Wisdom code raises an exception during gateway handling:

- log a minimal redacted warning if appropriate
- do not block normal Hermes chat
- do not crash gateway
- do not send noisy failure messages unless the user explicitly ran a `/wisdom` command

For explicit `/wisdom` commands, return a concise error.

For ordinary messages, silently pass through.

## 18. Privacy and Redaction

Implement `wisdom/redaction.py`.

It should detect and redact obvious secrets before storage/logging when feasible:

- API keys
- bearer tokens
- auth headers
- cookies
- obvious passwords
- private keys

Do not over-redact normal personal or business thoughts.

Hash:

- session keys
- message references
- platform IDs

Use stable local hashing with a local salt if appropriate.

Do not print salt.

Do not store raw Telegram chat IDs in Wisdom DB.

Avoid raw-content logs. If logging is needed, log IDs/counts/status only.

## 19. Rendering

Keep responses concise and Telegram-friendly.

Do not dump huge objects.

Renderers should be platform-independent:

- core returns structured result
- render formats text
- gateway sends through existing mechanism

Formatting should be readable in Telegram.

Avoid exposing raw hashes or internal implementation metadata.

## 20. Settings and Config

Add config support for:

- `HERMES_WISDOM_ENABLED`
- `HERMES_WISDOM_DB_PATH`
- `HERMES_WISDOM_CAPTURE_MODE`
- `HERMES_WISDOM_MAX_RESULTS`
- `HERMES_WISDOM_INTERPRET_TIMEOUT`

Defaults:

- enabled: true
- DB path: `~/.hermes/wisdom/wisdom.db`
- capture mode: explicit
- max results: safe small number, e.g. 5 or 10
- interpretation timeout: short/bounded

Persistent settings table should also track enabled/capture_mode.

Environment/config should override defaults where appropriate.

## 21. Documentation

Add:

`docs/wisdom_kernel.md`

Include:

- what Wisdom Kernel is
- what it is not
- DB path
- commands
- capture modes
- privacy behavior
- exact-original preservation rule
- how to enable/disable
- known limitations
- future v2 ideas

Also ensure `docs/specs/wisdom_kernel_v1.md` exists and reflects final implementation reality.

If source-backed Hermes skills live in the repo and there is a clear convention, add a Wisdom skill in the repo. Otherwise do not mutate `~/.hermes/skills`; document how to install a skill later.

## 22. Tests Required

Be rigorous.

Add tests for the categories below.

### 22.1 DB

- initialization creates DB and tables
- WAL mode enabled
- migrations idempotent
- raw event insert
- capture insert
- interpretation insert
- application insert
- FTS search if available
- LIKE fallback if FTS unavailable or hard to test

### 22.2 Exact Preservation

- original text is stored and retrieved exactly
- cleaned/interpretation does not overwrite original

### 22.3 Router

- ordinary chat passes through
- explicit capture triggers route to capture
- `/wisdom capture` routes to capture
- `/wisdom search` routes to search
- `/wisdom original` routes to original retrieval
- `/wisdom apply` routes to application proposal
- `/wisdom off` disables natural capture
- existing non-wisdom slash commands are not captured

### 22.4 Capture and Classification

- business classification
- investing classification
- health classification
- life classification
- inbox fallback
- source type detection
- confidence defaults

### 22.5 Interpretation

- schema validation
- interpretation failure still saves raw capture
- counterpoint field supported
- interpretation never overwrites original

### 22.6 Search and Retrieval

- search finds original wording
- search finds interpretation wording
- `/wisdom original <id>` returns exact original

### 22.7 Apply

- application proposals created
- no external productivity DB writes
- no external actions

### 22.8 Redaction and Privacy

- obvious secrets redacted
- raw chat IDs not stored
- hashes produced for session/message identifiers

### 22.9 Integration / Fail-Open

- Wisdom exceptions do not block normal chat path
- command path remains intact
- no direct Telegram sends from core package
- non-wisdom slash commands pass through

Run targeted tests.

Run relevant existing gateway tests if safe and quick.

Do not run tests that send real Telegram messages.

Do not mutate the real productivity DB.

Use temp HOME/HERMES_HOME in tests where possible.

## 23. Quality Bar

Implement like production local infrastructure:

- typed dataclasses or Pydantic models where appropriate
- small functions
- clear boundaries
- no model-generated SQL
- no duplicated SQL outside DB layer
- transactions for writes
- readable error handling
- fail-open integration
- deterministic tests
- no global mutable surprises
- no large invasive gateway edits if avoidable
- meaningful docstrings/comments where helpful
- platform-independent core logic
- no direct Telegram sends from core package
- no direct dependency on old productivity system

## 24. Verification Checklist

Before finishing:

1. Show files changed.
2. Run targeted Wisdom tests.
3. Run relevant safe existing gateway tests if feasible.
4. Confirm no writes to old productivity DB.
5. Confirm tests use temp DBs.
6. Confirm exact-original retrieval works.
7. Confirm ordinary chat pass-through routing works.
8. Confirm `/wisdom` commands work in unit/integration tests.
9. Confirm import works:
   - `python -c "import wisdom"`
10. Confirm docs exist:
    - `docs/specs/wisdom_kernel_v1.md`
    - `docs/wisdom_kernel.md`
11. Commit changes if tests pass.

Commit message:

`Implement Hermes Wisdom Kernel v1`

## 25. Final Response Requirements

Final response must include:

- commit hash
- files changed
- tests run and results
- DB path
- how to enable/disable
- command examples
- whether gateway integration is fully wired or package/command-ready only
- limitations
- whether a Hermes restart is required for the running gateway to load the new code
- any deviations from spec and where documented

## 26. Future v2 Ideas

Do not implement these now, but mention in docs as future directions:

- voice note transcript capture
- topic-specific capture modes
- weekly scheduled review
- monthly strategic review
- skeptic/challenge mode
- theme graph / recurring patterns
- first-class principles/checklists
- internal task/reminder execution
- optional import from old productivity DB
- Apple Notes import
- Readwise import
- semantic embeddings
- dashboard/export

Build the kernel brilliantly. Make the foundation so good that v2 can safely add voice, themes, skeptic mode, weekly review, and task/reminder execution later.
