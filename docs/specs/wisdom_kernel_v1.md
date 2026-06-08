# Hermes Wisdom Kernel v1 Specification

Status: final implementation specification for the next coding run.
Scope: source-backed Wisdom Kernel only. No dependency on the old custom productivity DB.

## Product Goal

Hermes Wisdom Kernel v1 is a local, Telegram-first capture and recall subsystem for durable personal knowledge.

V1 must do five things well:

1. Preserve accepted originals exactly.
2. Store captures durably in a new Wisdom-owned SQLite DB.
3. Classify lightly with deterministic rules.
4. Search originals, cleaned text, interpretations, and application proposals.
5. Create internal application proposals without taking external actions.

V1 is not the full Wisdom OS. It is the reliable local kernel that later voice, review, theme, skeptic, and action layers can build on.

## Non-Goals

Do not implement these in v1:

- automatic capture of ordinary chat
- semantic embeddings or vector DB
- scheduled reviews
- voice transcription or a new STT/TTS pipeline
- old productivity DB migration
- writes to `~/.hermes/productivity/productivity.db`
- external task/reminder execution
- Apple Notes, Readwise, Notion, Obsidian, or dashboard integration
- agent swarms for capture or interpretation
- Telegram adapter rewrites

## Repo Reality and Constraints

Validated local repo facts:

- The repo is `/Users/sanyamdugar/.hermes/hermes-agent`.
- Telegram handlers live in `gateway/platforms/telegram.py`.
- Gateway dispatch lives in `gateway/run.py`.
- Built-in slash commands are registered in `hermes_cli/commands.py`.
- Plugin slash commands exist via `PluginContext.register_command()` in `hermes_cli/plugins.py:412-464`.
- `pre_gateway_dispatch` exists in `gateway/run.py:6781-6818`, but it runs before authorization.
- Recognized slash commands are access-checked in `gateway/run.py:7455-7464`.
- Plugin slash command dispatch is available in `gateway/run.py:7739-7755`.
- Existing session DB patterns use SQLite WAL, schema migrations, and FTS5 in `hermes_state.py`.
- Plugin LLM structured calls exist through `ctx.llm.complete_structured()` and `acomplete_structured()`.

Architecture conclusion:

- `/wisdom ...` is realistic in v1 as a normal built-in gateway command.
- Natural explicit capture such as `Remember this: ...` is realistic only with a small post-auth/pre-agent gateway intercept. Do not use `pre_gateway_dispatch` for durable capture because it is pre-auth and cannot safely return a normal user-facing command response.
- The Wisdom core package must not import Telegram or send messages directly. Gateway integration may call Wisdom and return rendered strings through the existing gateway response path.

## Source Layout

Create a top-level source package:

```text
wisdom/
  __init__.py
  config.py
  models.py
  db.py
  redaction.py
  classify.py
  capture.py
  retrieve.py
  interpret.py
  apply.py
  render.py
  commands.py
  integration.py
```

Do not add unused placeholder modules. Add `review.py` or `errors.py` later only if implementation needs them.

Tests:

```text
tests/wisdom/
  test_db.py
  test_exact_original.py
  test_redaction.py
  test_classify.py
  test_capture.py
  test_retrieve.py
  test_interpret.py
  test_apply.py
  test_commands.py
  test_gateway_integration.py
```

Documentation:

```text
docs/wisdom_kernel.md
docs/specs/wisdom_kernel_v1.md
```

## Configuration

Use `config.yaml` as the primary configuration surface, consistent with Hermes' non-secret configuration policy.

Add a `wisdom` section to `DEFAULT_CONFIG`:

```yaml
wisdom:
  enabled: true
  db_path: ""
  capture_mode: explicit
  max_results: 5
  interpret_timeout_seconds: 5
  interpretation:
    mode: deterministic
```

Rules:

- Empty `db_path` resolves to `get_hermes_home() / "wisdom" / "wisdom.db"`.
- `capture_mode` values in v1: `off`, `explicit`.
- Do not include `smart` as a live config value in v1. It is a v2 idea.
- Environment variables may be accepted only as test/runtime overrides, not as the documented primary user configuration.
- No secrets belong in Wisdom config.

## Database

Default DB:

`~/.hermes/wisdom/wisdom.db`

Requirements:

- Use SQLite.
- Enable WAL when supported, with safe fallback to DELETE mode using a helper modeled on `hermes_state.py`.
- Use `schema_version` table, not ad hoc migration flags.
- Wrap writes in transactions.
- Keep SQL in `wisdom/db.py`; tests may inspect schema directly.
- Use temp DBs in tests.
- Never read or write `~/.hermes/productivity/productivity.db`.
- Never run model-generated SQL.

### Tables

#### schema_version

- `version INTEGER NOT NULL`

#### raw_events

Stores accepted original input exactly.

- `id INTEGER PRIMARY KEY`
- `created_at REAL NOT NULL`
- `channel TEXT NOT NULL`
- `source_kind TEXT NOT NULL`
- `session_key_hash TEXT`
- `message_ref_hash TEXT`
- `original_text TEXT NOT NULL`
- `metadata_json TEXT NOT NULL DEFAULT '{}'`
- `processing_state TEXT NOT NULL`

`original_text` is the exact accepted original. Do not store raw chat IDs, user IDs, thread IDs, or platform IDs.

#### captures

Stores capture metadata and normalized text. The exact original remains in `raw_events`.

- `id INTEGER PRIMARY KEY`
- `raw_event_id INTEGER NOT NULL REFERENCES raw_events(id)`
- `created_at REAL NOT NULL`
- `updated_at REAL NOT NULL`
- `title TEXT NOT NULL`
- `cleaned_text TEXT`
- `category TEXT NOT NULL`
- `source_type TEXT NOT NULL`
- `status TEXT NOT NULL`
- `confidence REAL NOT NULL`
- `importance_score REAL`
- `novelty_score REAL`
- `actionability_score REAL`
- `metadata_json TEXT NOT NULL DEFAULT '{}'`

Capture statuses:

- `active`
- `archived`

Categories:

- `business`
- `investing`
- `health`
- `life`
- `inbox`

Source types:

- `thought`
- `voice`
- `podcast`
- `book`
- `article`
- `meeting`
- `quote`
- `conversation`
- `other`

#### interpretations

- `id INTEGER PRIMARY KEY`
- `capture_id INTEGER NOT NULL REFERENCES captures(id)`
- `created_at REAL NOT NULL`
- `summary TEXT NOT NULL`
- `insight TEXT`
- `why_it_matters TEXT`
- `possible_application TEXT`
- `counterpoint TEXT`
- `confidence REAL NOT NULL`
- `method TEXT NOT NULL`
- `model_used TEXT`
- `metadata_json TEXT NOT NULL DEFAULT '{}'`

`method` values:

- `deterministic`
- `llm`

#### applications

Internal proposals only. No external actions.

- `id INTEGER PRIMARY KEY`
- `capture_id INTEGER NOT NULL REFERENCES captures(id)`
- `created_at REAL NOT NULL`
- `application_type TEXT NOT NULL`
- `title TEXT NOT NULL`
- `body TEXT NOT NULL`
- `status TEXT NOT NULL`
- `metadata_json TEXT NOT NULL DEFAULT '{}'`

Application types:

- `task_proposal`
- `reminder_proposal`
- `principle`
- `checklist`
- `client_language`
- `investment_rule`
- `health_experiment`
- `writing_idea`
- `decision_rule`

Application statuses:

- `proposed`
- `accepted`
- `dismissed`
- `archived`

`accepted` means accepted inside Wisdom only. It must not create a Hermes task, reminder, Telegram message, file, calendar event, or external action in v1.

#### settings

- `key TEXT PRIMARY KEY`
- `value TEXT NOT NULL`
- `updated_at REAL NOT NULL`

Settings mirror only Wisdom runtime state such as `enabled` and `capture_mode`. Config file values remain the startup/default authority.

### Search Index

If FTS5 is available, create a denormalized FTS table maintained only by `wisdom/db.py`:

```text
wisdom_fts(
  capture_id UNINDEXED,
  original_text,
  cleaned_text,
  title,
  interpretation_text,
  application_text
)
```

Refresh the FTS row after capture, interpretation, and application writes. If FTS5 is unavailable or errors, fall back to parameterized `LIKE` search across the same fields.

Search must find:

- exact original wording
- cleaned wording
- interpretation wording
- application proposal wording

## Exact Original Preservation

The user's exact accepted words are the source of truth.

For:

```text
Remember this: clients don't buy alpha. They buy the feeling that someone sensible is watching the road.
```

Store that string exactly in `raw_events.original_text`.

Do not normalize apostrophes, whitespace, capitalization, punctuation, emojis, or line breaks in `original_text`.

Cleaned text, title, interpretation, and application proposals are annotations. They must never overwrite or replace the original.

`/wisdom original <id>` must return the exact stored original for the capture ID.

Secret-like input is the only exception. If input appears to contain credentials, private keys, cookies, bearer tokens, or auth headers, v1 should reject the capture by default rather than store a redacted value while claiming exact preservation.

## Privacy and Redaction

Implement `wisdom/redaction.py` with two distinct responsibilities:

- `detect_secret_like_text(text)`: decides whether capture should be blocked.
- `redact_for_log(text)`: best-effort masking for errors/debug logs.

Rules:

- Do not log raw capture text.
- Do not store raw Telegram chat IDs, user IDs, message IDs, thread IDs, phone numbers used as IDs, cookies, auth headers, or credentials in metadata.
- Hash session/message/platform identifiers with HMAC-SHA256 and a local Wisdom salt.
- Store the salt outside SQLite at `~/.hermes/wisdom/salt`, mode `0600`, created on first Wisdom DB initialization.
- Never print the salt.
- Avoid over-redacting ordinary personal, business, investing, or health thoughts.
- For explicit `/wisdom capture`, secret-like text should return a concise blocked message without echoing the secret.
- For natural explicit capture triggers, secret-like text should silently skip Wisdom storage and pass through normal chat.

## Capture and Routing

Capture path must be deterministic and must not call an LLM.

Capture modes:

- `off`: only `/wisdom status`, `/wisdom help`, and `/wisdom on` should operate.
- `explicit`: capture only `/wisdom capture <text>` and recognized explicit natural triggers.

Explicit natural trigger prefixes:

- `remember this:`
- `remember this `
- `save this:`
- `save this thought:`
- `note this:`
- `business idea:`
- `investing thought:`
- `health note:`
- `life thought:`
- `book note:`
- `podcast idea:`

Prefix matching is case-insensitive after leading whitespace. Avoid broad "contains" matching in v1; it is too noisy.

Do not capture:

- ordinary chat
- non-Wisdom slash commands
- unknown slash commands
- `/todo`, `/today`, `/remind`, `/voice`, `/new`, `/status`, or any other existing Hermes command
- Telegram topic-root lobby messages

Natural explicit capture should consume the turn and return a short confirmation. It should not forward the capture text to the LLM unless the user sends a separate question.

## Classification

Use deterministic heuristics only in the capture path.

Precedence:

1. Explicit prefix category/source hints.
2. Keyword score by category.
3. `inbox` fallback.

Category hints:

- `business idea:` -> `business`
- `investing thought:` -> `investing`
- `health note:` -> `health`
- `life thought:` -> `life`
- `book note:` -> category by keyword, source `book`
- `podcast idea:` -> category by keyword, source `podcast`

Keyword hints:

- business: client, x10x, pms, aif, report, sales, team, prospect, business, meeting, ops
- investing: stock, market, option, portfolio, risk, macro, thesis, allocation, trade, sizing
- health: sleep, food, energy, exercise, gym, decision quality, health, lunch, cognition
- life: family, relationship, happiness, philosophy, courage, fear, meaning, habit

If tied or uncertain, choose `inbox`.

## Interpretation

Interpretation is optional and must never block capture.

V1 defaults to deterministic interpretation:

- short summary from cleaned text
- optional insight from category/source heuristics
- conservative counterpoint template
- confidence <= 0.6
- `method = deterministic`

LLM interpretation is allowed only for explicit user action (`/wisdom interpret <id>`) or a future config-gated path. It must not run during automatic capture.

If LLM interpretation is enabled:

- Use a strict schema.
- Use `ctx.llm.acomplete_structured()` when called from plugin context, or a small injectable interpreter wrapper.
- Hard timeout: `wisdom.interpret_timeout_seconds`, default 5.
- Validate parsed output before writing.
- On failure, return a concise message for `/wisdom interpret`; do not alter the capture.

Interpretation fields:

- summary
- insight
- why_it_matters
- possible_application
- counterpoint
- confidence

Counterpoint is required for successful LLM interpretations.

## Applications

`/wisdom apply <id>` creates internal application proposals only.

Default v1 application generation should be deterministic templates by category:

- business: `client_language`, `principle`, `task_proposal`
- investing: `investment_rule`, `checklist`, `decision_rule`
- health: `health_experiment`, `decision_rule`
- life: `principle`, `writing_idea`, `decision_rule`
- inbox: `principle`, `writing_idea`

No external action may be taken:

- no old productivity DB writes
- no Hermes todo/reminder writes
- no Telegram sends from Wisdom core
- no calendar/file/browser/terminal actions

## Commands

Add built-in gateway command:

```python
CommandDef("wisdom", "Capture and search Wisdom notes", "Tools & Skills", args_hint="[subcommand]")
```

Command handling should call `wisdom.integration.handle_gateway_command(event, gateway)` from `gateway/run.py` and return a rendered string.

Required subcommands:

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
- `/wisdom help`

Unknown subcommands return concise help.

Rendering rules:

- Keep responses Telegram-friendly and short.
- Limit lists to `wisdom.max_results`.
- Do not show raw hashes.
- Do not show secret-like text.
- `/wisdom original <id>` is the only command intended to return full original text, and only for accepted non-secret captures.

## Gateway Integration

Do not edit `gateway/platforms/telegram.py`.

Minimal gateway edits for v1:

1. Add `/wisdom` to `hermes_cli/commands.py`.
2. Add a small `/wisdom` command branch in `gateway/run.py` after slash command access checks and command hooks, before generic built-in command dispatch.
3. Add a small natural explicit-capture intercept in `gateway/run.py` after topic-root lobby handling and before the active-session sentinel.

Desired natural capture flow:

```text
authorized message
-> existing update/command/skill handling
-> topic-root lobby guard
-> if non-command and wisdom.capture_mode == explicit and prefix matches:
     wisdom.integration.maybe_capture_gateway_event(...)
     return short confirmation
-> else normal AIAgent flow unchanged
```

Fail-open rule:

- If Wisdom raises during ordinary non-command handling, log a minimal redacted warning and continue to normal Hermes agent.
- If Wisdom raises during `/wisdom` command handling, return a concise command error.

Do not use `pre_gateway_dispatch` for v1 capture. It is pre-auth and is the wrong durability boundary.

## Reliability

- Capture must be synchronous enough to confirm only after DB commit.
- Capture must not call LLMs.
- Search and list commands must use bounded result counts.
- DB writes must be transaction-wrapped.
- Errors must not crash the gateway.
- The running launchd gateway requires restart after implementation before new code is loaded.

## Tests Required

Use temp `HERMES_HOME` or explicit temp DB paths for all Wisdom tests.

DB tests:

- initializes DB and tables
- enables WAL or records fallback mode
- migrations are idempotent
- inserts raw event and capture in one transaction
- inserts interpretation
- inserts application proposals
- search works with FTS5
- LIKE fallback works when FTS setup is forced unavailable

Exact preservation:

- original text round-trips exactly, including whitespace and punctuation
- cleaned text does not overwrite original
- `/wisdom original <id>` returns exact original
- secret-like capture is blocked, not redacted into an allegedly exact original

Routing/classification:

- ordinary chat passes through
- explicit prefixes capture
- broad contains-only phrases do not capture
- non-wisdom slash commands are ignored by Wisdom
- topic-root lobby messages are not captured
- `off` disables natural capture
- deterministic category/source rules work

Commands:

- every required `/wisdom` subcommand has a unit test
- unknown subcommand returns help
- result limits are enforced
- archived captures disappear from default inbox

Interpret/apply:

- deterministic interpretation works
- LLM interpreter failure does not mutate capture
- schema-invalid LLM output is rejected
- application proposals are internal only

Privacy:

- chat IDs/user IDs/message IDs are hashed
- raw hashes are not rendered
- logs do not include raw capture text
- no old productivity DB access

Gateway integration:

- `/wisdom` recognized command bypasses active-session queue like other slash commands
- `/wisdom` dispatch returns a string without invoking AIAgent
- natural explicit capture returns confirmation and does not invoke AIAgent
- Wisdom exception during natural capture falls through to AIAgent
- Wisdom exception during `/wisdom` command returns concise error

Safe existing tests to run after implementation:

- `tests/gateway/test_command_bypass_active_session.py`
- `tests/gateway/test_gateway_command_help.py`
- `tests/gateway/test_pre_gateway_dispatch.py`
- any new `tests/wisdom/*`

Do not run tests that contact real Telegram or mutate real `~/.hermes` state.

## Manual Verification After Implementation

Before any live restart:

1. `python -c "import wisdom"`.
2. Run all `tests/wisdom/*`.
3. Run the safe gateway tests listed above.
4. Confirm `~/.hermes/productivity/productivity.db` mtime did not change.
5. Confirm test DBs were under temp dirs.
6. Confirm `docs/wisdom_kernel.md` exists.
7. Confirm `docs/specs/wisdom_kernel_v1.md` reflects implementation reality.
8. Confirm `git diff --stat` shows only intended files.

After explicit user approval to restart the gateway in a later run:

1. Verify `/wisdom status` in Telegram.
2. Verify `/wisdom capture test note`.
3. Verify `Remember this: test note` captures and returns confirmation.
4. Verify `/wisdom original <id>` returns exact text.
5. Verify `/today` and `/remind me tomorrow` are unaffected.

## Critical Review Findings

What was good in the draft:

- The central rule that exact originals are the source of truth is correct.
- The old productivity DB was correctly excluded.
- The draft correctly separated originals, interpretations, and applications.
- Fail-open behavior and no external actions were the right defaults.
- The command list and test categories were a strong starting point.

What was risky:

- It treated `pre_gateway_dispatch` as a possible capture point even though repo code shows it runs before auth.
- It allowed `smart` capture as a v1 setting, which creates ambiguity before a safe classifier exists.
- It implied redacting before storage while also requiring exact original preservation. Those conflict.
- It required too many source modules up front, including placeholders that may not be needed.
- It left LLM interpretation too close to the capture path.
- It did not distinguish plugin command integration from built-in command integration.
- It did not specify how natural explicit capture can safely return a confirmation without direct Telegram sends.

What changed and why:

- Gateway integration is now explicitly a tiny built-in command plus post-auth intercept, not `pre_gateway_dispatch`.
- `smart` capture was removed from v1.
- Secret-like input is blocked rather than silently redacted into `original_text`.
- Capture statuses were simplified to `active` and `archived`.
- `raw_events` now owns the exact original; captures reference it to avoid inconsistent duplicate originals.
- LLM interpretation is explicit-only and never part of capture.
- Config moved to `config.yaml` as the primary surface.
- Manual verification now includes old productivity DB mtime checks and launchd restart expectations.

What remains uncertain:

- Whether `/wisdom` should also be exposed in the classic CLI in v1. Telegram is the primary target, so CLI can be deferred unless implementation is trivial.
- Whether natural explicit capture should consume the message or also let the agent answer. This spec chooses consume-and-confirm for predictability.
- Whether LLM interpretation should be enabled by default later. This spec defaults to deterministic interpretation.

Must be confirmed before implementation:

- Confirm no newer source-backed productivity code has appeared; Wisdom must remain independent regardless.
- Confirm with the user before restarting `ai.hermes.gateway` after implementation.
- Confirm whether `/wisdom` should be added to Telegram's command menu immediately or only work as typed text.

## Implementation Plan for Next Run

Use this exact high-level plan for the implementation `/goal`:

1. Implement the `wisdom/` package:
   - `config.py`: resolve config from `DEFAULT_CONFIG`, env test overrides, and `get_hermes_home()`.
   - `models.py`: typed dataclasses/enums for commands, captures, interpretations, applications, and route results.
   - `redaction.py`: secret detection, log redaction, stable HMAC hashing, salt creation.
   - `db.py`: SQLite connection, schema versioning, WAL fallback, transactions, CRUD, FTS/LIKE search.
   - `classify.py`: deterministic trigger/category/source classification.
   - `capture.py`: exact capture transaction and secret-block behavior.
   - `retrieve.py`: inbox/search/original retrieval.
   - `interpret.py`: deterministic interpretation plus injectable optional structured interpreter.
   - `apply.py`: deterministic internal application proposals.
   - `render.py`: Telegram-friendly rendering.
   - `commands.py`: parse and execute `/wisdom` subcommands against core services.
   - `integration.py`: gateway-safe wrappers for command handling and natural explicit capture.
2. Add `wisdom` defaults to `hermes_cli/config.py::DEFAULT_CONFIG`.
3. Add `CommandDef("wisdom", ...)` to `hermes_cli/commands.py`.
4. Add the minimal `/wisdom` command branch and natural explicit-capture intercept in `gateway/run.py`.
5. Add `docs/wisdom_kernel.md`.
6. Add `tests/wisdom/*` and gateway integration regression tests.
7. Run targeted Wisdom tests and safe gateway command tests.
8. Verify no real productivity DB access or mtime change.
9. Commit with message `Implement Hermes Wisdom Kernel v1` only after tests pass.

The next implementation prompt should say, at a high level:

> Implement Hermes Wisdom Kernel v1 exactly according to `docs/specs/wisdom_kernel_v1.md`. Keep the implementation source-backed, independent of the old productivity DB, Telegram-first via `/wisdom` plus explicit natural capture, deterministic by default, exact-original preserving, fail-open, and covered by temp-DB tests. Do not restart Hermes or send live Telegram messages unless explicitly approved.

## Future V2 Ideas

- `smart` capture mode
- voice note transcript capture using existing STT output
- topic-specific capture modes
- weekly and monthly reviews
- skeptic/challenge mode
- theme graph and recurring patterns
- first-class principles/checklists
- internal task/reminder execution through a new source-backed task system
- optional one-time import from old productivity DB after source recovery
- Apple Notes/Readwise/Notion/Obsidian import/export
- semantic embeddings
- dashboard/export
