# OTel schema for Hermes conversation compaction

> Status: design contract for Hermes + `hermes-otel` instrumentation
> Source paths: `agent/turn_context.py`, `agent/conversation_loop.py`,
> `agent/conversation_compression.py`, `hermes_state.py`,
> `hermes_cli/plugins.py`
> Last updated: 2026-06-30

## Why this exists

Hermes context compaction changes the live prompt that future model calls see.
Today, a trace consumer can observe ordinary LLM requests, tool calls, turn
summaries, and errors, but it cannot reliably answer:

- Which user turn triggered compaction?
- Was compaction preflight, provider-error recovery, payload-too-large
  recovery, or manual `/compress`?
- Did the session id rotate or compact in place?
- How many messages/tokens were dropped from live context, and how many remain
  durable in `state.db`?
- Did the summarizer fail, fall back, or skip because another path held the
  compression lock?

This document defines the telemetry shape that should answer those questions
without changing compaction semantics and without leaking transcript contents.

## Runtime facts the schema must model

### Stable turn and API attempt IDs already exist

`agent/turn_context.py` increments `agent._user_turn_count` at the top of a
turn and the conversation loop derives request ids as
`{turn_id}:api:{api_call_count}`. Existing observer hooks already receive:

- `session_id`
- `task_id`
- `turn_id`
- `api_request_id`
- `api_call_count`
- `platform`, `model`, `provider`, `base_url`, `api_mode`

Compaction telemetry must reuse these identifiers instead of inventing a second
correlation scheme. Consumers should treat the string format as opaque and join
on the explicit fields.

### Compaction has multiple triggers

Observed call paths:

1. Preflight compression in `agent/turn_context.py`, before an API request, when
   the rough request estimate crosses the compressor threshold.
2. Provider `413` / payload-too-large recovery in `agent/conversation_loop.py`.
3. Provider context-overflow recovery in `agent/conversation_loop.py`.
4. Manual `/compress`, via the existing `_compress_context` path with
   `force=True` semantics at the call site.

The schema must record the trigger because it changes interpretation. A
preflight compaction avoided a failed provider call; a provider-error
compaction happened after a failed request and should be correlated to that
`api_request_id`.

### Session identity can either rotate or stay stable

`agent/conversation_compression.py` supports two modes:

- Legacy rotation: `end_session(old_id, "compression")`, create a child session,
  then continue on a new `session_id` with `parent_session_id=old_id`.
- In-place compaction: keep the same session id and call
  `SessionDB.archive_and_compact(session_id, compressed)`.

In-place compaction is non-destructive. `hermes_state.py` soft-archives active
messages with `active=0, compacted=1`, inserts compacted live rows with
`active=1`, and keeps archived rows discoverable by session search. Telemetry
must not imply that data was deleted just because it left the live prompt.

### Compaction can no-op or abort

`compress_context()` returns the original message list when:

- a compression lock is already held by another path for the same session;
- the summarizer aborts and sets `_last_compress_aborted`;
- the compressed output is not shorter and the caller cannot reduce context
  further.

No-op outcomes are operationally important and should be first-class statuses,
not missing spans.

## Event model

Emit a single span or event named:

```text
gen_ai.conversation.compacted
```

Recommended placement:

- As a child of the current turn span when compaction happens inside a user turn.
- As a child of the current API attempt span when the trigger is provider-error
  recovery and a failed `api_request_id` exists.
- As a root-like session event only if manual/background compaction runs outside
  a turn; still include `session_id` and any known `turn_id`.

The core Hermes hook surface does not currently include a dedicated compaction
observer hook. The preferred implementation is to add an observer-only hook,
for example `conversation_compacted`, and have `hermes-otel` translate it into
`gen_ai.conversation.compacted`. Avoid inferring compaction from session-id
changes, because in-place compaction intentionally keeps the id stable.

## Required attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `gen_ai.operation.name` | string | Always `conversation.compact`. |
| `gen_ai.conversation.id` | string | Durable active Hermes session id after the event. For legacy rotation this is the child id on success; for in-place it is unchanged. |
| `hermes.session.id_before` | string | Session id before compaction. |
| `hermes.session.id_after` | string | Session id after compaction. Same as before for in-place/no-op outcomes. |
| `hermes.compaction.mode` | string enum | `in_place`, `rotate`, or `none`. `none` is for skipped/aborted no-op outcomes. |
| `hermes.compaction.trigger` | string enum | `preflight_threshold`, `provider_context_overflow`, `provider_payload_too_large`, `manual`, `concurrent_lock`, or `unknown`. |
| `hermes.compaction.status` | string enum | `success`, `skipped`, `aborted`, or `error`. |
| `hermes.compaction.reason` | string | Short machine-readable reason, e.g. `threshold_exceeded`, `lock_contended`, `summary_failed`, `max_attempts_exhausted`. |
| `hermes.compaction.message_count.before` | int | Live message count before compaction attempt. |
| `hermes.compaction.message_count.after` | int | Live message count after the attempt. For no-op, equal to before. |
| `hermes.compaction.message_count.archived` | int | Number of messages removed from live context but preserved durably. `0` for rotation mode unless archived row accounting is known. |
| `hermes.compaction.in_place` | bool | True when `archive_and_compact()` succeeded and the session id did not change. |
| `hermes.compaction.duration_ms` | int | Wall-clock duration of the compaction attempt, including summary generation and DB update. |

## Recommended attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `turn.id` | string | Same value as observer-hook `turn_id`, if known. |
| `hermes.turn.index` | int | `agent._user_turn_count` at the time of compaction, if available. |
| `hermes.task.id` | string | Tool/task scope, especially for subagents and Kanban workers. |
| `hermes.api_request.id` | string | Failed or current API request id when compaction is tied to a provider request. |
| `hermes.api_call.count` | int | API call count from the conversation loop, if known. |
| `hermes.compaction.attempt` | int | Attempt number within the current API-call retry loop. |
| `hermes.compaction.max_attempts` | int | Maximum attempts allowed by that loop, currently `3` in `conversation_loop.py`. |
| `hermes.compaction.tokens.before_estimate` | int | Rough request/input token estimate before compaction. |
| `hermes.compaction.tokens.after_estimate` | int | Rough token estimate for the compacted live context, if cheaply available. |
| `hermes.compaction.threshold_tokens` | int | Compressor threshold used for preflight decisions. |
| `hermes.compaction.context_length` | int | Compressor context length at the time of the decision. |
| `hermes.compaction.protect_first_n` | int | Compressor leading-message protection setting. |
| `hermes.compaction.protect_last_n` | int | Compressor trailing-message protection setting. |
| `hermes.compaction.focus_topic.present` | bool | True if guided compression was requested. Do not record the raw focus text. |
| `hermes.compaction.summary.model` | string | Summarizer model/provider when available. |
| `hermes.compaction.summary.used_main_model_fallback` | bool | True when configured auxiliary compression model failed and main model recovered. |
| `hermes.compaction.summary.error_type` | string | Exception/type category for summary failure. No raw provider body. |
| `hermes.compaction.lock.acquired` | bool | Whether this path acquired the compression lock. |
| `hermes.compaction.lock.contended` | bool | True when compaction skipped because another holder was active. |
| `hermes.compaction.db.active_before` | int | Active rows in `messages` before DB rewrite, if measured. |
| `hermes.compaction.db.active_after` | int | Active rows after DB rewrite, if measured. |
| `hermes.compaction.db.compacted_archived_after` | int | Rows for this session with `active=0 AND compacted=1`, if measured. |

## Status and reason vocabulary

Use stable low-cardinality values.

### `hermes.compaction.trigger`

- `preflight_threshold` — rough request estimate exceeded the configured
  compressor threshold before a provider call.
- `provider_context_overflow` — provider reported context overflow for the
  input prompt.
- `provider_payload_too_large` — provider rejected payload size, including HTTP
  413 recovery.
- `manual` — user or command explicitly requested compression.
- `concurrent_lock` — compression was considered but skipped because another
  path held the session compression lock.
- `unknown` — fallback for older callers.

### `hermes.compaction.status`

- `success` — live context was replaced with a shorter compacted context.
- `skipped` — no compaction was attempted or this path intentionally sat out
  because another path was active.
- `aborted` — summarization failed or produced no usable summary; live context
  is unchanged.
- `error` — unexpected exception escaped the compaction path.

### `hermes.compaction.reason`

Suggested values:

- `threshold_exceeded`
- `provider_context_overflow`
- `provider_payload_too_large`
- `manual_force`
- `lock_contended`
- `summary_failed`
- `not_shorter`
- `child_session_create_failed`
- `db_archive_failed`
- `max_attempts_exhausted`
- `unexpected_exception`

## Privacy and security rules

Compaction is adjacent to full conversation content, so telemetry must be more
conservative than normal request telemetry.

Do not record:

- raw summary text;
- raw pre-compaction or post-compaction messages;
- focus-topic text;
- tool arguments, tool results, terminal output, URLs with credentials, headers,
  cookies, API keys, tokens, passwords, or connection strings;
- provider raw error bodies.

If an error message is operationally useful, pass it through the same redaction
path used for hook payload sanitization and cap it to a short length. Prefer
`error.type`, `status_code`, and low-cardinality `reason` over free text.

## Consumer queries this should support

Examples of questions Honeycomb/OTel consumers should be able to answer:

- Count compactions by trigger and status over time.
- Find turns where provider context overflow happened, compaction succeeded,
  and the retry still failed.
- Compare in-place vs rotation compaction rates during rollout.
- Detect lock contention from background-review forks or concurrent gateway
  paths.
- Detect summary-model failures that recovered via main-model fallback.
- Verify that in-place compaction preserves archived rows while reducing active
  live rows.
- Identify sessions that hit `max_attempts_exhausted` and need `/new` guidance.

## Implementation guidance

1. Add a dedicated observer hook rather than parsing logs or session id changes.
   Suggested hook payload fields should mirror the required/recommended
   attributes above and should include `telemetry_schema_version` automatically
   through the plugin manager.
2. Emit the hook from `compress_context()` at every terminal outcome:
   successful in-place, successful rotation, lock-contended skip, summary abort,
   child-session create failure rollback, and unexpected error.
3. Pass trigger/attempt context from call sites instead of guessing in
   `compress_context()`:
   - preflight path: `trigger=preflight_threshold`;
   - 413 path: `trigger=provider_payload_too_large`;
   - context-overflow path: `trigger=provider_context_overflow`;
   - manual command: `trigger=manual`.
4. Keep the hook observer-only and fail-open. A telemetry plugin failure must
   never block compression, alter the live prompt, or change retry behavior.
5. In `hermes-otel`, create `gen_ai.conversation.compacted` with the same trace
   context as the current turn/API request when available. If no active span is
   available, emit a short standalone span with the session and turn ids.
6. Add tests on both sides:
   - Hermes core: hook fires once for each terminal outcome with sanitized
     fields and no raw content.
   - `hermes-otel`: hook maps to the documented span name and attributes, and
     no transcript text is exported.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| High-cardinality attributes from summaries or focus topics | Record booleans/counts/reasons only; never raw text. |
| Inferring compaction from session id changes misses in-place compaction | Add a dedicated hook and explicit `mode`. |
| Duplicate spans from retry loops | Include `turn_id`, `api_request_id`, `attempt`, and terminal `status`; emit once per compaction attempt outcome. |
| Telemetry changing compaction behavior | Hook is observer-only, fail-open, and invoked after local outcome fields are computed. |
| Archived rows misread as deleted data | Use `message_count.archived`, DB active/compacted counters, and wording that distinguishes live-context removal from durable deletion. |
| Provider raw errors leak secrets or huge bodies | Use low-cardinality `reason` and redacted/capped `summary.error_type`; avoid raw provider body. |
| Plugin/core version skew | Treat unknown attributes as optional; keep `telemetry_schema_version` and make consumers accept additive fields. |
| Runtime overhead on already-large prompts | Avoid walking message contents. Prefer lengths and row counts already known in the compaction path; DB counters are optional/recommended, not required. |

## Open questions

- Should `hermes.compaction.tokens.after_estimate` be computed synchronously, or
  left absent unless the compressor already has it?
- Should manual `/compress` include a separate `hermes.compaction.manual=true`
  convenience attribute, or is `trigger=manual` enough?
- For legacy rotation, should the parent-session `on_session_end(reason="compression")`
  span be linked to `gen_ai.conversation.compacted`, or should compaction itself
  be the only canonical event?
- Should archived-row counters be sampled/optional on very large SQLite
  sessions to avoid write-path overhead?
