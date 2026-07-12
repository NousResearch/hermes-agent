# WhatsApp Delivery Reliability Plan

**Status:** Complete — all four tasks implemented and verified (focused suite + legacy auth/media tests green).

**Goal:** add safe retry classification, idempotency metadata and dead-letter observability to Hermes→WhatsApp bridge delivery without duplicating ambiguous sends or violating Sawi outreach policy.

## Constraints
- TDD strict.
- No production/config/credential changes.
- No retries after ambiguous timeout unless delivery absence is provable.
- Retry only connection-refused before request acceptance and explicit 429/502/503/504 responses.
- Never retry 400/401/403/404 or unknown exceptions.
- Preserve existing adapter interfaces and standalone cron delivery.
- No message body, phone, token or PII in dead-letter logs.
- Do not embed Sawi-specific DDD19/30-day policy in upstream generic Hermes; expose a policy callback/hook or metadata so profile code can enforce it separately.

## Task 1 — Pure retry classifier
- Create a pure helper under WhatsApp plugin module.
- Return retryable/non-retryable/ambiguous plus sanitized category.
- Tests first for HTTP classes, connection refusal, timeout, unknown exception.

## Task 2 — Bounded delivery attempts
- Extend adapter mutating POST helper to use max 3 attempts with injectable sleep/backoff `[1,5]` and jitter disabled in tests.
- Generate one idempotency key per logical delivery and reuse it across attempts via `Idempotency-Key` header.
- Keep current auth headers.
- A 2xx ends attempts; a permanent or ambiguous failure stops immediately.
- Tests prove no retry on timeout/401/400 and bounded retry on 429/503/connection-refused.

## Task 3 — Sanitized dead-letter ledger
- Add local JSONL ledger utility under Hermes state/output path, configurable and disabled by default for upstream.
- Record timestamp, platform, route, idempotency key hash, attempt count, sanitized error category/status and resolution state; never content/chat/token.
- Atomic append with file lock appropriate for Windows.
- Tests scan output for phone/email/Bearer/message text leakage.

## Task 4 — Scheduler standalone parity
- Reuse the same retry classifier/ledger in standalone cron WhatsApp delivery rather than copy logic.
- Preserve `last_delivery_error`; add machine fields for category, attempts and dead-letter reference.
- Tests prove 401 is permanent, 503 reaches DLQ after 3 attempts, and a later 200 avoids DLQ.

## Verification
- Focused WhatsApp tests.
- Scheduler delivery tests.
- Existing auth/media tests.
- Full related suite.
- Commit each task separately; no push/merge.
