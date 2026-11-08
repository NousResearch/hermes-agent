# Telegram-to-Hermes Bot Room Relay — Implementation Plan

> **For Hermes/Codex:** Implement on a feature branch with strict TDD, independent security review, and no live deployment until the complete test suite passes.

**Goal:** Let only the allowlisted authenticated Telegram private DM submit an explicit task to the existing Hermes Desktop Bot Mode room and return only that task's completed result to the same DM.

**Architecture:** Add a disabled-by-default, narrowly scoped core relay capability and a Telegram-facing tool. Derive platform, chat type, sender, profile, and session from `gateway.session_context`—never from tool arguments. Submit a uniquely identified `@<member>` message to one locally configured room through the hosted-room service. Wait for and return only the correlated discussion's bounded member replies; preserve the existing room-owner authorization unchanged.

**Tech Stack:** Hermes Agent Python gateway, hosted-room service/driver, registered agent tool, pytest.

---

## Security Requirements

- One fixed `room_id`, configured locally; no room ID, owner identity, or authorization parameter accepted from the model/tool caller.
- Enable only for profile `default`, platform `telegram`, chat type `dm`, exactly one nonempty `allowed_senders` string in the local config list, and the configured room's approved initial recipient (`server`). The relay does not restrict later group mentions or follow-ups to other room members; only committed replies from configured profiles in this discussion are returned, prefixed with profile names. The unique nonempty `members` profile list must exactly match the full local served room roster and contain `server`; its source default is `[]`. Revalidate the roster before returning any reply text. The canonical hosted-room service rejects roster changes for a locally owned room; the relay refuses legacy/unowned rooms.
- Initially permit only relay recipient handle `server`. Send ordinary `@server ...` text with no responder ceiling; standard Bot Mode follow-up mentions to teammates remain enabled.
- Do not impersonate the room owner, alter owner records, expose room links, write raw SQLite, or weaken ordinary `groups.*` owner checks.
- Do not expose general `groups.list`/`groups.log`; return only events correlated to this message's generated discussion/thread ID.
- Bot Mode member sessions intentionally retain room history across threads; this relay does not isolate that in-group context. The Telegram response must still contain only this task's correlated output, never unrelated room events.
- Use a stable idempotency key derived from bound inbound Telegram message identity (sender, chat, profile, platform, and message ID), not rotating session keys. Never silently resubmit after an ambiguous result.
- Bound the entire tool call from entry with a fixed maximum wait of 120 seconds. Run blocking hosted-room I/O in one context-copied daemon worker; reject overlapping calls while that worker is still active. A timeout is incomplete/ambiguous, never success.
- Do not auto-approve room member actions. Timeout, failure, or bounded discussion must be reported as incomplete.
- Source defaults stay disabled. No personal sender ID or secret is committed. Local enablement is separate after tests/review.

## Task 1 — Contract tests (RED first)

**Files:**
- Create `tests/gateway/test_hosted_room_telegram_relay.py`
- Create `tests/tools/test_bot_room_delegate.py`
- Extend only if required: `tests/gateway/test_session_group_controls.py`, `tests/gateway/test_session_context.py`

Test denial for non-Telegram, group chat, wrong sender/profile, disabled config, missing context, missing/inactive room driver, missing room membership, and unapproved recipient. Every denial must prove that no room event was appended. Test idempotent retry, completion correlation, timeout/failure behavior, and that unrelated room history is never returned.

Run each focused test and observe the expected failure before implementation.

## Task 2 — Core scoped relay

**Files:**
- Create `gateway/hosted_room_telegram_relay.py` (or justify a smaller existing-module implementation)
- Read/modify, without weakening owner checks: `gateway/session_context.py`, `gateway/session_hosted_service.py`, `gateway/session_group_controls.py`, and `gateway/hosted_rooms.py` (read-only indexed event lookup only; no raw SQL from the relay).

Implement a narrow internal operation that takes trusted request context plus validated recipient/message, checks local relay config and active driver, submits via the canonical hosted-room service, and awaits only the matching discussion's terminal event. Use the existing discussion event/state machinery and canonical `hosted_rooms.get_event`/`read_events` APIs; do not access SQLite directly. Return bounded committed replies plus a stable request ID for later retrieval, or an explicit denied/busy/incomplete/timeout result. Use a fixed `CALL_TIMEOUT_SECONDS = 120`, never a configuration timeout; tests monkeypatch the constant. Separate nonblocking submit and result-read guards ensure a timed-out send cannot block later retrieval.

Run focused tests to GREEN.

## Task 3 — Telegram-facing tool and configuration

**Files:**
- Create `tools/bot_room_delegate.py`
- Modify only the required registration/config files among `tools/registry.py`, `toolsets.py`, `model_tools.py`, `hermes_cli/config_defaults.py`, and `hermes_cli/config_schema.py` (verify exact paths before editing)
- Add focused configuration/registration tests

Expose `bot_room_delegate(recipient, message)` and read-only `bot_room_result(request_id)` tools. Both must derive identity from gateway-bound session context; model arguments cannot choose platform, user, chat, profile, room, owner, RPC method, or token. The result tool must reauthorize the same sender and fixed room, accept only a stable request ID returned by submission, and return only the bounded correlated discussion replies. The tools must be unavailable/denied unless enabled for the configured default profile and exact Telegram DM identity.

Run focused tests and relevant tool/gateway suites.

## Task 4 — Documentation and skill

**Files:**
- Update `website/docs/user-guide/bot-mode.md` and/or the canonical configuration reference with the opt-in relay and its authorization boundaries.
- Update `/root/.hermes/skills/devops/telegram-bot-flotte/SKILL.md` only after implementation and deployment are verified.

Document that ordinary Bot Mode room ownership is unchanged; this is a separate, explicitly allowlisted single-room bridge.

## Task 5 — Independent security review and deployment gate

- Review the diff from a fresh context before commit.
- Run focused tests plus the relevant/full suite; scan additions for owner impersonation, raw database access, broad transcript return, unbounded wait, caller-supplied identity, or secrets.
- Verify defaults are disabled and local allowlist/room configuration remains outside Git.
- Keep the production gateway untouched until code, tests, and review pass. Before live activation, announce the brief gateway interruption and run an end-to-end authorized DM test plus negative authorization tests.

## Timeout caveats

The deadline bounds caller waiting, not the lifetime of blocking service I/O. A send
already in flight may commit after timeout; do not cancel it or automatically retry.
A permanently blocked daemon keeps the relay busy until it exits or the process
ends. Daemon worker guards are process-local;
only the stable `hosted_rooms.user_event_id` key and committed events survive a
restart. Explicit same-message retries call canonical `service.send` again: the
unique event key and immutable content enforce durable idempotency and reject
changed payloads. No in-memory ambiguity set or initial full-log scan is needed. No cross-process cancellation or exactly-once execution is promised.
The caller deadline uses monotonic time and a bounded event wait; Python thread
scheduling and the GIL still preclude a real-time wall-clock guarantee.

## Acceptance Criteria

- From the exact allowlisted Telegram private DM, an explicit `@server` delegation enters the configured Hermes Bot Mode room and the completed correlated result returns to the same Telegram DM.
- Any other platform, chat type, sender, profile, member, disabled config, or unavailable room produces a denial with no room side effect.
- Same inbound message retried does not create a duplicate discussion.
- Unrelated room history is never returned; timeouts/failures are not claimed as success.
- Existing Desktop group access, room-owner checks, and unrelated Bot Mode behavior remain unchanged.

## Independent review hardening

The submit tool returns its stable 64-hex `request_id` after authorization on
completed, incomplete, and timeout outcomes. Timeout before authorization returns
no ID. Retain this ID and call the registered read-only `bot_room_result` tool;
never silently resubmit. Both tools belong to `bot_room_relay` and `hermes-telegram`.
Polling repeats bound-context, local-grant, service, and exact-roster checks, then
uses indexed `hosted_rooms.get_event` to verify the deterministic `message.user`,
exact relay thread, and `@server ` payload. It reads only correlated committed
allowed-member replies. Missing events and unfinished discussions are incomplete,
and an empty settled discussion never claims success.

Each caller path has a separate one-in-flight guard, a context-copied daemon, and
a fixed 120-second hard caller timeout. Wait loops check deadlines between pages.
Completion is rechecked under synchronization after waiting, and only work
finished by the deadline can return a completed result. Blocked submission I/O
may outlive the caller; polling can still recover its eventual result. Normal
`@other` collaboration and all ordinary `groups.*` owner checks remain intact.
