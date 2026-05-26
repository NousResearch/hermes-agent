# Status — Plan 007: Sessions to Neon

**Status:** COMPLETE 2026-05-25
**Last updated:** 2026-05-25
**Blocked by:** None
**Blocks:** Nothing — Plan 005 (Fargate cutover) is now unblocked

## Final summary

Plan 007 shipped end-to-end. Every Slack interaction now durably persists to Neon:
- User + assistant message content → `messages` table (Plan 007-C)
- Full event payloads (post-redaction) → `raw_events` table (Plan 007-D)
- 👍/👎 reactions → `skill_feedback` table (Plan 004-A, now wired end-to-end)
- Skill output correlation → `skill_output_map` table (Plan 004-A)

Live UAT 007-E results (2026-05-25 22:14 round-2 interaction):
- 19 messages rows | 18 outbound audit + 1 inbound audit | 1 reaction | 19 skill_output_map
- Restart-survival proven: cold-start probe loaded 5 messages from Neon using explicit tenant_id

7 commits on `feat/plan-004-self-improvement` branch:
- `6d341a7` 007-A migration
- `3d8a300` 007-B NeonBackend.append_raw_event + dedup-fix
- `5fca860` 007-C Slack adapter → Neon messages
- `c0fd092` 007-C docs
- `54738a1` 007-D raw_events audit hooks
- `cdffedd` 007-D SlackResponse.data fix
- (current) 007-E get_conversation_history fix

External actions Blake completed: applied Neon migrations 006+007 via owner DSN (inline only, not persisted), added `reactions:read` + reaction event subscriptions to the Slack app, reinstalled to workspace.

## Phase Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 007-A | Migration + schema (raw_events + RLS) | **Complete (2026-05-25)** | Migration applied to live Neon; raw_events with RLS + grants verified; 4 acceptance tests pass (insert, isolation, dual-tenant, idempotency) |
| 007-B | NeonBackend session-entry CRUD + raw_events write | **Complete (2026-05-25)** | 1 new method (`append_raw_event`) + 4 unit tests + idempotency bug fix in migration 007. Live integration verified. |
| 007-C | Slack adapter → Neon messages (saas mode) | **Complete (2026-05-25)** | Pivoted from SessionStore wiring to Slack-adapter hooks; both user + assistant turns persist via probe-verified path. JSONL-drop deferred to Plan 005. |
| 007-D | Raw audit hooks (inbound + outbound) — Slack adapter sites | **Complete (partial, 2026-05-25)** | Inbound+outbound shipped + verified end-to-end. Tool-call hook DEFERRED (sync/async bridge in tool_executor + Plan 006 overlap). |
| 007-E | End-to-end UAT (Slack → Neon round-trip + restart survival) | **Complete (2026-05-25)** | All Neon row counts green; surfaced + fixed 3 bugs (SlackResponse.data, reactions:read Slack scope, get_conversation_history cold-start) |

## Resumption context

- Next phase: 007-D (in progress) — raw_events audit hooks
- Adaptations recorded below — Phases A/B/C all completed with adapted scope per discovered ground truth.

## Open Questions (none blocking)

- Whether `SQLiteBackend` should stub the new methods or raise `NotImplementedError` — left to implementer's judgment in Phase 007-B per Implementation Note #5
- Whether to remove `SessionDB` (`hermes_state.py`) after Plan 007 settles — deferred; keep for rollback safety

## Adaptations log
_(records any mid-execution deviations from the phase files)_

### 2026-05-25 — Phase 007-A scope reduction after schema inspection

When Phase 007-A inspected the live Neon DB, found that Plan 001-D already shipped MORE than the master plan credited:

- The table called `sessions` in Plan 007's master plan is actually `conversations` (Plan 001-D semantics). Same concept; just renamed in spec to align.
- `messages` table already exists with `(id, conversation_id, tenant_id, user_id, role, content, tool_calls, metadata, created_at)` + RLS policy. The SQLite-extras (`tool_name`, `tool_call_id`, `reasoning`, `reasoning_content`, `reasoning_details`, `codex_reasoning_items`, `codex_message_items`) do NOT exist as columns — but will be stuffed into the `metadata` JSONB on write. **No ALTER TABLE needed.**
- `NeonBackend` already has: `get_or_create_conversation`, `append_message`, `get_conversation_history`, `search_sessions`, `_get_or_create_user`. **Phase 007-B's "5 new methods" shrinks to 1: `append_raw_event`.**

**Net effect:**
- Phase 007-A → only the new `006_raw_events.sql` migration (no other DDL needed)
- Phase 007-B → only the `append_raw_event` method + 1 test (was: 5 new methods + 5 tests)
- Phase 007-C → unchanged but now uses existing `get_or_create_conversation` + `append_message` rather than new methods
- Phase 007-D → unchanged
- Phase 007-E → unchanged (UAT verifies the wiring, not new code)

Estimated effort revised down from ~2 days to ~1 day total.
