# Phase 007-E: End-to-end UAT

## Goal
Validate the full Slack → Neon write path under realistic load: every message and tool call lands in Neon, conversation memory survives gateway restart, RLS isolation holds, and Plan 005 (Fargate cutover) is unblocked.

## Context
Phases A–D shipped code. This phase is verification-only: a fresh Slack interaction, a gateway bounce, a re-query, and an RLS sanity check. If E passes, Plan 005 can begin.

## Dependencies
Phases 007-A, 007-B, 007-C, 007-D all Complete and committed.

## Scope

### Files to Create
None (verification only).

### Files to Modify
None. (Any test failures here trigger fixes in earlier phases, not new code in 007-E.)

### Explicitly Out of Scope
- Load testing beyond a single interaction (out of scope for this plan; revisit during Plan 005's 48h soak)
- Migrating filesystem sessions into Neon
- Cross-account / cross-tenant scenarios beyond the 2-tenant fixture from Phase 007-B

## Implementation Notes

1. **Use the live Slack workspace** (`T0B16FV0KFF`, tenant `ac85d33a-c466-4d4c-9747-0a8d69efbe6f` from Plan 004-A). No need to simulate.
2. **Run with the launchd gateway**, not a debug shell — this exercises the real path including secret redaction, RLS, and the asyncio bridge from sync `SessionStore` to async `NeonBackend`.
3. **Restart test is the key signal.** If a gateway bounce loses the in-flight conversation transcript, Plan 005 stays blocked. This is the load-bearing assertion of the entire plan.
4. **No new code allowed.** If anything fails, the fix lives in 007-A/B/C/D. Reopen the relevant phase.

## Acceptance Criteria
- [ ] Send `@Hermes UAT test 007-E from Blake` in Slack; receive a reply
- [ ] React 👍 to the reply
- [ ] Neon row count summary returns: `sessions ≥ 1`, `messages ≥ 2`, `raw_events ≥ 3`, `skill_feedback ≥ 1`, `skill_output_map ≥ 1` (all within the last 5 minutes)
- [ ] Bounce the gateway (`launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`)
- [ ] In the SAME Slack thread (no re-mention needed if F.1 threading fix holds, or re-mention if not), send a follow-up: `@Hermes what did I just say?`
- [ ] Reply demonstrates the agent loaded the prior turn's content from Neon (proves restart-survival of session transcript)
- [ ] `~/.hermes/sessions/` contains no files from the test session (saas mode write skipped)
- [ ] 2-tenant RLS sanity: set `app.tenant_id` to a different tenant UUID, query `messages` → 0 rows for this session_id (proves RLS isolation holds for the new write path)

## Verification Steps

```bash
# After Slack interaction + reaction, run as hermes_app under tenant T0B16FV0KFF
TENANT='ac85d33a-c466-4d4c-9747-0a8d69efbe6f'
DSN='postgresql://hermes_app:w3fdElmBnKwUOGfrvchBjyLDlj7RMMpF@ep-weathered-credit-aqq9kjyf.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require'

/opt/homebrew/opt/libpq/bin/psql "$DSN" -c "
SELECT set_config('app.tenant_id', '$TENANT', false);
SELECT 'sessions' AS tbl, COUNT(*) FROM sessions WHERE updated_at > NOW() - INTERVAL '5 minutes'
UNION ALL SELECT 'messages', COUNT(*) FROM messages WHERE created_at > NOW() - INTERVAL '5 minutes'
UNION ALL SELECT 'raw_events', COUNT(*) FROM raw_events WHERE recorded_at > NOW() - INTERVAL '5 minutes'
UNION ALL SELECT 'skill_feedback', COUNT(*) FROM skill_feedback WHERE reacted_at > NOW() - INTERVAL '5 minutes'
UNION ALL SELECT 'skill_output_map', COUNT(*) FROM skill_output_map WHERE registered_at > NOW() - INTERVAL '5 minutes';
"

# RLS isolation sanity (use a different tenant_id)
OTHER_TENANT='eba76ace-7093-489d-adbe-70166ce7d1da'  # TTEST_98a4d7ce from prior test fixtures
/opt/homebrew/opt/libpq/bin/psql "$DSN" -c "
SELECT set_config('app.tenant_id', '$OTHER_TENANT', false);
SELECT COUNT(*) FROM messages WHERE created_at > NOW() - INTERVAL '5 minutes';
"
# Expected: 0 — the rows belong to T0B16FV0KFF, not TTEST

# Filesystem check
ls -la ~/.hermes/sessions/ | wc -l
# Expected: roughly same count as before the test (no new session files in saas mode)
```

## Status
Not started
