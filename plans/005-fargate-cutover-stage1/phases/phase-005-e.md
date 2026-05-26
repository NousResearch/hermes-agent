# Phase 005-E: Cloud-side smoke test (Slack mention → Neon round-trip)

## Goal
Prove cloud Hermes handles a real Slack interaction end-to-end and the resulting rows land in Neon as expected (no regressions vs. local).

## Context
Plan 007-E already verified the local-Hermes write path. This phase verifies the same path works from cloud Hermes. Both gateways (local + cloud) are listening to the same Slack workspace simultaneously — expected duplicate replies are FINE for this phase only.

## Dependencies
- Phase 005-D complete (Fargate task RUNNING + Socket Mode connected)
- Blake at Slack

## Scope

### Files to Create
None.

### Files to Modify
None — verification only.

### Explicitly Out of Scope
- Single-listener invariant (that's Phase 005-F's job)
- Load testing (deferred to Phase 005-G's soak)
- Multi-channel scenarios

## Implementation Notes

1. **Expect duplicate replies** — both local launchd and cloud Fargate are listening. This is by design for the duration of phases D+E. Phase F kills the laptop instance.
2. **Identify cloud vs. local** in the Neon rows by `created_at` timestamps + cross-reference with CloudWatch logs. Cloud responses should have a corresponding CloudWatch `inbound message` log line.
3. **If ONLY local replies arrive (no cloud)**, cloud Socket Mode is broken — roll back via `desired-count 0` and debug.

## Acceptance Criteria
- [ ] Slack `@Hermes UAT 005-E cloud test` → at least one reply arrives within 10s
- [ ] CloudWatch `/ecs/hermes-saas` shows `inbound message: ... msg='UAT 005-E cloud test'` matching the Slack message ts
- [ ] CloudWatch shows the corresponding `response ready` + `Sending response` log lines
- [ ] Neon `messages` has new rows in the last 5 minutes (both user + assistant for the cloud path)
- [ ] Neon `raw_events` has new `slack_inbound` + `slack_outbound` rows for the cloud path
- [ ] React 👍 to the cloud reply → `skill_feedback` row appears in Neon within 5s
- [ ] Remove the 👍 → row deleted from `skill_feedback` (idempotency confirmed)

## Verification Steps

```bash
# Blake: post @Hermes UAT 005-E cloud test in any channel
# Wait for reply, react 👍, then run:

PROFILE='AgenticHub-162471567408'
TENANT='ac85d33a-c466-4d4c-9747-0a8d69efbe6f'

# CloudWatch — cloud Hermes processed the message?
aws logs tail /ecs/hermes-saas --since 5m --profile $PROFILE | grep -E "UAT 005-E|response ready|Sending response"

# Neon — rows landed?
/opt/homebrew/opt/libpq/bin/psql "postgresql://hermes_app:w3fdElmBnKwUOGfrvchBjyLDlj7RMMpF@ep-weathered-credit-aqq9kjyf.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require" -c "
SELECT set_config('app.tenant_id', '$TENANT', false);
SELECT 'messages' AS tbl, COUNT(*) FROM messages WHERE created_at > NOW() - INTERVAL '5 min'
UNION ALL SELECT 'raw_inbound', COUNT(*) FROM raw_events WHERE recorded_at > NOW() - INTERVAL '5 min' AND event_kind='slack_inbound'
UNION ALL SELECT 'raw_outbound', COUNT(*) FROM raw_events WHERE recorded_at > NOW() - INTERVAL '5 min' AND event_kind='slack_outbound'
UNION ALL SELECT 'skill_feedback', COUNT(*) FROM skill_feedback WHERE reacted_at > NOW() - INTERVAL '5 min';
"
```

## Status
BLOCKED — 2026-05-26 — infrastructure debt from Plan 001-E

### What happened
- Cloud Hermes successfully received UAT 005-E message + reaction (verified in CloudWatch + skill_feedback row landed)
- But cloud Hermes couldn't REPLY because LLM call returned HTTP 401: "Missing Authentication header"
- Investigation chained through 3 nested pre-existing infra gaps:
  1. Portkey image missing from ECR (fixed tonight by pulling upstream `portkeyai/gateway:latest`)
  2. Portkey has no provider config (vanilla install, no virtual-key → provider mapping)
  3. ALL LLM-routing secrets in Secrets Manager are placeholders (`PLACEHOLDER...`, `bed...`)
- Cloud Hermes scaled back to `desiredCount=0` to stop being a silent zombie burning AWS dollars
- Local launchd Hermes restored as the sole working Hermes

### Why this isn't fixable inside Plan 005
The cloud LLM-routing layer was scaffolded by Plan 001-E but never finished. No service in cloud has ever successfully made an LLM call. Bringing it to working state requires:
- Decision: keep Portkey (author full provider config) OR remove Portkey (wire direct Anthropic/Bedrock per service)
- Populating real credentials in Secrets Manager
- Possibly building + pushing Rooben image (same gap as Portkey)
- Coordinating with Atlas's Portkey usage (same root cause; Atlas in cloud is also probably broken)

This is its own focused plan. Tracked as **Plan 009 — Cloud LLM Routing Infrastructure**.

### What got fixed tonight (kept for next attempt)
- v8 image with all Plan 004-A + Plan 007 code is in ECR (`plan-001-E-amd64-v8`)
- Build script patched to enforce `--platform linux/amd64` (committed)
- Task def revisions registered: :9 (v8 image), :10 (+PYTHONUNBUFFERED), :11 (+-v flag), :12 (+SLACK_ALLOWED_USERS)
- Portkey image pushed to ECR (no config but image is there)
- Three discoveries that need permanent fixes in Dockerfile.saas/entrypoint.saas.sh (tracked in Plan 009-E)
- One discovery requiring kanban dispatcher disable in cloud config (Plan 009-F)
