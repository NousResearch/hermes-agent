# Pending Interaction Handoff

## Behavior

Hermes records a small profile-local pending interaction when a Discord-visible agent or cron response appears to ask the user for input. A later non-command Discord message in the same channel or thread is checked against open records before the agent runs. When exactly one record matches, Hermes prefixes the user turn with a visible handoff block that includes the pending id, origin profile, source session id or cron job id, question summary, expected reply shape, and artifact paths.

The handoff text explicitly tells the agent to treat the visible pending interaction as primary context and not let runtime recall override it unless the user says so. This keeps stale Honcho recall from hijacking replies like `그거 계속해줘`.

When multiple open records match the same visible Discord location, Hermes does not resolve any of them. It prefixes the turn with an ambiguity block and asks the agent to clarify which pending item the user means. Expired records are marked `expired` and are not used for handoff.

## Storage

Records are stored under the active profile's Hermes data directory:

```text
${HERMES_HOME}/pending_interactions/records.json
```

The record schema is:

```text
id
origin_profile
platform
channel_id
thread_id
source_session_id
job_id
question_summary
expected_reply_shape
artifact_paths
created_at
expires_at
status
```

`source_session_id` is used for gateway and `/goal` continuations. `job_id` is used for cron deliveries. Records default to a 24 hour TTL.

## Limits

This is not a workflow engine. It does not run autonomous bot-to-bot loops, does not write persistent user memory, and does not promote anything to Obsidian. Detection is intentionally minimal: responses with clear question or input-request markers create records; ordinary completion messages do not. Matching is limited to Discord channel/thread visibility.

The JSON store is profile-local and written atomically for normal gateway use, but it is not a durable cross-host queue.

## Verification

Targeted tests cover cron-to-Discord follow-up, `/goal` continuation handoff, ambiguous Korean replies, multiple pending interactions, expiry, and stale runtime recall not overriding a visible pending interaction.

Verified with:

```bash
scripts/run_tests.sh tests/gateway/test_pending_interactions.py tests/cron/test_pending_interactions.py tests/cron/test_scheduler.py tests/gateway/test_discord_channel_prompts.py -q
```
