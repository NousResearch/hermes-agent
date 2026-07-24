---
name: cron-operations-guardrails
description: Diagnose and harden Hermes cron operations against watchdog loops, mode drift, provider failures, schedule collisions, and oversized completion responses.
version: 0.1.0
author: Community Contributor
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [cron, reliability, monitoring, gateway, model-routing, operations]
    related_skills: [hermes-agent]
    requires_toolsets: [cronjob]
---

# Hermes Cron Operations Guardrails

Use this skill when Hermes scheduled jobs are failing intermittently, producing repeated alerts, changing model behavior after configuration updates, colliding at the same minute, or truncating long report responses.

## Safety Boundary

- Start read-only: inspect job definitions, execution metadata, persisted run artifacts, gateway logs, and active model configuration before changing anything.
- Never print or persist API keys, platform tokens, chat IDs, private prompts, or credentials.
- Treat model/provider/schedule settings as operator-controlled policy.
- Drift monitors should alert but must not automatically rewrite configuration unless the user explicitly requests a specific change.
- Resolve paths through the active Hermes profile/configuration. Do not assume the default `~/.hermes` location in reusable scripts.

## Triage Procedure

### 1. Inventory jobs and scheduler health

Run the supported CLI first:

```bash
hermes cron status
hermes cron list --all
hermes status --all
```

Record, without exposing private prompt text:

- Job ID and sanitized name
- Enabled/paused state
- Schedule and timezone
- Agent versus no-agent mode
- Script presence
- Explicit provider/model pin
- Delivery target type
- Last status and last/next run time

Do not diagnose a complex job from a truncated prompt preview. Read the complete local definition only when necessary and redact it before sharing.

### 2. Classify each failure

Use these categories:

1. Scheduler did not trigger.
2. Script/process failed.
3. Agent/model invocation failed or was skipped.
4. Artifact generation succeeded but final response failed.
5. Delivery failed after successful execution.
6. Watchdog produced a false positive or failed itself.
7. Remote provider capacity/authentication failure.

Read the complete persisted run artifact and nearby gateway logs. A short chat-delivered error snippet is not sufficient evidence.

### 3. Verify provider attribution

For capacity, quota, authentication, or rate-limit errors, identify:

- Provider selected for the failed attempt
- Model identifier
- Remote endpoint or adapter
- Whether the failure occurred on the primary route or a fallback
- Whether multiple jobs reached the same provider concurrently

Do not infer local CPU, memory, or worker exhaustion from an error containing words such as `local worker` unless host process/system evidence confirms it.

## Reliable No-Agent Watchdog Pattern

For platform health checks and cron failure monitors:

- Healthy: print nothing and exit 0.
- Alert successfully generated: print one concise alert and exit 0.
- Monitor implementation failure: exit non-zero.
- Exclude the monitor itself and sibling watchdogs by stable job ID.
- Require consecutive failed probes before paging when transient failures are common.
- Latch repeated alerts and reset the latch only after recovery.
- Match specific critical patterns in a bounded recent time window; do not alert on every warning mentioning a platform.

Why: if an alerting watchdog exits non-zero, Hermes marks the watchdog as failed. A failure monitor that scans itself can then create an endless self-alert loop.

## Agent and Script Mode Guardrails

### Deterministic script jobs

Use no-agent mode when script stdout is already the exact content to deliver and no LLM reasoning is required.

Verify:

- `no_agent=true`
- Script exists and is executable/readable as appropriate
- Healthy execution may intentionally produce empty stdout
- Secrets needed by the script are available through the supported environment/configuration path

### Agent-driven jobs

Use agent mode when interpretation, selection, summarization, or other reasoning is required.

For reproducible or cost-sensitive scheduled workloads:

- Explicitly pin provider and model.
- Attach only required skills/toolsets.
- Keep smart/hidden model routing disabled unless intentionally approved.
- Smoke-test the primary model and approved fallbacks after routing changes.

A script attached to an agent job may provide context to the model; it is not equivalent to a no-agent script-only job.

## Model and Configuration Drift Checks

Maintain an operator-approved baseline containing only non-secret policy values:

- Primary provider/model
- Ordered fallback providers/models
- Whether smart model routing is allowed
- Jobs required to be no-agent
- Jobs required to pin provider/model
- Critical script identities or hashes when appropriate

A drift check should report differences and suggested remediation, but remain read-only.

## Schedule Collision Checks

Group enabled jobs by effective start minute and flag clusters containing multiple expensive jobs.

Prefer:

- Moving low-priority heartbeat/monitor jobs by 5–10 minutes
- Separating large research jobs
- Limiting retries against a saturated provider
- Using per-provider concurrency controls when available

Do not increase concurrency limits until provider-side and host-side capacity have been distinguished with evidence.

## Long Artifact and Output-Truncation Pattern

For reports, transcripts, presentations, or audio:

1. Generate the artifact.
2. Write it to a durable path.
3. Verify that the artifact exists and is non-empty.
4. Deliver the artifact.
5. Return a short completion receipt containing status and artifact reference.
6. Do not repeat the complete report/transcript in the final cron response.

If increasing `max_tokens` merely causes the output to hit the new exact ceiling, inspect whether the final response duplicates already-created artifacts.

## Post-Fix Verification

After an approved change:

1. Run syntax/static checks for modified scripts.
2. Run the script manually with sanitized output expectations.
3. Trigger the target cron job once using the supported CLI/tool.
4. Confirm the persisted execution status and full run artifact.
5. Confirm delivery separately from execution success.
6. Re-run the monitor and verify that healthy mode is silent.
7. Confirm the monitor does not include itself in findings.
8. Confirm model/provider routing from live runtime data, not memory.

## Reporting Template

```markdown
### Incident
- Symptom:
- Affected job class:
- First/last observed:

### Evidence
- Scheduler status:
- Full artifact result:
- Provider/model/endpoint:
- Delivery result:

### Root cause
- Verified:
- Inferred:
- Not established:

### Remediation
- Change made:
- Safety boundary:

### Verification
- Manual script test:
- Cron rerun:
- Delivery test:
- Monitor silent-health test:
```

## Common Pitfalls

- Treating every gateway warning as an outage.
- Returning exit 1 after successfully emitting a watchdog alert.
- Allowing a failure monitor to scan its own status.
- Assuming scheduled jobs inherit a full interactive shell environment.
- Confusing a remote provider capacity error with local host exhaustion.
- Leaving cost-sensitive agent jobs unpinned unintentionally.
- Starting several heavy jobs in the same minute.
- Reporting cron `completed` as proof that external delivery succeeded.
- Repeating a large generated artifact in the final response.
- Automatically repairing model routing or schedules without operator approval.
