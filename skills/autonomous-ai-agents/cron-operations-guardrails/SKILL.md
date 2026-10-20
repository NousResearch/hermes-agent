---
name: cron-operations-guardrails
description: Triage and stabilize failing or drifting Hermes cron jobs.
version: 0.2.1
author: Good Chang (goodchang77) + Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [cron, reliability, monitoring, gateway, model-routing, operations]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
    requires_toolsets: [cronjob]
---

# Hermes Cron Operations Guardrails Skill

Diagnose intermittent cron failures, repeated alerts, routing drift, schedule collisions, and truncated completion responses. This skill keeps diagnosis read-only by default and does not rewrite operator-controlled policy without explicit approval.

## When to Use

Use this skill when:

- Scheduled jobs fail intermittently or do not trigger.
- A watchdog repeatedly alerts on itself or on healthy services.
- Model behavior changes after provider or configuration updates.
- Multiple expensive jobs start in the same minute.
- A report artifact succeeds but the final response is truncated.
- Execution succeeds but delivery fails.

## Prerequisites

- Hermes CLI access for `hermes cron` and `hermes status` commands, run via the `terminal` tool.
- The `cronjob` toolset when managing jobs through Hermes tools.
- Read access to job definitions, persisted run artifacts, and gateway logs.
- Operator approval before changing schedules, providers, models, or routing policy.

Resolve paths through the active Hermes profile and configuration. Do not assume the default `~/.hermes` location in reusable scripts.

## How to Run

Start with the supported read-only CLI commands via `terminal`:

```bash
hermes cron status
hermes cron list --all
hermes status --all
```

Use the `cronjob` tool to inspect job definitions and execution state. Use `read_file` for complete local definitions or persisted artifacts and `search_files` for bounded log searches. Redact private prompt text and credentials before sharing evidence.

Do not diagnose a complex job from a truncated prompt preview or short chat-delivered error snippet.

## Quick Reference

### Safety boundary

- Start read-only and gather evidence before changing anything.
- Never print or persist API keys, platform tokens, chat IDs, private prompts, or credentials.
- Treat model, provider, and schedule settings as operator-controlled policy.
- Drift monitors alert; they do not automatically rewrite configuration.
- Confirm execution and delivery separately.

### Failure classes

1. Scheduler did not trigger.
2. Script or process failed.
3. Agent or model invocation failed or was skipped.
4. Artifact generation succeeded but final response failed.
5. Delivery failed after successful execution.
6. Watchdog produced a false positive or failed itself.
7. Remote provider capacity or authentication failed.

### Reliable no-agent watchdog contract

| Outcome | Stdout | Exit code |
|---|---|---|
| Healthy | Empty | 0 |
| Alert generated | One concise alert | 0 |
| Monitor implementation failed | Diagnostic only | Non-zero |

Exclude the monitor itself and sibling watchdogs by stable job ID. Use consecutive-failure thresholds and a recovery-reset latch when transient failures are common.

### Agent versus script mode

- Use no-agent mode when script stdout is the exact delivery content and no reasoning is required.
- Use agent mode when interpretation, selection, or summarization is required.
- A script attached to an agent job provides context; it is not equivalent to a script-only no-agent job.

## Procedure

### 1. Inventory jobs and scheduler health

Record, without exposing private prompt text:

- Job ID and sanitized name.
- Enabled or paused state.
- Schedule and timezone.
- Agent versus no-agent mode.
- Script presence.
- Explicit provider and model pin.
- Delivery target type.
- Last status and last or next run time.

### 2. Classify the failure

Assign one failure class from the quick reference. Read the complete persisted run artifact and nearby gateway logs before deciding on a root cause.

### 3. Verify provider attribution

For capacity, quota, authentication, or rate-limit errors, identify:

- Provider selected for the failed attempt.
- Model identifier.
- Remote endpoint or adapter.
- Primary route versus fallback route.
- Whether multiple jobs reached the same provider concurrently.

Do not infer local CPU, memory, or worker exhaustion from wording such as `local worker` unless host process or system evidence confirms it.

### 4. Check no-agent watchdog behavior

Verify:

- `no_agent=true` for deterministic script-only monitors.
- The script exists and is readable or executable as appropriate.
- Healthy execution may intentionally produce empty stdout.
- Required secrets are available through supported configuration paths.
- The monitor excludes itself from scans.
- A successful alert exits zero to avoid a self-alert loop.
- Pattern matching is specific and limited to a recent time window.

### 5. Check agent-job routing

For reproducible or cost-sensitive agent jobs:

- Pin the provider and model explicitly.
- Attach only the required skills and toolsets.
- Keep smart or hidden routing disabled unless intentionally approved.
- Smoke-test the primary model and approved fallbacks after routing changes.

Maintain an operator-approved, non-secret baseline containing the primary route, ordered fallbacks, smart-routing policy, required no-agent jobs, required model pins, and critical script identities or hashes.

Report drift and suggested remediation without changing the baseline automatically.

### 6. Check schedule collisions

Group enabled jobs by effective start minute and flag clusters containing multiple expensive jobs.

Prefer:

- Moving low-priority heartbeat or monitor jobs by 5–10 minutes.
- Separating large research jobs.
- Limiting retries against a saturated provider.
- Applying per-provider concurrency controls when available.

Do not increase concurrency limits until provider-side and host-side capacity are distinguished with evidence.

### 7. Handle long artifacts safely

For reports, transcripts, presentations, or audio:

1. Generate the artifact.
2. Write it to a durable path.
3. Verify that the artifact exists and is non-empty.
4. Deliver the artifact.
5. Return a short completion receipt with status and artifact reference.
6. Do not repeat the complete artifact in the final cron response.

If raising `max_tokens` only moves the response to the new exact ceiling, check whether the final response duplicates an already-created artifact.

### 8. Apply an approved remediation

Make the smallest change that addresses the verified root cause. Preserve an evidence trail and avoid unrelated schedule, routing, or delivery changes.

## Pitfalls

- Treating every gateway warning as an outage.
- Returning exit 1 after successfully emitting a watchdog alert.
- Allowing a failure monitor to scan its own status.
- Assuming scheduled jobs inherit a full interactive shell environment.
- Confusing remote provider capacity errors with local host exhaustion.
- Leaving cost-sensitive agent jobs unpinned unintentionally.
- Starting several heavy jobs in the same minute.
- Treating cron `completed` as proof that external delivery succeeded.
- Repeating a large generated artifact in the final response.
- Automatically repairing model routing or schedules without operator approval.

## Verification

After an approved change:

1. Run syntax or static checks for modified scripts.
2. Run the script manually with sanitized output expectations.
3. Trigger the target cron job once with the supported CLI or `cronjob` tool.
4. Confirm the persisted execution status and complete run artifact.
5. Confirm delivery separately from execution success.
6. Re-run the monitor and verify that healthy mode is silent.
7. Confirm the monitor does not include itself in findings.
8. Confirm model and provider routing from live runtime data, not memory.
9. Verify that no credentials or private prompt text appear in reports.

Report with:

```markdown
### Incident
- Severity: [LOW|MEDIUM|HIGH|CRITICAL]
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
