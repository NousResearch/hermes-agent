# Alert notification routing decision

Task: t_2773e8e2
Parent decision: t_46a89f4e confirmed the alert classes and deferred delivery mechanics.
Scope: Hermes Maintenance notification behavior across Matrix, Kanban, GitHub, watchdogs, and general project operation.

## Canonical alert classes

- SILENT/INFO: routine, log-only, or FYI outcomes.
- ATTENTION: awareness or degraded-risk states where automation can continue.
- BLOCKED_HUMAN: concrete active work that cannot proceed without human action.
- INCIDENT: high-severity system integrity, routing, data-loss, security, or blast-radius failures.

`blocked` remains human-only. Non-human waits, retries, CI, worker availability, dependency waits, schedules, and active automated repair stay `waiting`/ATTENTION unless they cross an INCIDENT threshold.

## Routing decision

| Alert class | Matrix/user chat | Kanban | GitHub | Watchdogs/cron | Durable audit |
| --- | --- | --- | --- | --- | --- |
| SILENT/INFO | No proactive user ping. Allow delivery only when user is already subscribed to that task/thread or requested a summary. | Record as event/comment/run metadata when tied to a task. Do not change live status. | No issue/PR/comment unless already part of a PR/test run. | Silent stdout for no-agent watchdogs unless a user explicitly configured heartbeat delivery. | Logs plus Kanban run/event metadata when task-scoped. |
| ATTENTION | One low-urgency notice to the active project room or subscribed task room when the condition affects project awareness but automation can continue. No DM escalation by default. | Use `waiting` for non-human waits or add a comment for degraded-but-continuing states. Never mark `blocked`. | PR/issue comment only if the attention item changes review/merge expectations. | Emit a bounded notification with rate limit and stable dedupe key. | Kanban comment/event with class=ATTENTION, reason, source, dedupe key, and next automated action. |
| BLOCKED_HUMAN | Direct user-visible ping in the relevant room/channel, including the exact human decision/action needed. If a dedicated profile room exists and is validated, use it; otherwise use the subscribed origin/home route. | `kanban_block(reason=...)` with a concise human-action reason. Add a comment first when context is longer than one sentence. | Create/comment only if the blocker is GitHub-native, e.g. PR review required, missing issue decision, or merge/release approval. | Watchdogs should not create BLOCKED_HUMAN unless a human action is the only resolution. | Block event, comment thread context, and any linked GitHub URL/decision handle. |
| INCIDENT | Immediate user-visible ping to the most reliable configured home route plus the responsible project room when available. Use short, high-signal text. | Create or update an incident card if one does not already exist for the dedupe key; keep affected work `waiting` unless a human action is required. | Open/comment an issue only for repo-affecting incidents or persistent reproducible bugs. | Incident watchdogs may notify on first detection, then throttle repeats while appending audit records. | Incident card/comment, source logs/handles, first-seen/last-seen timestamps, mitigation status, and closure note. |

## Dedupe and throttle policy

1. Every proactive alert must carry a stable dedupe key: `class:source:scope:subject`.
   - Examples: `INCIDENT:gateway:matrix:profile-room-isolation`, `BLOCKED_HUMAN:kanban:t_abc123:review-required`, `ATTENTION:github:pr-19:ci-wait`.
2. SILENT/INFO is never proactive, so no throttle is needed beyond ordinary logs.
3. ATTENTION sends at most once per dedupe key per 6 hours unless the summary changes materially.
4. BLOCKED_HUMAN sends when the card first blocks and again only after an unblock/reblock cycle or changed human-action request.
5. INCIDENT sends immediately on first detection, then at most once per hour per dedupe key while still active, with a required closure/mitigation audit note when resolved.
6. Kanban notification cursors remain the dedupe mechanism for task terminal events; subscriptions should survive retry-loop events until task finality (`done`/archived) so repeated real cycles still reach the user.

## Escalation policy

- SILENT/INFO never escalates.
- ATTENTION escalates to INCIDENT if it threatens data loss, routing integrity, security, project-wide dispatch, or repeated notification failure beyond the configured retry limit.
- ATTENTION escalates to BLOCKED_HUMAN only when automation can name a specific human action that is required now.
- BLOCKED_HUMAN may become INCIDENT only if the blocked condition reveals high-severity integrity/security/blast-radius risk; otherwise it stays a human blocker.
- INCIDENT may create BLOCKED_HUMAN follow-up only when mitigation requires a concrete human decision.

## Channel-specific behavior

### Matrix

- Use Matrix for project/profile-aware user-visible notices.
- Validate room isolation and gateway activation before using a dedicated profile room for direct user contact.
- Do not create new Matrix rooms as part of notification routing without an explicit encrypted/unencrypted choice.
- Include the profile identity, alert class, task/PR handle if any, and the one-line requested action or next automated action.

### Kanban

- Kanban is the durable system of record for task-scoped alerts.
- `blocked` is only for BLOCKED_HUMAN.
- Non-human waits and automated repair states are `waiting` with ATTENTION comments when user awareness is useful.
- `done` is completion history only, not a live notification/availability state.

### GitHub

- GitHub is not a general notification sink.
- Use GitHub comments/issues for repository-native review, CI, bug, release, and PR state.
- Do not duplicate Matrix/Kanban alerts into GitHub unless the GitHub artifact is where the user or reviewer must act.

### Watchdogs and cron

- Default watchdog behavior stays quiet when healthy and emits only when a threshold class is reached.
- Script-only watchdogs should produce empty stdout for SILENT/INFO, bounded prose for ATTENTION/INCIDENT, and explicit human-action text for BLOCKED_HUMAN.
- Watchdogs must include dedupe keys in their durable state or output metadata so the gateway/Kanban layer can suppress repeat noise.

### General project operation

- Routine project summaries are pull-based or scheduled digest material, not immediate pings.
- Active profile status lines stay canonical: `Self status` and `Lineage status` with working/waiting/blocked/dormant.
- Use BLOCKED only for human action blockers in status lines and Kanban states.

## Implementation implications for the next phase

- Add a small alert-routing helper or policy module rather than scattering class-specific routing in gateway, Kanban, GitHub, and watchdog code.
- Expose fields for `alert_class`, `dedupe_key`, `source`, `scope`, `subject`, `human_action`, and `audit_ref` in task comments/events or notification payloads where practical.
- Keep existing Kanban notifier behavior that advances cursors for dedupe and unsubscribes only on true finality.
- Add tests for: non-human waits not producing `blocked`, repeat BLOCKED_HUMAN after unblock/reblock, ATTENTION throttle, INCIDENT repeat throttle, and GitHub-not-a-default-sink behavior.
