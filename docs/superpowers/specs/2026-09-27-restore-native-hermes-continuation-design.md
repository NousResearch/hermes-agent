# Restore Native Hermes Continuation — Design

**Lane:** POL-170
**Status:** Approved direction 1
**Outcome:** Restore the useful autonomous behavior Hermes had before the software factory, without restoring ARES or any replacement orchestration framework.

## User outcome

A clear request such as “build,” “fix,” or “change” starts one durable engineering job. Hermes keeps that same job moving through implementation, verification, merge, deployment, and live acceptance without Kevin babysitting it. Ordinary questions remain ordinary chat.

## Architecture

Use Hermes's existing gateway, conversation, Kanban, process-watcher, checkpoint, and worker capabilities. Do not create a new service, scheduler, dispatcher, database, project, or control plane.

The request-facing Hermes agent remains the accountable owner. It admits at most one durable card for the requested outcome and binds that card to the originating conversation and existing Linear outcome when one exists. The native gateway dispatcher may start the card's worker. No decomposer, execution governor, proactive supervisor, review dispatcher, or factory intake process participates.

## Work flow

1. A Hermes agent recognizes an unambiguous build, fix, or change request.
2. It reconciles existing Linear work, branches, pull requests, active sessions, and Kanban cards before creating anything.
3. If no matching active job exists, it creates exactly one durable card and records the originating session.
4. One worker owns the outcome and continues within the same job. Context compression and a gateway restart keep that worker alive. A host loss preserves the same card, worktree, and checkpoint but does not automatically start another model; an explicit unblock resumes the same card rather than creating a replacement.
5. The worker develops and tests locally first on one branch.
6. It requests one independent read-only review of the final exact head.
7. It runs hosted CI only on that final reviewed head, using at most one pull request.
8. It merges, deploys or restarts, and exercises the live user path automatically unless a protected human gate applies.
9. It marks the durable job complete only after live acceptance succeeds.

## Admission rules

- Automatically admit clear implementation requests: build, fix, change, add, remove, update, ship, deploy, or finish.
- Do not admit questions, research-only requests, explanations, brainstorming, status checks, or requests explicitly marked hold, local-only, no commit, no push, no merge, or no deploy.
- Reuse a matching active job instead of creating a duplicate.
- A changed request updates the existing outcome only when it is plainly the same desired result; otherwise Hermes asks one short question.
- Keep the existing POL-170 lane for this restoration. Do not create a replacement project.

## Continuation and limits

- Native dispatch is enabled only for admitted durable cards.
- Allow one active worker per profile and one owner per outcome.
- Do not enable automatic decomposition, review dispatch, proactive supervision, global backlog polling, or dispatcher-level automatic retries.
- A worker may correct failures during its own active run. When the worker exits unsuccessfully, Hermes checkpoints the exact state and stops; it does not launch a fresh model automatically.
- Native active-run ceilings are 20 model calls and eight hours for the request-facing software profiles. Direct-delivery cards also carry an eight-hour process cap and `max_retries=1`.
- The deterministic external limiter enforces hard processed-token, attempt, and process-time ceilings. It can checkpoint and stop the existing job, but cannot call a model, create work, or retry.
- Protected human gates remain limited to product or business judgment, spend, secrets, credentials, destructive data risk, force-push or history rewrite, live financial risk, irreversible infrastructure removal, legal commitments, or unavailable OS permissions.

## Notifications

- Send Kevin a message only when an outcome is complete, a real human gate is reached, or a failure has stopped work and cannot be corrected within the active run.
- Do not send per-step, reviewer, retry, polling, or CI-failure chatter.
- GitHub receives one final-head CI run per outcome. Intermediate local failures do not create remote runs or email noise.

## Configuration boundary

The intended live posture is:

- native gateway dispatch enabled;
- review dispatch disabled;
- automatic decomposition disabled;
- proactive supervisor disabled;
- one active worker per profile;
- first failed worker run checkpoints and stops rather than redispatching;
- ARES and all software-factory services remain absent.

Any implementation must use existing Hermes extension points and configuration. A user-specific skill or prompt contract is preferred over widening Hermes core tools.

**Implementation note:** On `default`, `acqlens-agent`, and `surveyor-agent`, set `kanban.dispatch_in_gateway`, `kanban.notify_in_gateway`, and `kanban.auto_subscribe_on_create` to `true`; set `kanban.review_dispatch` and `kanban.auto_decompose` to `false`; set `kanban.failure_limit`, `kanban.max_in_progress`, `kanban.max_in_progress_per_profile`, and `agent.api_max_retries` to `1`, `agent.max_turns` to `20`, `agent.run_budget_seconds` to `28800`, and `agent.auto_recovery_cycles` to `0`. After reconciling existing work, direct admission uses one `kanban_create` call with `title`, `assignee`, `body`, `one_per_request=True`, `max_retries=1`, `max_runtime_seconds=28800`, and `completion_contract="OWNER/REPO"`. The one-request key applies to admission, not decomposition; the first failed attempt blocks the card, with no model poller or retry loop.

## Verification

Verification must prove behavior, not only configuration:

1. A normal question creates no durable card.
2. A clear small change creates exactly one card tied to its originating session.
3. Repeated delivery of the same request does not create a second card.
4. A gateway restart leaves the same card and worker alive. Host loss preserves the same card/checkpoint and requires explicit unblock rather than automatic model respawn.
5. A failed worker checkpoints and stops without automatic redispatch.
6. The final-head review and CI occur once.
7. The change merges, deploys, and passes a live acceptance check.
8. No ARES, software-factory, governor, decomposer, review-dispatch, or proactive-supervisor process is running.
9. Notification output contains only completion, protected-gate, or terminal-stop messages.

## Non-goals

- Rebuilding ARES or the software factory.
- Importing the entire Linear backlog into Hermes.
- Polling for new work.
- Creating parallel implementation or review lanes.
- Automatically retrying failed model workers.
- Adding a Mission Control engineering scheduler.
- Replacing Hermes's existing Kanban, session, checkpoint, or process-watcher infrastructure.
