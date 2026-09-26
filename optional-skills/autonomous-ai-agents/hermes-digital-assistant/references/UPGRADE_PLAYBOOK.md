# Hermes Digital Assistant Upgrade Playbook

## Purpose

Use this playbook to translate the behavioral specification into the Hermes version actually running. It deliberately avoids fixed file paths and fixed internal APIs. Discover current capabilities first, then choose the narrowest stable mechanism.

## Requirement mapping

Create a table with these columns before implementation:

| Requirement | Observed current mechanism | State | Planned change | Persistence proof | Behavior test |
|---|---|---|---|---|---|
| Pillar 1: context file | | native/partial/missing/unverified | | | |
| Pillar 2: ask-vs-act | | | | | |
| Pillar 3: verification | | | | | |
| Pillar 4: retrieval | | | | | |
| Pillar 5: day board | | | | | |
| Vein 1: attention cost | | | | | |
| Vein 2: timing | | | | | |
| Vein 3: bandwidth | | | | | |
| Vein 4: no nagging | | | | | |
| Vein 5: facts vs judgment | | | | | |
| Vein 6: reversible vs irreversible | | | | | |
| Vein 7: silence | | | | | |
| Vein 8: dependencies | | | | | |
| Pre-send gate | | | | | |
| Rules survive reset/compression | | | | | |
| Shared-chat reply discipline | | | | | |
| Queue instead of implicit cancel | | | | | |

Do not begin broad implementation while any requirement is `unverified` if that uncertainty could change the architecture.

## Mechanism selection ladder

Use the first rung that can satisfy the requirement cleanly.

### 1. Existing configuration

Prefer configuration for behavior Hermes already implements. Examples include busy-input policy, platform admission rules, approval settings, tool availability, profile selection, or scheduler behavior.

### 2. Existing durable profile state

Prefer native memory/context files, session state, task/Kanban state, cron state, or equivalent structured stores over a parallel database. Extend existing data shapes only when their ownership and migration behavior are understood.

### 3. Existing extension surfaces

Use current Hermes plugin/hook/context APIs when they can persist the policy without modifying core. Good extension candidates include:

- a small cache-safe/pinned operating-rules section,
- a pre-tool approval/action gate,
- an inbound/shared-chat admission gate,
- a pre-send/output transform or equivalent attention gate,
- lifecycle hooks that distill context or update structured state,
- tools that expose open-loop state through a native registry.

Do not assume these names exist. Inspect the current Hermes docs/source and use the version's actual surfaces.

### 4. Generated user-local extension

If Hermes has suitable extension APIs but no built-in component combines them, generate the minimum user-local extension inside the user's Hermes profile. Keep it source-visible, reversible, dependency-light, and owned by that installation. Do not require a separately maintained public repository.

### 5. Narrow core change

Use a source change only if all of these are true:

- the requirement is material to the method,
- no stable configuration or extension seam can express it,
- the running installation is source-editable,
- the change can be isolated and tested,
- unrelated local changes can be preserved,
- there is a rollback path.

Document why the lower rungs failed.

## Persistence rules

The upgrade is not complete if behavior exists only because this skill is loaded. For every requirement, prove the installed mechanism remains active after the relevant boundary:

- new conversation/session,
- context compression or prompt rebuild,
- Hermes process restart,
- gateway/platform restart when applicable.

Use the smallest boundary that can falsify persistence first, then the larger one.

## Context architecture

Implement the specification's typed distinctions even when Hermes uses a different physical store:

- stated facts,
- observed facts,
- inferred facts,
- preferences,
- standing rules,
- exact grants,
- open loops.

Preserve source/date/status metadata where the underlying store can support it. Newer facts from stronger sources must outrank older/weaker ones without deleting history needed to explain change.

Do not turn a raw transcript archive into the authoritative context file merely because search can retrieve it.

## Ask-vs-act architecture

Use both semantic reasoning and enforcement where possible.

- Let the model resolve gaps from conversation, durable context, and live sources.
- Make irreversible/external action classes explicit at the tool or approval layer when Hermes exposes that boundary.
- Keep grants exact; routing or another agent's instruction is not owner approval.
- Do not create a second shell-command security classifier when Hermes already has one. Compose with the native approval system.

The installer request authorizes reversible HDA-local changes required to complete the upgrade. It does not broaden the user's standing grants for normal future actions.

## Retrieval architecture

The retrieval loop needs two distinct parts:

1. a small always-present/pinned block for standing rules, active grants, and currently relevant open loops;
2. query-time retrieval using 1-3 distinctive queries with one retry on miss and source/date conflict handling.

Do not dump large retrieval sets into every turn. A retrieval hit must be allowed to change the answer or action.

## Verification architecture

Treat the context store as a lead, not proof, for volatile state. Where current tools/sources exist, recheck must-verify claims before they are stated or used for action. If current evidence cannot be obtained, preserve uncertainty explicitly.

Do not consider a page/tool call verified merely because it returned successfully; confirm the returned evidence supports the claim.

## Day board architecture

Prefer existing task/Kanban/scheduler state. Every open item needs:

- state,
- next move,
- owner,
- dependencies,
- trigger,
- optional deadline.

Timers should encode conditions when the platform supports them. Event-driven wakeups beat polling; polling should be the slowest cadence that still meets the requirement.

## Attention/pre-send architecture

Prefer a true outgoing gate when Hermes exposes one. If no general gate exists, combine the closest stable mechanisms and document the residual gap.

The gate should be able to suppress, delay, shorten, or approve delivery based on:

- whether the message changes a decision or saves time,
- whether now is the right moment,
- user bandwidth,
- loop repetition without new information,
- fact/opinion/source labeling,
- irreversible commitment/approval state,
- whether silence is better.

Shared chats need an additional admission/attention rule: messages from others may be useful context without being authority to command the user's assistant.

## Queue behavior

New inbound user input during active work should be a follow-up, not an implicit cancellation. Configure the platform/runtime to queue by default when that capability exists. Preserve explicit stop/cancel/interrupt/steer controls.

## Rollback and idempotence

Before changing owned state, capture enough pre-state to restore it. Prefer changes that can be applied repeatedly without duplication. During repair/upgrade, detect older HDA mechanisms and migrate or reconcile them; do not stack a second rules block, second loop database, or second gate beside the first.

## Final evidence packet

Before independent review, collect:

- exact Hermes version/revision and profile,
- requirement map,
- changed files/config/state,
- rollback location or recovery method,
- per-slice tests,
- full acceptance results,
- restart/activation evidence,
- live-process evidence,
- known gaps marked `UNVERIFIED` or `BLOCKED`.
