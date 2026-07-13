# Muncho production-shaped capability canary

This is the gate after the isolated clean-room full canary. It is not a direct
promotion of the clean-room configuration: that configuration intentionally
disables memory, cron, skills, ordinary tools, and the Discord adapter so the
core model/writer/egress invariants can be proved without unrelated state.

The production-shaped canary proves that useful everyday capability can be
restored at the edges without moving semantic authority out of GPT/Hermes.
Production remains unchanged until both canaries pass and the owner approves
the exact promotion plan.

## Authority split

GPT/Hermes alone owns:

- interpretation of the user's request;
- decomposition, prioritisation, and plan revision;
- choosing which available tool or skill to use;
- deciding whether live DB, Bitrix, browser, file, or channel evidence is
  needed;
- choosing a model-authored `high` to `xhigh` effort escalation;
- deciding that an ambiguity must be explained to the user;
- deciding the content of a handoff and its explicitly selected recipient.

Deterministic runtime code may only:

- expose a statically configured capability;
- validate schemas, exact identities, paths, command hashes, TTLs, and
  permissions;
- execute the exact model-authored action;
- enforce safety, idempotency, and Discord-DM denial;
- persist receipts and reconcile external state after a lost response.

It must not classify task text, infer intent from keywords, select a worker or
channel from prose, decompose work, rewrite a model plan, or turn historical
approval evidence into current mutation authority.

## Target capability surface

The target is a fixed, reviewed Discord toolset assembled from existing Hermes
edges. No new permanent core model tool is required merely to expose access
that terminal, file, browser, an existing skill, or an existing plugin already
provides.

Required first-wave capabilities:

1. `canonical_brain` and `todo` for durable model-authored plans,
   verifications, handoffs, and exact-plan approval receipts.
2. `clarify` so unresolved people, channels, and intentions are explained and
   learned after the owner's answer.
3. `file` and `terminal` for the reviewed Cloud workspaces. Read-only work is
   not approval-gated merely because it is operational.
4. `browser` and `web` for public or separately authenticated Cloud browser
   work.
5. `skills` and `session_search` for existing operational procedures and
   conversation evidence.
6. `memory` as supplemental context only. Canonical Brain remains the source
   of truth for cases, plans, handoffs, approvals, and route-back outcomes.
7. `delegation` only when GPT explicitly authors the delegation. No auxiliary
   decomposer or dispatcher may manufacture or assign subtasks.
8. Public Discord channel/thread delivery through the privileged egress
   boundary. The gateway never receives the Discord token and a DM target is
   rejected before dispatch.

The first production-shaped canary excludes `kanban` from the model toolset and
requires all of the following even if upstream defaults change:

```yaml
kanban:
  auxiliary_planning_enabled: false
  auto_decompose: false
  dispatch_in_gateway: false
```

`code_execution`, `computer_use`, image generation, TTS, vision, and cron are
added one gate at a time after their actual runtime access and approval
contracts are proved. Their absence from the first wave is not a semantic
restriction on GPT; it prevents an unverified edge from being confused with a
working capability.

## Files and terminal

The canary runs as a distinct identity against a disposable copy or an exact
read-only bind of representative workspaces. It proves:

- read, search, git inspection, and test execution without owner prompts;
- writes only inside an explicitly writable canary root;
- no read access to Discord, database, browser-session, or provider secrets;
- no environment-variable passthrough beyond an exact allowlist;
- exact-plan command capabilities bypass repeated prompts only for the
  approved command hashes, session epoch, owner identity, TTL, and use count;
- hardline safety blocks remain non-bypassable;
- a failed approach is reported to GPT as tool evidence so GPT can choose a
  different safe approach instead of the runtime declaring the task complete.

Access to files on the owner's personal computer is a separate local-edge
gate. Cloud filesystem authority does not imply Mac filesystem authority. A
future local bridge must expose explicit roots and operations, keep secrets
out of Cloud, and execute only an exact GPT-authored handoff or owner-approved
mutation.

## Database and business systems

Database and business-system access stays at the edge:

- reads use existing reviewed scripts, APIs, SSH paths, or skills and return
  bounded evidence to GPT;
- writes require the applicable exact owner/role approval and an idempotency
  key;
- an ambiguous response is reconciled by reading live state before retry;
- success is recorded only after external readback or an authoritative
  receipt;
- Canonical Brain receives the intent, approval, outcome, and evidence refs,
  not raw credentials or customer secrets.

Bitrix currently depends on a separately authenticated browser session. The
Cloud browser must not pretend that it owns that session. GPT may explicitly
author a handoff to the pinned local browser worker; deterministic delivery
may send that exact handoff to the selected worker, but may not select the
worker from keywords. The local worker returns evidence and the originating
GPT/Hermes instance decides the next step. Bitrix mutations remain separately
approved and receipt-backed.

## Cron and unattended work

Previously blocked or unpinned jobs are not re-enabled by copying old state.
Each job must have:

- an explicit owner-authored schedule and objective;
- a pinned provider/model contract compatible with `gpt-5.6-sol` where model
  work is required;
- an explicit toolset and identity;
- a non-interactive mutation policy;
- an idempotency/reconciliation contract;
- a Canonical Brain outcome receipt;
- a bounded failure path that reports the blocker instead of silently
  succeeding or retrying forever.

No cron job may invoke the retired Kanban semantic decomposer. Unattended
dangerous mutations remain denied unless a separate standing approval contract
is explicitly reviewed and installed.

## Canary scenarios

The production-shaped canary must complete all scenarios through the normal
gateway/model loop, not by calling the implementation functions directly:

1. Inspect two workspaces, compare live git/runtime facts, and produce a
   verified answer without an approval prompt.
2. Diagnose a deliberately missing fact, obtain it through a second available
   read path, and continue rather than stopping at the first blocker.
3. Execute a sustained multi-step task with a model-authored Canonical Task
   Workspace; after a controlled restart, resume from the next unverified step
   without replaying completed mutations.
4. Give GPT a genuinely difficult multi-step objective without mentioning an
   effort level, reasoning control, or a required `todo` call. Prove GPT itself
   requests `xhigh` through the existing `todo.reasoning` field and that the
   later same-turn Codex request uses it. No task-text classifier participates.
5. Present one exact mutation plan, consume one durable command capability for
   all approved steps, and finish without repeated micro-approvals.
6. Prove an unapproved command, expired capability, changed command byte, wrong
   owner, wrong session epoch, and stale plan revision are each denied.
7. Perform a DB read, then a transactionally safe canary-only write with live
   readback and idempotent lost-response reconciliation.
8. Perform a Bitrix read through the explicitly selected authenticated edge;
   keep a mutation blocked until its separate approval is present.
9. Send to an allowlisted public Discord thread, verify the platform receipt
   and public readback, then append `route_back.sent`.
10. Attempt a Discord DM and prove denial occurs before platform dispatch, with
    `route_back.blocked` recorded instead.
11. Simulate tool, browser, DB, writer, and egress failures. GPT must retain
    semantic control, try other safe evidence paths where available, and leave
    the durable plan either completed or honestly blocked.

## Promotion evidence

Promotion requires one digest-bound evidence bundle containing:

- exact fork/release SHA and installed-wheel manifest;
- exact effective model/provider/effort policy;
- exact static tool and plugin inventory;
- proof that Kanban auxiliary planning and dispatch are disabled;
- prompt-cache stability and message-alternation checks;
- Canonical Writer readiness and least-privilege PostgreSQL attestation;
- every scenario transcript, tool receipt, and Canonical Brain event ID;
- public Discord delivery/readback and DM pre-dispatch denial;
- service stop/cleanup receipts and absence of canary credentials;
- a read-only production diff of code, config, identities, permissions, jobs,
  and data migrations.

Only after that bundle is reviewed is a fresh owner approval requested for the
exact production mutation plan. Clean-room canary success, this document, a PR
approval, or an older conversational approval is not production authority.
