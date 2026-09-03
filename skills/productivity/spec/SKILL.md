---
name: spec
description: "Use for a structured interview and verified build spec."
version: 2.0.0
author: Double Rook AI
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Specification, Linear, Intake, Build-Packet, Routing, Kanban]
    related_skills: [github]
---

# /spec — Structured Spec Interview → Linear Build Packet

## Purpose

Turn a rough idea into a genuinely *buildable* spec by interviewing the operator **and** doing read-only reconnaissance, then writing it to Linear as a ready-to-route build contract.

This is the intake stage of the Double Rook build loop:

```text
idea → /spec → Linear issue with verified routing → bridge → dispatcher
     → maker builds → independent checker reviews → operator approves
```

The spec is the maker's contract and the checker's rubric. A spec that only lists requirements is a product brief; a spec with verified paths, real commands, and expected outcomes is a build contract. This skill produces the latter.

## When to Use

Use when the operator says:

- `/spec <idea>`
- `spec out <idea>`

The idea may be one line or a paragraph. Close the gap between that rough input and something an agent can build without making decisions that belong to the operator.

Do not use this skill to start implementation, create multiple issues, dispatch a worker directly, or change bridge configuration.

## Hard Rules

- **Interview + reconnoiter; do not assume.** Batch questions instead of dripping them out.
- **Question bar:** eliminate ambiguity that would force the builder to make product, risk, or architectural decisions that belong to the operator. Do not demand decisions resolvable by read-only reconnaissance or ordinary implementation judgment. Record non-blocking unknowns as assumptions/open questions instead of asking more questions. “Unambiguous” is the wrong bar—it turns intake into a negotiation over button padding.
- **Reconnaissance is read-only.** Inspect files, tests, scripts, CI, repository state, and source systems. Modify nothing, create no branch, and write no code.
- **Write one Linear issue at the end.** Do not build, dispatch, route live, or create a Kanban card. If the idea cannot fit one buildable and verifiable issue, narrow it with the operator or create one planning/investigation issue whose deliverable is the decomposition. Never spawn multiple issues.
- **Verify routing; do not copy a name.** Apply an `agent:<profile>` label only when the exact Linear label exists, the current bridge accepts that routing shape, and the exact Hermes profile exists. Never create a routing label silently and never substitute an executor.
- **Gate denylist domains.** Auth, payments/money, secrets, production deploys, migrations, security posture, legal/document generation, and trading require elevated/gated risk flags and explicit review/approval gates. Never place them on a casual fast lane.
- **Never put secret values in a spec.** Record secret names or required secret-store locations only.
- **Maker ≠ checker.** Coding packets must retain `Linear issue → branch → commit → PR → independent review → merge`; the builder cannot approve its own work.

## Procedure

### 1. Classify the idea silently

Bucket the request as one of:

- bug fix
- new feature
- refactor
- infrastructure
- investigation

Use the matching Bucket Rubric below. Do not recite the rubric to the operator.

**Done when:** the request has one primary bucket and the minimum evidence/decisions for that bucket are known.

### 2. Identify the app/repository and denylist risk

Determine the target application and repository from the operator's context, current workspace, project/repository indexes, and read-only source inspection. Identify denylist-domain contact now because it changes status and gates downstream.

**Done when:** the repository is verified rather than guessed, and risk classification is `normal`, `elevated`, or `gated` with explicit machine-readable flags.

### 3. Check duplicates and supersession in Linear

Search existing Linear issues using the intended team plus title/component terms. Include open, completed, canceled, and superseded attempts. Compare actual scope, not title similarity alone.

Choose exactly one path:

1. create a new issue and link/reference related work; or
2. ask whether the existing issue should be updated instead.

Do not manufacture duplicate backlog.

**Done when:** the draft names the duplicate-check terms and result, and no materially equivalent unresolved issue is being duplicated without operator approval.

### 4. Perform read-only repository reconnaissance

After the repository is known and before drafting:

1. Confirm repository identity, current branch/reference, and working-tree constraints.
2. Inspect likely files and existing implementation patterns.
3. Find neighboring tests, fixtures, and integration boundaries.
4. Read package scripts, test commands, lint/typecheck commands, and CI workflows.
5. Confirm every path and command the spec will reference actually exists.
6. Identify dependencies, consumers, invariants, and likely failure surfaces.

Modify nothing. Do not create a branch. Do not write code.

**Done when:** the eventual build packet can cite verified paths, real commands, and expected outcomes instead of guesses.

### 5. Ask batched clarifying questions

Ask one batch of approximately five or fewer questions covering only decisions that reconnaissance could not answer and that belong to the operator. If the original idea or recon answered a dimension, reflect it back instead of asking again.

Ask another round only if the answers expose new implementation-changing ambiguity.

**Done when:** no unresolved product, risk, or architecture decision is being delegated accidentally to the builder. Ordinary implementation choices may remain with the builder.

### 6. Reflect the proposed build packet

Show the complete draft and ask:

> Create this issue, or adjust?

Do not write to Linear until the operator gives explicit approval. Incorporate requested edits and show the corrected packet when changes are material.

**Done when:** the operator explicitly approved the exact issue packet to be written.

### 7. Verify Linear routing and bridge compatibility

Before any mutation, verify all metadata against live sources:

1. Query Linear and verify team `Build Ops` with key `BUI` exists.
2. Query the team's workflow states and resolve states by `type`, not by a guessed display name.
3. Query Linear labels and require an exact existing `agent:<profile>` label before applying it.
4. Run `hermes profile list` and require the exact profile to exist.
5. Inspect the current `kanban.linear_bridge` configuration with `hermes config` or the current runtime config source. Confirm:
   - `enabled` is `true`;
   - `dry_run` is `false` for actual dispatch (dry-run mode is not live routing);
   - `routing_label_prefix` accepts `agent:`;
   - `team_keys` includes `BUI` or is intentionally unrestricted;
   - the selected immediate-execution state type appears in `status_types` (currently `unstarted`);
   - `allowed_profiles` is empty/unrestricted or contains the exact requested profile;
   - `issue_id_allowlist` is empty for a newly created issue; and
   - `max_creates_per_tick` is a positive integer.
6. If any routing check fails, apply no agent label and write this in the issue body:

```text
ROUTING REQUESTED: <agent> — label/profile/bridge mapping unverified; operator or ops owner must resolve before dispatch.
```

Never create a missing label, infer a near-match, or swap in a different profile.

**Done when:** the proposed issue is either proven bridge-consumable or explicitly and intentionally gated outside the bridge.

### 8. Write one Linear issue

Use this body structure.

#### Goal

What and why, in one or two sentences.

#### Acceptance criteria

Concrete, checkable done conditions. Avoid subjective completion language.

#### Scope

Verified repository/application, in-scope files or areas, explicit non-goals.

#### Constraints & patterns

Existing patterns found during recon, invariants, protected boundaries, and behavior that must not break. For code work include:

```text
Linear issue required before code. Branch → commit → PR → independent review → merge. No exceptions.
```

#### Risk

Use this machine-readable shape:

```yaml
risk:
  tier: normal | elevated | gated
  flags:
    auth: false
    secrets: false
    payments: false
    migration: false
    production: false
    legal: false
    trading: false
    security: false
    external-communication: false
  required_gates:
    - <explicit gate or "standard maker-checker PR gate">
```

Any denylist flag set to `true` normally requires a non-`unstarted` initial state plus named specialist/operator approval before bridge eligibility.

#### Verification

Include expected outcomes, not commands alone:

- **Builder verification:** focused tests/checks and the required result.
- **Independent checker verification:** exact-head behavior, failure modes, and regression criteria.
- **Runtime/live verification:** when applicable.
- **Rollback verification:** required for infrastructure or migration work.

#### Routing

Use exactly one of:

```text
ROUTING VERIFIED: agent:<profile> — Linear label exists; Hermes profile exists; live bridge is enabled and non-dry-run; prefix/team/state/profile-allowlist/issue-allowlist/capacity gates verified.
```

```text
ROUTING REQUESTED: <agent> — label/profile/bridge mapping unverified; operator or ops owner must resolve before dispatch.
```

```text
ROUTING UNSET — operator to label.
```

#### Assumptions / open questions

Record only non-blocking unknowns.

### 9. Apply Linear metadata

- **Team:** Build Ops / `BUI` unless the operator explicitly specifies another verified team.
- **Project:** only when identified and verified; never invent one.
- **Initial state:** backlog/triage by default. Use an `unstarted` state only when the operator explicitly wants immediate bridge eligibility and every live bridge gate in Step 7 passes.
- **Priority:** unset unless the operator specified it or urgency was established.
- **Labels:** verified before applying.
- **Assignee:** none unless explicitly requested and identity mapping is verified.

Create exactly one issue and stop after a successful `issueCreate`. Do not mutate its status again as a shortcut to dispatch.

### 10. Report and stop

Report:

- created issue URL;
- team and state, including the state's type;
- applied labels;
- routing condition;
- whether it is immediately bridge-eligible or deliberately gated.

Do not dispatch, build, create a Kanban card, or change bridge configuration.

## Bridge Contract

For an issue to flow through the live Linear → Kanban bridge, all of these must be true:

| Contract field | Required immediate-flow value |
|---|---|
| Linear team | `BUI` |
| Routing label | one exact, existing `agent:<profile>` label |
| Hermes executor | exact profile exists and bridge can resolve it |
| Bridge mode | `enabled: true` and `dry_run: false` |
| Profile allowlist | empty/unrestricted, or contains the exact requested profile |
| Workflow state | state `type` accepted by `kanban.linear_bridge.status_types`—currently `unstarted` |
| Issue allowlist | empty for a newly created issue; otherwise `/spec` must not claim immediate flow |
| Bridge capacity | `max_creates_per_tick` is a positive integer |
| Risk gate | no unresolved gate requiring operator/specialist approval |

A packet that should wait for operator approval stays in a non-accepted state such as `backlog` or `triage` deliberately. Report that it is gated; never describe it as queued for the bridge.

## Dry-Run Mode

When the operator requests a dry run or uses `/spec` to test the workflow:

1. Perform the interview, duplicate check, repository reconnaissance, risk classification, and routing/bridge verification read-only.
2. Render the complete proposed issue body and metadata.
3. Mark the output `DRY RUN — NO LINEAR MUTATION`.
4. Do not call `issueCreate`, `issueUpdate`, Kanban mutation tools, Git mutation commands, or build/deploy commands.

A dry run proves the packet's shape and routing contract; it never creates or dispatches work.

## Bucket Rubrics

Consult these; do not recite them.

### Bug fix

- reproduction steps;
- expected versus actual behavior;
- environment/version;
- evidence such as logs, screenshots, request IDs, or failing tests;
- regression-test requirement.

### New feature

- operator/user workflow;
- entry point and output;
- success, empty, and failure states;
- permissions/roles;
- explicit non-goals.

### Refactor

- invariant behavior that must not change;
- reason for the refactor;
- baseline tests/metrics;
- prohibited behavior changes.

### Infrastructure

- target environment;
- current versus desired state;
- rollback plan;
- observability/health verification;
- blast radius and downtime constraints.

### Investigation

- question to answer;
- evidence sources;
- deliverable format;
- decision it should enable;
- clear stopping condition.

## Anti-Patterns

- Skipping recon and writing a spec the operator must make buildable.
- Treating “unambiguous” literally and interrogating the operator over trivia.
- Copying a routing name without verifying Linear label, Hermes profile, and bridge contract.
- Manufacturing duplicate issues instead of checking first.
- Splitting an epic into multiple issues instead of creating one planning issue.
- Selecting `backlog` while claiming the issue will flow through an `unstarted`-only bridge.
- Selecting `unstarted` for gated auth, money, legal, migration, production, security, or trading work.
- Building “to save a step.” `/spec` ends at one written, verified Linear issue.

## Why This Shape

Read-only reconnaissance between interview and draft is the difference between a spec that is requirements-complete and one that is genuinely buildable. Verified paths, real commands with expected outcomes, machine-readable risk flags, and verified routing turn the packet into an objective contract. An independent checker can then render a real `APPROVE` or `CHANGES REQUIRED` verdict, and downstream gates are operationally enforceable rather than advisory.

## Verification Checklist

- [ ] Idea classified and denylist risk identified.
- [ ] Target repository/application verified.
- [ ] Duplicate and supersession search completed.
- [ ] Referenced paths and commands verified read-only.
- [ ] Batched questions resolved operator-level decisions only.
- [ ] Operator approved the exact packet before mutation.
- [ ] Team, state type, project, label, profile, and bridge contract verified live.
- [ ] Risk flags and required gates are machine-readable.
- [ ] Exactly one Linear issue created—or zero in dry-run mode.
- [ ] Report states whether the issue is bridge-eligible or intentionally gated.
- [ ] No build, dispatch, Kanban creation, or bridge/config mutation occurred.
