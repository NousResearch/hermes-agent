---
name: sprocket-product-management
description: Operate Sprocket's issue-centered product management system: capture bugs, features, tasks, cleanup, and research as GitHub issues; use each issue as the durable hub for discussion and feedback; request reviewer input; keep status current; and progressively refine the work.
version: 0.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [GitHub, Issues, Product-Management, Research, Triage, Workflow, Sprocket]
    related_skills: [github-issues, github-pr-workflow, requesting-code-review]
---

# Sprocket Product Management

This skill defines how agents should manage product work for **Sprocket**.

The core rule is simple:

> **Every meaningful unit of product work gets a GitHub issue, and that issue becomes the durable hub for the topic.**

That includes bugs, feature requests, tasks, cleanup, research spikes, and larger umbrella efforts.

Use this skill when an agent is asked to triage work, capture new requests, organize discussion, ask for feedback, refine problem statements, or keep a topic moving forward over multiple interactions.

For the mechanics of creating/editing/commenting on issues, use the `github-issues` skill alongside this one. This skill is about **operating discipline**, not just CLI syntax.

## Goals

1. **Capture all incoming work durably** so nothing important lives only in chat.
2. **Keep one canonical hub per topic** where people can find status, context, decisions, and next steps.
3. **Make progress legible** by keeping labels, status, and summaries up to date.
4. **Request the right human input at the right time** instead of waiting silently.
5. **Refine understanding over time** so vague requests become researched, actionable work.

---

## Core Operating Principles

### 1. Issues are the source of truth

Discord, terminal sessions, and ad hoc notes are transient. If a topic matters, record it in GitHub.

Agents should avoid letting important conclusions live only in:
- chat threads
- temporary scratchpads
- unlinked PR descriptions
- private reasoning

If useful discussion happens elsewhere, summarize it back into the issue.

### 2. One topic, one hub

Prefer a single canonical issue per discrete topic.

Before creating a new issue:
1. Search for an existing issue covering the same problem or request.
2. Reuse and update the existing issue if it is still the right home.
3. Only create a new issue if the topic is meaningfully distinct.

If you discover overlap later:
- choose one canonical issue
- link the duplicates
- close or deprecate the redundant one with a short explanation

### 3. Classification matters

Every issue should clearly declare what kind of work it represents.

Use these primary work types:
- `type:bug` — something is broken or behaving incorrectly
- `type:feature` — a user-facing or system capability request
- `type:task` — concrete implementation or operational work
- `type:cleanup` — simplification, refactor, debt reduction, hygiene
- `type:research` — investigation, discovery, validation, scoping
- `type:epic` — umbrella issue coordinating several related child issues

If the repo uses different exact label names, map to the closest equivalent but keep this conceptual taxonomy.

### 4. The issue body is the current canonical summary

Comments are the event log.

The **issue body** should be kept updated so a reviewer can open the issue and quickly understand:
- what the topic is
- why it matters
- current status
- latest understanding
- open questions
- next actions
- success criteria

Do not force humans to reconstruct the current state from 40 comments.

### 5. Ask for review explicitly, not passively

If human input is needed, request it clearly.

Do not merely leave a pile of research and hope someone notices. Instead:
- identify the reviewer or stakeholder if known
- state exactly what input is needed
- ask specific questions
- summarize the current recommendation
- note what will happen after feedback arrives

### 6. Refine continuously

A vague request is not a failure state. It is the beginning of a refinement loop.

Agents should help move work through stages such as:
- raw intake
- clarified problem statement
- scoped proposal
- researched recommendation
- implementation-ready issue
- completed / closed

The issue should evolve as understanding improves.

---

## Standard Workflow

## 1. Intake

When new product input arrives, first decide whether it deserves an issue.

Create or update an issue when the topic has any of the following:
- expected follow-up work
- a decision to make
- research to do
- user impact
- code or process changes likely to occur
- a need for durable tracking across sessions

Examples of inputs that should usually become issues:
- bug reports
- feature ideas
- complaints about confusing UX
- requests from users or teammates
- suspected technical debt
- flaky tests or broken harnesses
- unclear architectural questions worth investigating
- cleanup opportunities that would improve future agent velocity

### Intake checklist

- Is there already an issue for this?
- What kind of work is it?
- Who is affected?
- What triggered the request?
- Is this actionable now, or does it need research first?
- What would “done” look like?

## 2. Triage and classify

Once an issue exists, immediately make it legible.

At minimum, the issue should have:
- a clear title
- a primary type label
- an initial status label if available
- a concise body with context and next step

Recommended status flow:
- `status:triage` — newly captured, needs sorting
- `status:research` — gathering evidence / clarifying problem
- `status:ready` — sufficiently defined and ready for execution
- `status:blocked` — waiting on external input or dependency
- `status:in-progress` — active implementation or active investigation
- `status:review` — waiting on reviewer or stakeholder decision
- `status:done` — completed and ready to close / closed

Use the repo’s actual labels if they differ, but preserve this intent.

## 3. Build the hub

Every canonical issue should read like a compact living brief.

Use this structure for the issue body whenever possible:

```markdown
## Summary
One-paragraph description of the topic and why it matters.

## Type
Bug / Feature / Task / Cleanup / Research / Epic

## Status
Current stage and what is actively happening now.

## Problem / Opportunity
What is broken, missing, risky, or promising?

## Current Understanding
What we know so far. Include evidence, observations, and constraints.

## Open Questions
- Question 1
- Question 2

## Proposed Direction
Current recommendation or leading hypothesis.

## Next Actions
- [ ] Action 1
- [ ] Action 2

## Success Criteria
How we will know this topic is successfully resolved.

## Links
Related issues, PRs, docs, reports, incidents, or chats.
```

Not every issue needs every section in the same depth, but every issue should make the current state obvious.

## 4. Use comments as an event log

Post comments for meaningful updates such as:
- new findings
- decisions made
- blockers discovered
- reviewer requests
- implementation progress
- changed recommendations
- closure summaries

Good update comments are:
- timestamped by GitHub automatically
- short but informative
- written for someone who has not followed every chat message
- linked to evidence when available

After a significant comment thread or research pass, update the issue body to reflect the newest canonical summary.

## 5. Request reviewer input intentionally

When the issue needs feedback, leave a structured review request comment.

A good review request includes:
- who should weigh in
- why their input is needed
- the shortest useful summary of the topic
- the current recommendation
- specific questions to answer
- the decision or action unlocked by their response

Example:

```markdown
Requesting product review.

Current recommendation: ship Option B first because it addresses the user pain with lower implementation risk.

Questions:
1. Do we agree the first milestone should optimize for reliability rather than breadth?
2. Is the proposed scope small enough for this release?
3. Are there user segments or constraints we are missing?

If approved, next step is to split this into implementation issues and mark it ready.
```

If a reviewer is not known, say what kind of reviewer is needed, e.g.:
- product owner
- infra owner
- API reviewer
- design reviewer
- release owner

## 6. Refine research into action

Research issues should not stay as vague exploration forever.

A research thread is healthy when it converges toward one of these outcomes:
- close as answered / not needed
- convert into an implementation-ready issue
- split into several follow-up issues
- escalate into an epic if scope is larger than expected

When refining research:
1. collect evidence
2. summarize findings
3. compare options
4. recommend a direction
5. identify unresolved risks
6. define next concrete actions

If the issue started vague, edit the title and body once the real problem becomes clearer.

## 7. Keep status fresh

Whenever the state changes, update the issue.

Common triggers:
- new reproduction confirmed
- root cause found
- scope narrowed or expanded
- waiting on human input
- implementation started
- PR opened
- PR merged
- rollout or validation complete
- topic abandoned or folded into another issue

The minimum acceptable behavior is:
- update labels/status
- leave a brief comment for the transition
- revise the issue body if the canonical summary changed

## 8. Close with a useful resolution

When an issue is done, do not just close it silently.

Leave a short closing summary describing:
- what happened
- what changed
- what evidence suggests it is resolved
- any follow-up work that remains

If closing as not planned / duplicate / obsolete, say why and link the replacement issue if applicable.

---

## Issue Type Guidance

## Bugs

Use for broken behavior, regressions, reliability failures, or incorrect system outcomes.

A good bug issue should capture:
- expected behavior
- actual behavior
- reproduction steps if known
- severity / user impact
- suspected scope or component
- evidence (logs, screenshots, harness output, traces)

Prefer converting vague complaints into concrete failure reports.

## Features

Use for product or platform capabilities that do not exist yet.

A good feature issue should capture:
- user need or operator need
- why current behavior is insufficient
- proposed experience or behavior
- constraints and trade-offs
- success criteria

Avoid writing feature requests as solution-only tickets with no problem statement.

## Tasks

Use for bounded work items that are already understood well enough to execute.

Good task issues are:
- specific
- implementation-oriented
- clearly scoped
- linked back to parent feature / bug / research when relevant

## Cleanup

Use for debt reduction, simplification, dead code removal, flaky harness fixes, repo hygiene, or workflow improvements.

A cleanup issue should explain:
- what friction exists today
- why it matters
- what simplification or repair is proposed
- expected payoff (reliability, speed, clarity, lower maintenance)

## Research

Use for questions that still need discovery.

Research issues should answer:
- what we are trying to learn
- why that knowledge matters
- what evidence we need
- what decision the research should unlock

Research is successful when it reduces uncertainty enough to choose an action.

## Epics

Use epics sparingly, for multi-issue efforts with a shared product goal.

An epic should contain:
- the outcome being pursued
- why this matters now
- major subtracks
- child issue list
- current overall status
- major risks / dependencies

Epics should not replace the child issues; they coordinate them.

---

## Recommended Agent Behaviors

### When creating a new issue

1. Search for duplicates first.
2. Choose the right issue type.
3. Write a title that names the real topic, not a vague symptom.
4. Fill in the initial body with the current known facts.
5. Add labels for type and status.
6. Link related issues or PRs.
7. If immediate feedback is needed, add a review request comment.

### When updating an existing issue

1. Read the current issue body and recent comments.
2. Post a concise update comment with new findings.
3. Edit the issue body to reflect the current state.
4. Adjust labels / status.
5. Add or revise next actions.

### When splitting scope

Split an issue when:
- multiple distinct workstreams have emerged
- one issue now mixes research and implementation for separate topics
- execution requires several independently trackable tasks
- a single issue has become too large to reason about clearly

When splitting:
- keep one parent hub if useful
- create child issues with clear scopes
- link them both ways
- update the parent with a checklist of children

### When asking for input

Ask only after doing enough work to make the question sharp.

Bad:
- “Any thoughts?”
- “Please review.”

Good:
- “We found two viable approaches; Option B is lower risk because it avoids schema churn. Do you agree we should choose B for the first pass?”

### When context lives in chat

If you learn something important in Discord or elsewhere:
1. summarize it into the issue
2. quote or paraphrase the key takeaway
3. link back if a durable chat link exists
4. update the canonical summary

---

## Anti-Patterns to Avoid

Do **not**:
- create duplicate issues without checking existing work
- leave issues with only a title and no useful body
- let status drift for days while the issue body becomes stale
- bury the current recommendation in a long comment chain
- ask reviewers to read raw logs without summarizing them
- keep vague research open without defining what decision it should unlock
- close issues with no explanation
- use ephemeral chat as the only record of a product decision

---

## Practical Decision Rules

### Create a new issue when
- the topic is meaningfully new
- the work needs durable tracking
- the existing issue would become muddled by combining them

### Update an existing issue when
- new evidence changes understanding of the same topic
- the next step changed but the topic did not
- discussion or review happened on the same underlying problem

### Create an epic when
- several child issues are needed to reach one outcome
- coordination matters more than any single task
- stakeholders need a single high-level progress hub

### Close an issue when
- the work is complete
- the question has been answered
- the request is explicitly not planned
- the topic has been superseded by another canonical issue

---

## Suggested Templates and References

This skill includes linked templates for:
- bug issues
- feature issues
- task issues
- cleanup issues
- research issues
- reviewer update comments
- closure summaries

Use them as starting points, then tailor them to the actual topic.

---

## Sprocket-Specific Mindset

For Sprocket, agents should optimize for:
- durable coordination across many sessions
- easy handoff between agents and humans
- visible product reasoning, not just visible code changes
- making uncertain work progressively more concrete
- keeping GitHub issues as the backbone of planning, discussion, and execution

If you are unsure whether something belongs in GitHub, err toward creating or updating an issue with a concise summary. It is better to have a durable hub early than to reconstruct intent later.
