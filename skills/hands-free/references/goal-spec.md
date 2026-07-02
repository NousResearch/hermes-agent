# The `/goal [...]` Command Specification

The `/goal` block is the contract handed to the **Lead orchestration agent**. It
must be self-contained: a fresh orchestrator with no prior conversation context
executes it verbatim. Render it as a single fenced code block.

## Anatomy

```
/goal [
  OUTCOME: <one sentence: the end state the user wants to exist>

  SUCCESS CRITERIA:
    - <observable, testable condition 1>
    - <observable, testable condition 2>
    - ...

  CONSTRAINTS:
    - <hard rules, brand/voice, budget, deadline, tech, do-nots>

  ASSUMPTIONS:
    - <every default you chose on the user's behalf>

  ORCHESTRATION:
    LEAD: <the orchestrator's standing instructions — see "Lead mandate" below>

    MILESTONE 1 — <name>  [depends on: none]
      goal: <what this milestone produces>
      done when: <definition of done for this milestone>
      parallel subtasks:
        - [@<agent-skill>] <subtask> → produces <artifact> | done when <check>
        - [@<agent-skill>] <subtask> → produces <artifact> | done when <check>
      gate: <none | review by @reviewer | HUMAN APPROVAL before proceeding>

    MILESTONE 2 — <name>  [depends on: Milestone 1]
      goal: ...
      done when: ...
      parallel subtasks:
        - [@<agent-skill>] ...
      gate: ...

    MILESTONE N — <name>  [depends on: ...]
      ...

  DELIVERABLES:
    - <final artifact(s) and where they should land>

  ESCALATION:
    - If blocked > <threshold>, or a gate is rejected twice, pause and report to
      the user with the specific blocker and options.
]
```

## Field rules

- **OUTCOME** — exactly one sentence, outcome-shaped ("a published, reviewed
  landing page for the Q3 launch"), never task-shaped ("write some copy").
- **SUCCESS CRITERIA** — 2–6 bullets, each independently verifiable. These are
  what the Lead checks before declaring the goal complete.
- **CONSTRAINTS** — carry through every hard rule the user or repo imposes
  (brand voice, "never post without approval", budget, deadline, stack).
- **ASSUMPTIONS** — list every default chosen for the user. Empty only if the
  user specified everything.
- **MILESTONES** — the **linear spine**. Ordered, each with an explicit
  `depends on`. A milestone may only start once its dependencies are `done`.
- **parallel subtasks** — the **ribs**. Independent units *within* one milestone
  that the Lead dispatches concurrently to skilled sub-agents. Each line:
  `[@skill] <imperative subtask> → produces <artifact> | done when <check>`.
  Never place two subtasks in the same milestone if one depends on the other —
  split them across milestones instead.
- **gate** — `none`, a `review by @reviewer`, or `HUMAN APPROVAL`. Mandatory
  before any outward-facing action (publish/send/deploy).

## Lead mandate (standing instructions for the orchestrator)

Embed these as the `LEAD:` value:

> Execute milestones strictly in dependency order. For each milestone, dispatch
> all parallel subtasks at once to the named skilled sub-agents, then collect and
> integrate their artifacts. Verify the milestone's "done when" before advancing.
> Honor every gate — never cross a HUMAN APPROVAL gate without explicit sign-off.
> On subtask failure, retry once with feedback, then escalate. Maintain a running
> status of milestone/subtask state. Declare the goal complete only when all
> SUCCESS CRITERIA are met; then surface the DELIVERABLES.

## Quality bar before you emit

- [ ] OUTCOME is a single outcome-shaped sentence.
- [ ] Every milestone has `depends on`, `done when`, and a gate decision.
- [ ] Every subtask names a roster skill, an artifact, and a done-check.
- [ ] No parallel subtask depends on a sibling in the same milestone.
- [ ] Every outward-facing action sits behind an explicit gate.
- [ ] Assumptions are listed, not buried.
- [ ] The block reads correctly with zero external context.
