# Weekly Build Ideas Digest Pattern

Use this for scheduled collection of project ideas, feature suggestions, and build opportunities mentioned in the last 7 days.

## Goal
Produce a structured digest only when there is genuine new or active build signal. Return exactly `[SILENT]` when no new ideas, feature suggestions, or improvements were found.

## Source sweep
Scan at least three source classes before deciding silence:

1. **Conversation/session history**
   - Search recent sessions for broad idea terms: `idea`, `build`, `feature`, `improvement`, `suggestion`, `pilot`, `plugin`, `product`, `repo-suggestion`.
   - Also search concrete domains surfaced in recent work, e.g. `webhook`, `GitRadar`, `prompt optimizer`, `TutorBot`, `CLI themes`, `rate limiter`, `filetree`.

2. **GitHub activity**
   - Search issues/PRs updated in the last 7 days authored by Sahil/KENSEI.
   - Include ecosystem suggestion issues, upstream Hermes PRs, and local repo commits where they imply new reusable capabilities.

3. **Kanban state**
   - Check whether each idea already maps to a kanban task.
   - Match by robust keywords, not exact title only.
   - Report status: `exists: task_id/status`, `related tasks exist`, or `no direct task found`.

## Digest shape
Use a compact table:

| Idea | Context | Feasibility | Kanban status |
|---|---|---|---|

Then add a short **Priority call** section with 3-5 concrete recommendations. Do not create tasks from this digest unless the cron prompt explicitly asks for task creation.

## Feasibility notes
Classify feasibility in plain language:

- **High** — already scoped, small blast radius, existing repo/task/PR, or read-only/task-creation pilot.
- **Medium-high** — strong value but touches prompts, routing, automation, or shared behaviour.
- **Medium** — concept is clear but needs product/design/platform decisions.
- **Low / defer** — community polish, unclear owner, or weak mission fit.

## Pitfalls
- Do not treat every bugfix as a build idea. Include only work that points to a reusable project, product, feature, automation, plugin, or capability.
- Do not output a status-only digest if all findings are stale and already closed with no next action.
- Do not claim “no kanban task” until checking at least obvious keyword matches against the board.
- Avoid huge raw session dumps in the visible message. Summarise the candidate idea and cite enough context for Sahil to recognise it.
