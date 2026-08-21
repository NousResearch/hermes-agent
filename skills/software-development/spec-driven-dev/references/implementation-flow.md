# Implementation Flow

The orchestration logic that connects the 5 stages. Use this as the
authoritative gating table; if a stage is unclear, re-read its section here.

## Stage gates

| Stage | Produces | Freeze rule | May amend later? |
|-------|----------|-------------|------------------|
| 1. Constitution | `constitution.md` | User accepts in writing | Yes, via Amendment log only |
| 2. Spec | `spec.md` | All Open questions resolved or moved out; AC list final | Only if AC prove un-testable; record reason |
| 3. Plan | `plan.md` | All spec acceptance criteria mapped to implementation steps | Yes, before tasks freeze |
| 4. Tasks | `tasks.md` | Each task is 2-5 min and TDD-complete | Until stage 5 starts |
| 5. Implement | git history + AC checkmarks | When `subagent-driven-development` reports all tasks green | N/A |

## When to stop and ask the user

- Spec stage: any criterion you cannot make binary. Do NOT write "TBD" -- ask.
- Plan stage: a tech choice with no clear winner. Do NOT pick arbitrarily.
- Tasks stage: a slice cannot be made under 5 min without dropping a step.
  Ask whether to merge or split.

## When NOT to stop

- Constitution stage: trivial wording tweaks. Edit inline.
- Spec stage: renaming a user story, tightening a criterion. Edit inline.
- Plan stage: reordering file layout, swapping a non-load-bearing lib.
- Tasks stage: typo fixes, command-line flag corrections.

## Hand-off to implementation

Once `tasks.md` is frozen:

1. Load `optional-skills/software-development/subagent-driven-development`.
2. Dispatch the first task per that skill protocol.
3. After each task lands, mark the corresponding AC in `spec.md`.
4. On spec deviation: stop, append a `## Deviations` section to `spec.md`,
   surface the deviation to the user before proceeding.

## Failure recovery

- **Implementer asks a clarifying question:** answer in writing; update
  `plan.md` if the answer changes a tech choice.
- **Reviewer flags spec gap:** add an acceptance criterion to `spec.md` and
  re-derive affected tasks in `tasks.md` before re-dispatching.
- **Constitution violation discovered mid-implement:** STOP. Surface the
  violation. Either amend the constitution (with rationale) or fix the plan.
  Never silently violate.

## Anti-patterns

- Skipping straight from spec to implementation ("the plan is obvious").
  A plan is rarely obvious once you write tasks.
- Letting tasks.md drift from spec.md. If AC-3 changes, update the tasks
  that reference AC-3 before the next subagent dispatch.
- Reopening constitution mid-feature. Amendments are expensive; verify
  before freezing.
