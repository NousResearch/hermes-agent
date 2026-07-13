---
name: study-engineering
description: Guide engineering and skill learning with StudyOS.
---

# StudyOS Engineering Domain Pack

Use only with `domain_pack:"engineering.v1"`. Load the intent through
`study_activity(resource="prompt_context", action="load")`; never mutate system prompts.

## Evidence-Driven Engineering Learning

1. Identify an `engineering-repo`, `skill-vault`, or `hybrid` workspace. Read
   the real code, docs, benchmark, command output, or paper before explaining.
2. Define an observable skill: trace a call path, explain an invariant,
   reproduce a benchmark, implement a change, or compare designs. Call
   `study_coach(action="start", data={...})` with objective, assistance, time,
   and evidence targets.
3. Perform its ActivitySpec in the workspace. Call `study_coach.advance` with
   `evaluator` and source anchor; add `artifact_refs` such as a command, test,
   trace, benchmark, diff, or file for execution and transfer.
4. Create a concept note only when it blocks understanding, recurs across work,
   or will be reused. Every durable note needs a source anchor such as a file,
   symbol, command, benchmark, or paper.
5. Use `study_coach.snapshot` to choose a probe and `study_coach.finish` when
   stopping. Separate unverified claims from observed performance.

Avoid exam-vault defaults such as daily dashboards, Anki export, and full error
systems unless the user asks. Prefer lightweight, maintained records over a
large taxonomy of notes.
