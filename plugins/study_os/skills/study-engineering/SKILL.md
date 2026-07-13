---
name: study-engineering
description: Guide engineering and skill learning with StudyOS.
platforms: [linux, macos, windows]
---

# StudyOS Engineering Domain Pack

Use only with `domain_pack:"engineering.v1"`. Load the active workflow intent
with `study_activity(resource="prompt_context", action="load", data={"intent":"..."})`;
never mutate system prompts.

## Evidence-Driven Engineering Learning

1. Identify the workspace: `engineering-repo` for source exploration,
   `skill-vault` for durable concepts, or `hybrid` for both. Read the relevant
   code, docs, benchmark, command output, or paper before explaining it.
2. Define a concrete skill and observable artifact: trace a call path, explain
   an invariant, reproduce a benchmark, implement a small change, or compare
   two designs. Do not substitute a generic study plan for source inspection.
3. Create a concept note only when it blocks understanding, recurs across work,
   or will be reused. Every durable note needs a source anchor such as a file,
   symbol, command, benchmark, or paper.
4. Test understanding through a retrieval/application attempt and record actual
   evidence. Keep unverified claims separate from observed performance.

Avoid exam-vault defaults such as daily dashboards, Anki export, and full error
systems unless the user asks. Prefer lightweight, maintained records over a
large taxonomy of notes.
