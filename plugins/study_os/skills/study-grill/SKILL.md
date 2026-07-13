---
name: study-grill
description: Bridge grilling sessions into StudyOS decisions.
platforms: [linux, macos, windows]
---

# StudyOS Grill

Use for a strategic learning decision, not routine planning, 整理, 复习,
assessment, or 错题 repair. Call `study_activity(resource="prompt_context",
action="load", data={"intent":"planning"})` and inspect existing `decision`
records; never mutate system prompts.

## Decision Flow

1. Define the decision, its owner, deadline, constraints, and reversible versus
   irreversible consequences. Search project notes or source material before
   asking the learner for discoverable facts.
2. Follow `/grilling`: ask one high-leverage question at a time, state a
   recommendation with reasoning, and compare concrete options rather than
   producing motivational prose.
3. Persist with `study_activity(resource="decision", action="create")` only
   after a stable decision is accepted. Include options, consequences, concepts,
   sources, and linked sessions. Do not write one for an open brainstorm.
4. Hand reusable knowledge to `study-organize`; route the resulting action plan
   to `study-plan`. Report the decision id or say clearly that nothing was
   persisted.
