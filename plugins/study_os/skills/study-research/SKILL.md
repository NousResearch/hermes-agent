---
name: study-research
description: Guide source-grounded research learning, paper comprehension, replication, claim evaluation, and hypothesis transfer with StudyOS.
---

# StudyOS Research Domain Pack

Use only with `domain_pack:"research.v1"`. Tie claims to exact sources. Call
`study_activity(resource="prompt_context", action="load", data={"intent":"teaching"})`;
never mutate system prompts.

## Flow

1. Read the objective and source anchors. Separate claim, reported evidence,
   learner inference, and uncertainty.
2. Start an explicit contract with `study_coach(action="start", data={...})`.
   Include one objective, source-backed `objective_ids`, assistance, time
   budget, and evidence targets. Reading and agent explanation are not evidence.
3. Perform its ActivitySpec. For replication, record method, environment,
   command, result, and divergence; for explanation, name an assumption and
   limitation.
4. Call `study_coach(action="advance", data={...})` with `evaluator` and
   `source_anchors`, plus `artifact_refs` for execution or transfer. Preserve
   failed and partial results.
5. Use `snapshot` to choose the next probe and `finish` when stopping. Report
   unverified dimensions.

## Research Integrity

Prefer exact locators and versioned artifacts. One replication supports only
its tested setup. Change one variable for near transfer; require a falsifiable
hypothesis and rejection condition for far transfer.
