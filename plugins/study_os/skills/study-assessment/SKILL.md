---
name: study-assessment
description: Analyze StudyOS exams and mistakes.
platforms: [linux, macos, windows]
---

# StudyOS Assessment

Use for mock exams, weekly review, 错题, and diagnostics. Call
`study_activity(resource="prompt_context", action="load",
data={"intent":"assessment"})` or use `"error_analysis"`; never mutate system prompts.

## Diagnose From Evidence

1. Set the scope: one attempt, a concept, a session, a week, or the project.
   Read existing attempts first through `study_activity`. Use `study_coach` with the same scope; it may
   summarize or recommend but does not prove unobserved dimensions.
2. For a new answer, classify outcome, reasoning, missed conditions, concept,
   pattern, confidence, and next action. Record an immutable `attempt` first;
   log an `error` only when a concrete failure needs durable remediation.
3. Use `study_activity(resource="review", action="weekly_report")` for a
   requested weekly artifact. Create `review.create_task` only for an accepted
   follow-up, not every recommendation.
4. For repeated evidence, request `study_coach.generate_probe`, ask one
   controlled retest before feedback, then record it as a new attempt. Keep
   pattern proposals as candidates until explicitly saved and validated.
5. Return the evidence ids, diagnosis, highest-impact next step, and what is
   still unverified. Sync memory only after a meaningful completed session.

Separate careless execution from missing conditions, concept confusion, and
method gaps. Never convert a score, a review count, or a single correct answer
into a mastery claim.
