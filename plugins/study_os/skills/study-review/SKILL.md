---
name: study-review
description: Run flexible StudyOS spaced-repetition reviews.
platforms: [linux, macos, windows]
---

# StudyOS Review

Use for 复习 and 艾宾浩斯 review. Load context with
`study_activity(resource="prompt_context", action="load",
data={"intent":"reviewing"})`; never mutate system prompts.

## Queue

Call `study_activity(resource="review", action="due", data=...)`. Default:
`{"review_state":"due","sort":"priority","limit":10}`. Do not broaden
the user's requested scope without asking.

Selectors combine with AND:

- Selected questions: `notes:["course/examples/a.md"]`
- Topic: `subjects:[...]`, `tags:[...]`, `concepts:[...]`
- Difficulty: `difficulties:["easy","hard"]`
- Level: `review_levels:[0,1]`, `min_review_level`, `max_review_level`
- State: `review_state:"due"` (default), `"new"`, `"reviewed"`, or `"all"`
  (targeted non-due practice).
- Topic match: `match:"any"` (default) or `"all"`.
- Order: `priority`, `oldest`, `newest`, `difficulty_asc`, `difficulty_desc`,
  or `title`.

If ambiguous, give the count and ask one scope question. If empty, report
filters and offer to relax one constraint; never switch to all.

## Loop

1. Read one selected note with `study_activity(resource="note", action="read",
   data={"note":"...","include_body":true})`. Present the question.
   Hide solutions and grading until the answer; give hints only on request and
   increment `hints_used`.
2. Grade after the answer as `correct`, `partial`, or `incorrect`; point out
   missing conditions/invalid reasoning and give a concise correction. Missing
   a required condition is not fully correct.
3. Ask for self-confidence (1-5) if absent, then record exactly once:

   `study_activity(resource="review", action="submit", project_id="...",
   data={"note":"...","response":"...","result":"correct|partial|incorrect",
   "duration_seconds":0,"self_confidence":1-5,"hints_used":0,"diagnoses":[]})`

   `review.submit` atomically saves the attempt and advances spacing. Do not
   pair `attempt.record` with `review.record` for the same answer.
   Non-empty `diagnoses` items are
   `{"kind":"condition_missed","evidence":"observed reason"}` objects, never strings.
4. Continue only if submission succeeds. If it fails, say the result was not
   recorded and retry it; do not move on or invent a count.
5. End with attempted/correct/partial/incorrect counts, weak concepts, and one
   next action. Call `study_activity(resource="memory", action="sync")` when
   available.
