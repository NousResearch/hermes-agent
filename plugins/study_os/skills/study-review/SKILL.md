---
name: study-review
description: Run flexible StudyOS spaced-repetition reviews.
platforms: [linux, macos, windows]
---

# StudyOS Review

Use for 复习, daily review, 艾宾浩斯 drills, and spaced repetition. Load turn-local
context with `study_activity(resource="prompt_context", action="load",
data={"intent":"reviewing"})`; never mutate system prompts.

## Queue

Call `study_activity(resource="review", action="due", data=...)`. Default:
`{"review_state":"due","sort":"priority","limit":10}`. Do not broaden
the user's requested scope without asking.

Combine selectors (categories are ANDed):

- Selected questions: `notes:["course/examples/a.md"]`
- Topic: `subjects:[...]`, `tags:[...]`, `concepts:[...]`
- Difficulty: `difficulties:["easy","hard"]`
- Level: `review_levels:[0,1]`, `min_review_level`, `max_review_level`
- State: `review_state:"due"` (default), `"new"`, `"reviewed"`, or `"all"`
  (targeted non-due practice).
- Multi-value topic matching: `match:"any"` (default) or `"all"`.
- Order: `priority`, `oldest`, `newest`, `difficulty_asc`, `difficulty_desc`,
  or `title`.

For ambiguity, give the queue count and ask one scope question. For an empty
selection, report filters and offer to relax one constraint; never switch to all.

## Loop

1. Read one selected note with `study_activity(resource="note", action="read",
   data={"note":"...","include_body":true})`. Present only its question.
   Never reveal a solution, hint, or grade before the answer. Give a minimal
   hint only when requested and increment `hints_used`.
2. Grade after the answer as `correct`, `partial`, or `incorrect`; point out
   missing conditions/invalid reasoning and give a concise correction. Missing
   a required condition is not fully correct.
3. Ask for self-confidence (1-5) if absent, then record exactly once:

   `study_activity(resource="review", action="submit", project_id="...",
   data={"note":"...","response":"...","result":"correct|partial|incorrect",
   "duration_seconds":0,"self_confidence":1-5,"hints_used":0,"diagnoses":[]})`

   `review.submit` atomically saves the attempt and advances spacing. Do not
   pair `attempt.record` with `review.record` for the same answer.
4. Continue only if submission succeeds. If it fails, say the result was not
   recorded and retry it; do not move on or invent a count.
5. End with attempted/correct/partial/incorrect counts, weak concepts, and one
   next action. Call `study_activity(resource="memory", action="sync")` when
   available.
