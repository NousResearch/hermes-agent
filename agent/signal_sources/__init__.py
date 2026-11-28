"""Signal sources for the AIDE² SkillEvalProducer.

Each module exports one signal detector that consumes turn-level
metadata (user message text, task IDs, etc.) and produces one of the
private signals on ``SkillEval``:

- :mod:`user_correction_detector` — bool: did the user correct/retry
  in the next N turns after this turn?
- :mod:`rework_detector` — int: how many retries of the same task_id
  happened within the rework window?
- :mod:`reuse_tracker` — bool: when this skill was reused, did the next
  reuse succeed?

All detectors are pure functions over their inputs (no Hermes runtime
coupling) so they can be unit-tested in isolation and reused wherever
a signal needs computing.
"""
