---
name: signal-feedback-loop
description: >-
  Interpret Matt's reactions to surfaced signals (save, ignore, more/less like
  this, research deeper, turn into content, test tool, not relevant) and turn
  them into durable preference adjustments — which sources, topics, and signal
  types the scouting system should weight up or down — while preserving
  exploration so it doesn't overfit to one reaction. Use this whenever the user
  gives explicit feedback on a signal, tool, or brief, or asks for a summary of
  what's been working versus not. Trigger proactively after a
  weekly-creative-brief or batch of signals has been reviewed and reacted to,
  so preference tuning happens as a matter of course rather than only on
  request.
---

# Signal Feedback Loop

Read [../../references/operating-model.md](../../references/operating-model.md) first — preference changes made here should ultimately reshape how `source-manager` prioritizes sources and how `signal-scout` scores new candidates.

## Objective
Learn from Matt's reactions so the scout becomes more relevant over time.

## Supported Feedback
Interpret:

- `save`
- `ignore`
- `more_like_this`
- `less_like_this`
- `research_deeper`
- `turn_into_content`
- `add_to_project`
- `test_tool`
- `watch`
- `not_relevant`

## Feedback Record
```yaml
signal_id:
feedback:
reason:
timestamp:
related_project:
preferred_attributes:
rejected_attributes:
follow_up_action:
```

## Learning Rules
Increase preference weight when:
- similar signals are repeatedly saved
- similar tools are repeatedly tested
- similar topics become content
- signals lead to client work or revenue
- sources consistently produce useful findings

Decrease preference weight when:
- similar signals are ignored
- topics are repeatedly marked irrelevant
- sources produce low-value noise
- trends are too mainstream or too late
- recommendations create no action

## Important Constraints
- Do not overfit to one reaction.
- Require repeated feedback before major changes.
- Preserve some exploration.
- Keep at least 20 percent of results outside the strongest known preferences.
- Do not suppress strategically important negative signals.
- Do not infer sensitive personal traits.
- Store explicit preference changes with dates.

## Weekly Review
Summarize:
- strongest positive preferences
- strongest negative preferences
- source performance
- category performance
- content outcomes
- business outcomes
- recommended tuning changes

## Output
```markdown
# Scout Feedback Update

## Learned Preferences
- ...

## Reduced Priorities
- ...

## Source Changes
- ...

## Exploration Topics
- ...

## Configuration Changes
- ...
```
