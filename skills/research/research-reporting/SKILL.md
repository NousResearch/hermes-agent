---
name: research-reporting
description: Produce clear iteration memos and final research summaries from project artifacts. Use when converting experiment state, training outcomes, evaluation results, and open questions into reports for later review.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [reporting, memo, summary, writeup]
---

# Research Reporting

## Output shape

Every report should answer:
- what was attempted
- what changed
- what worked
- what failed or regressed
- what should happen next

## Style rules

- Prefer a short, decisive summary over a raw changelog.
- Separate observations from interpretations.
- Include uncertainty explicitly when the result is ambiguous.
- End with the next concrete action or stopping decision.

## Persistence

- Save iteration reports with `research_state(action="generate_report", ...)` or `research_loop(action="finish_project", ...)`.
- Leave an inbox item when the result should be reviewed later.
