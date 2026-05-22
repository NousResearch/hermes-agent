# Design Review

Use when the user has a proposed system, API, architecture, schema, or plan and wants it critiqued. The input is a design; the question is "what's wrong with it" or "what would we regret."

## Shaping the prompt

Include the design itself — paste the spec, schema, API surface, or plan into the prompt or the `reference` field. Models can't review what they can't see.

Frame it as a review, not a redesign. "Review this API surface for consistency, missing edge cases, and things we'd regret at scale" is better than "Design a better API." If you want alternatives, make that a separate deliberation.

State the constraints the design was built under. Models that don't know the constraints will critique decisions that were already intentional.

## What to look for in round 1

Design reviews produce two kinds of signal:
- **Convergent concerns** — multiple models flag the same issue. These are almost always real. Mark them CORE.
- **Isolated concerns** — one model raises something the others didn't. These are either the most valuable insight in the session or noise. Use EXPLORE to find out which.

Watch for models critiquing style rather than substance. The claim map can flatten severity — naming conventions and concurrency bugs get equal-weight rows. Read the prose to distinguish what actually matters.

## When to append

Append when a model raised a concern that feels real but underdeveloped. Use EXPLORE with a comment like "this seems important — can you be more specific about the failure mode?" or just "say more about this."

Also append when models disagree about whether something is actually a problem — a CHALLENGE on a design critique forces the critic to defend it with specifics, or when a hidden constraint needs to be injected that would change the analysis.

## When to stop

Stop when the review has surfaced concrete, actionable concerns. The user doesn't need the models to agree on a revised design — they need a list of things to worry about, ranked by the panel's attention pattern. Summarize: "Three models flagged X. Two raised Y but disagreed on severity. One raised Z, which the others didn't engage with — worth investigating."
