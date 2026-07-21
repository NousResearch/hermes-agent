# Wave 1 — Findings

## Repository and codebase

`GoalManager` records a judge `done` as `judge_done_unconfirmed`; explicit
`/goal confirm` is required before a receipt becomes `achieved_confirmed`.
Reusable outcomes depend on fresh session/workspace verification, but the
receipt previously hashed only the headline goal. File tools marked edits
stale, while foreground terminal activity did not.

## External evidence

OpenAI separates durable sessions, human-in-the-loop state, and tracing in its
[sessions](https://openai.github.io/openai-agents-python/sessions/),
[HITL](https://openai.github.io/openai-agents-python/human_in_the_loop/), and
[tracing](https://openai.github.io/openai-agents-python/tracing/) references.
Anthropic's [long-running harness guidance](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
and [evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
support durable artifacts and reproducible evaluation criteria.

## Decision

Add a privacy-preserving criteria digest to receipts and conservatively stale
workspace evidence after foreground terminal execution. Do not add autonomous
learning, prompt injection, or background-process provenance.
