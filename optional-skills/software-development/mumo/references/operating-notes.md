# Operating Notes

Practical agent mechanics that don't belong in the kernel but matter during real sessions.

## Recovery

Use `list_sessions` to recover sessions when:

- the agent restarted mid-session
- a round timed out but may still be running
- multiple sessions are active

Use `get_session` for full state. Prefer `wait_for_round` while a round is in flight.

## Idempotency

MCP write tools are idempotent by arguments. Retrying the same call after a transport failure replays the same operation rather than creating duplicates.

Don't make meaningless edits to a prompt just to "try again" — that creates a new operation.

## Synthesis boundary

mumo's deliberation surface is raw participant output plus moderator attention. Don't write your synthesis back into mumo as an `append_round` unless the user explicitly wants a synthesis round.

Your final answer to the user can freely synthesize your read of the panel. Make the distinction clear: mumo produced the raw material; your response is your interpretation for the user.

## Platform meta in edge cases

The kernel's test — "does this help participants think about the user's problem?" — covers most cases. A few edge cases worth noting:

- **Model behavior is the user's topic.** If the user is evaluating model capabilities, model-comparative observations are problem context, not platform meta. Include them.
- **Budget constraints are real.** If the user has said they're watching costs, "this is a $0.50/round panel" is useful context for deciding whether to append. But frame it as a decision input, not a report.
- **Tool failures.** If a model failed to respond and the round is incomplete, that's worth noting in your prompt or to the user. It changes the epistemic state of the session.
