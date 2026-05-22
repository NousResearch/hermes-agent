# Snippets — Extended Guidance

Snippets are moderator attention. This reference expands on the kernel's snippet guidance with examples and edge cases.

## What makes a good snippet

A good snippet has a specific quote and a genuine reaction. The reaction can take many forms:

- **Reflective** — "This feels like the crux of the disagreement."
- **Evaluative** — "This reasoning is strong but assumes constant network latency."
- **Skeptical** — "I'm not convinced this scales past 10K users."
- **Clarifying** — "This seems to be saying X — is that right?"
- **Directive** — "Develop this further in the next round."
- **Minimal** — No comment at all. The snippet type alone is sometimes enough.

You do not need a next-step directive before selecting a snippet.

## What makes a bad snippet

- **Block quotes** — Quoting a 500-word passage. Quote the specific claim that sparked your reaction.
- **Process narration** — "I am using this CHALLENGE to redirect the panel." Just react.
- **Detached platform meta** — "This model used the most tokens." That's about the session, not the problem.
- **Private audit notes** — "I think this model is underperforming today." Keep that local unless model behavior is the user's actual topic.

## Types in practice

**KEEP** is "this resonates with me." Use it when a claim is valuable and should remain visible.

**EXPLORE** is "there's an undeveloped thread here." Use it when a model mentioned something promising but moved on. EXPLORE without a comment is fine — the type itself says "say more."

**CHALLENGE** is "I'm not sold." Use it when you have a specific reason for skepticism, even if you can't fully articulate the counter-argument. "I don't think this holds under load" is a good CHALLENGE comment.

**CORE** is "this is what it comes down to." Use it sparingly — maybe once or twice per round. It marks the claim that the decision hinges on. If everything feels equally important, nothing is CORE.

**SHIFT** is "this changed the frame." Use it when a model recast the problem in a way that reorients your reasoning. SHIFT is the rarest type and the most powerful.

## Snippet count

There is no ideal number. Your goal is to steer additional deliberation via directly attributed reactions and commentary.

## Prompt vs. snippets

Use `append_round.prompt` for broad steering:
- "Focus on the caching layer. The storage question seems resolved."
- "Assume attribution accuracy matters more than density."

Use snippets for situated reactions:
- CORE on the exact trust invariant.
- CHALLENGE on a claim that over-merges null attribution.
- EXPLORE on an underdeveloped mitigation.

If both are useful, use both.
