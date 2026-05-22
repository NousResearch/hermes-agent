# Contested Decision

Use when the user is choosing between options with real tradeoffs — technology choices, design approaches, product directions, pricing structures, migration strategies.

## Shaping the prompt

Frame the deliberation around the actual decision, not around describing the options. "Should we use Postgres or DynamoDB for this workload?" is better than "Compare Postgres and DynamoDB." Include the constraints that make the decision hard — scale requirements, team expertise, timeline, reversibility. The more concrete the constraints, the more useful the disagreement.

If the user has a leaning, include it. Models that know the current favorite can attack it specifically rather than giving balanced pros/cons lists.

## What to look for in round 1

The claim map will usually show agreement on easy dimensions and disagreement on the dimensions that actually matter. The useful signal is *where* the panel splits, not whether it agrees overall.

Watch for:
- **Criteria smuggling** — models agree on the answer but disagree on *why*, which means they'd diverge on a slightly different version of the question.
- **Unstated assumptions** — one model assumes the team has Postgres expertise, another doesn't. The disagreement isn't about the technology; it's about a context gap.
- **Reversibility asymmetry** — if one option is easy to reverse and the other isn't, that often dominates the other tradeoffs. Flag it with CORE if a model names it.
- **An option nobody defends well** — sometimes the best choice lacks an advocate. That's worth an EXPLORE.

## When to append

Append when the claim map shows a contested claim that's load-bearing for the decision. Use CHALLENGE on the weakest link in whichever position you find less convincing. Use EXPLORE if a model gestured at a consideration without developing it.

Don't append just because the panel is split. Some decisions are genuinely close calls — the deliberation's job is to surface why, not to force convergence.

## When to stop

Stop when you can articulate: "The panel agrees on X and Y. They disagree on Z, and the disagreement comes down to [specific assumption or priority]." That's usually enough for the user to decide. You don't need the panel to reach consensus.
