# Red Team

Use when the user wants adversarial review — finding failure modes, abuse cases, security gaps, things that will break under pressure, or reasons a plan will fail.

## Shaping the prompt

Be explicit that you want adversarial thinking: "Try to break this. What fails first?" or "What's the most likely way this goes wrong in production?" Models default to helpful analysis; you need to push them toward constructive hostility.

Include the thing being attacked in full — the API spec, the auth flow, the launch plan, the pricing model. Vague descriptions produce vague attacks.

If there's a specific threat model, state it. "Assume a motivated attacker with a valid API key" produces better results than "find security issues."

## What to look for in round 1

Red team rounds produce two kinds of findings:
- **Concrete attack paths** — "If you send X to endpoint Y, Z happens." These are immediately actionable. Mark them CORE.
- **Structural concerns** — "This design assumes trusted clients." These are often more important but harder to act on. Mark them EXPLORE to force specifics.

Watch for models pulling punches. If a response reads like "this is generally well-designed but you might consider..." that model is being helpful, not adversarial. Use CHALLENGE: "Be more specific. What actually breaks?"

## When to append

Append when attacks were shallow or when a model identified a category of risk without a concrete scenario. Use the round prompt to escalate: "Assume the easy fixes are done. What's the next layer of risk?"

Also append when models disagreed about severity — a CHALLENGE on "this is a minor issue" forces justification.

## When to stop

Stop when the attacks have gotten specific enough to be actionable. Present findings by severity, not by model. The user needs a punch list, not a transcript. Don't keep appending until every conceivable failure mode is exhausted.
