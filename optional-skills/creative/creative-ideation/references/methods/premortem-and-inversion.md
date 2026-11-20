# Premortem and Inversion

Two methods for failure-oriented ideation:
- **Premortem** — Gary Klein, *HBR* September 2007. Imagine the project has already failed catastrophically; work backwards to causes.
- **Inversion** — Charlie Munger via Carl Jacobi: *"Tell me where I'm going to die so I'll never go there."* Solve problems by figuring out how to fail and avoiding that.

Both exploit prospective hindsight (Mitchell, Russo, Pennington 1989): people generate more concrete reasons for an event when imagining it has *already happened* than when imagining it might.

## When to use

### Premortem
- Choosing between project options
- Pressure-testing a near-term decision
- Late-stage planning for a long-horizon project
- Group decisions with social pressure suppressing dissent

### Inversion
- Strategic direction choice (easier to identify clear failures than clear successes)
- Personal life decisions (career, marriage, investments, health)
- Identifying hidden anti-patterns in your own behavior
- Designing systems against adversaries (security, abuse-prevention)

## Don't use when

- Early generative phase — corrosive to fragile ideas
- You can't act on the failure modes (anxiety, not planning)
- Group lacks psychological safety to articulate fears about the leader's project
- Decisions that need urgency (premortem takes 60–90 minutes done well)

## Premortem procedure

1. **State the project as if it's complete and failed.** "It is [date 6 months from now]. We launched. The result was a complete disaster."
2. **Generate failure narratives independently.** Each member writes a paragraph describing what happened, in concrete terms. *Independence is essential* — group brainstorming surfaces socially safe concerns; independent writing surfaces uncomfortable ones.
3. **Round-robin failure causes.** Each shares one cause; no comment. Continue until exhausted.
4. **Cluster and assess.** Group similar; estimate probability and severity.
5. **Generate mitigations for the top 3.** Update the plan.
6. **Re-run periodically.** Failures unlikely at planning time may have become likely.

## Inversion procedure

1. State the goal: "I want to [original goal]."
2. Invert: "How would I guarantee the *opposite*?"
3. List 5–10 things that would guarantee the inverted goal. Be specific.
4. Self-check: which am I accidentally doing or could drift into?
5. Avoid those; return to original goal.

## Worked inversion example

**Goal**: I want my open-source project to attract sustained contributors.

**Inversion**: how would I guarantee that no one ever contributes?

1. Have no CONTRIBUTING.md or unclear norms.
2. Reject PRs without explanation, slowly.
3. Make the build hard to reproduce locally.
4. Use a tone in issue threads that makes contributors feel stupid.
5. Use a license requiring CLAs new contributors won't sign.
6. Take 6+ months to merge anything.
7. Reply to issues with one-word answers.
8. Have only the founders in the maintainer org.

**Self-check**: which am I doing? Honest answer surfaces 2–3 of these. Those are the highest-leverage fixes.

## Anti-slop notes

- Premortem slop = generic risk lists ("execution risk", "market risk"). Real premortem narrative says *specifically* what went wrong.
- Inversion slop = "do the opposite of successful people" — that's contrarianism. Real inversion identifies *specific* failure-guaranteeing actions in *your* situation.
- Don't generate fake fears. If there are no real concerns, the premortem is short.
- Don't use these to talk users out of pursuing things they should pursue. Premortem and inversion are pressure tests, not vetoes.

Source: Klein, "Performing a Project Premortem", *HBR* Sept 2007. Munger, *Poor Charlie's Almanack* (PCA, 2005).
