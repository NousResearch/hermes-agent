# Example 1 - "what's next?" answered with a map instead of a move

Source: a real end-of-day moment in my own exam-practice platform. Condensed;
no client or third-party material.

## What the agent answered

> Current state: feat/scoring-fix is merged, feat/results-email is WIP on my
> branch (email template done, SMTP wiring pending), PR #31 (mobile layout)
> has green CI awaiting review. Backlog: PostHog events not instrumented,
> testimonials section still stubbed, pricing page copy TODO, oral-exam
> simulator needs the TTS voice swapped. Also the Vercel preview for #31 404s
> on the results route, investigating. What should I prioritize?

Everything true, nothing usable. The person asked for the next move and got
the dependency graph, ending in the question they had just asked the agent.

## `whats-next` mode

> The scoring fix landed earlier, so practice tests grade correctly again.
> Nothing was ready to finish in this pass: the mobile layout has green checks
> but no review yet, and its preview breaks on the results page, so it stays
> open until that is traced.
>
> **I can do alone:** trace the broken results page on the mobile-layout
> preview (its pull request) - it blocks the review; finish the results email
> (my branch) - no result reaches the student today; wire the two analytics
> events (backlog) - the launch decision needs them.
> **Needs you:** two real student testimonials (backlog) - only you can ask
> them; the ads budget stays parked until testimonials and analytics exist -
> your earlier rule.
>
> **Next I take: the broken results page on the mobile-layout preview - small,
> and until it is fixed nobody can review that work.**
> I did not check the ad account or the payment provider's dashboard.

## What the rewrite did

| Move | Where |
|---|---|
| Nothing was merged that had not earned it | green checks with no review and a broken preview is a claim, not a finish - it stays open with its blocker named (step 2, last gotcha) |
| The graph became two lists | doable-alone and needs-you; the broken preview became an item with its why, not a section of its own |
| Every item names its source | the pull request, my branch, the backlog, the user's own standing rule on ads - nothing invented (rule 2) |
| One move, with its why | what it unblocks, in the product's words, not the branch's |
| The unread was named | an unchecked source is not an empty source (rule 5) |
| The closing question disappeared | the skill decides; asking back "what should I prioritize" is the failure being fixed |
