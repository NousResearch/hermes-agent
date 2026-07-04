# Label taxonomy — examples per label

Use these to pick 1–3 labels each turn. Titles are lowercase; pass them exactly
to `chatwoot_labels` `assign_labels`.

## gig-discovery

- "What gigs are near me?"
- "How do I find gigs in Explore?"
- "Are there any open gigs this week?"
- Browsing availability, store locations, gig listings

## gig-execution

- "How do I submit proof?"
- "Give me details about the Amazon gig?"
- "Tell me about the Night Before gig"
- "My submission was rejected" (pair with `handoff-escalation`)
- "What do I need for this gig?"
- Active gig work, deadlines, proof requirements, resubmission

## payment-payout

- "Did I get paid?"
- "When will I be paid?"
- "Show my payment history"
- Dot payout status, timing, sent-but-not-received (disputes → add `handoff-escalation`)

## app-navigation

- "Where is Home vs Explore?"
- "How do I open a gig?"
- "Where do I see my active gigs?"
- UI wayfinding only — not broken behavior (that's `troubleshooting`)

## troubleshooting

- Link won't open, page won't load, button does nothing
- App or site appears broken after normal steps
- Pair with topic label when broken UI blocks the main ask (e.g. payout page +
  `payment-payout`)

## handoff-escalation

- Frustration, anger, repeated unresolved issue
- Rejected submission needing human explanation
- Genuine money dispute, ban, legal question
- Always **add alongside** the topic label, not alone unless intent is unclear

## account-membership

- Account banned or suspended
- Membership status, eligibility
- Account settings the bot cannot change

## general-inquiry

- CRWD questions that don't fit other labels
- Company/policy questions without a clearer bucket
- Use sparingly when nothing else fits
