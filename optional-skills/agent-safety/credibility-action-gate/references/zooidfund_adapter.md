# Zooidfund Adapter

Use this only for zooidfund-like humanitarian campaign review. The core skill does not depend on zooidfund and should stay neutral.

## Mapping

- Campaign evidence review maps to lane `evidence`.
- Public web/context corroboration maps to lane `external_context`.
- Recipient and donation history checks map to lane `graph_history`.
- Donation budget, repeat/top-up rules, and mission constraints map to the operator `policy`.

## CauseClaw Lessons To Preserve Without Copying Persona

- Fetch the full active campaign corpus before reviewing individual campaigns. Sequential fetch-and-review can miss repeated-recipient patterns that appear later in the same corpus.
- Public disaster, school, food, or medical context supports only need plausibility. It does not prove claimant linkage or use of funds.
- Many similar search results, copied snippets, or repeated campaign titles are not independent corroboration.
- Repeat or top-up support needs a fresh marginal reason: new claim-specific evidence, new urgency, changed funding state, strong prior linkage, or no better current candidate at only a tiny action size.
- A no-action result can be a valid receipt: it records why funds were not moved on the current record.

## Suggested Zooidfund Policy Choices

Keep these in the operator policy, not in the core skill:

- Maximum autonomous donation amount.
- Smallest test donation amount.
- Whether direct human welfare outranks platform, animal, environmental, or development campaigns.
- Repeat/top-up limits.
- Any legal, jurisdiction, restricted-party, or platform-specific blockers.
