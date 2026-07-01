# Example Invocations

Use these as prompt shapes when invoking `cyber-vc-analyst`.

For Slack threads, prefer the `!cyber-vc-analyst ...` form because Slack blocks
native slash commands inside thread replies.

## 1. Standard company memo

```text
Analyze Noma Security as an early-stage cybersecurity investment.
Use my vault notes about prior meetings, use Return on Security MCP for market
intelligence if available, and save the final memo back into my vault.
```

## 2. Company plus known URL

```text
Create a cyber VC memo for Permiso using https://permiso.io as the primary
company reference. Pull in any prior notes from my vault, classify the company,
map it to investment themes and theses, and give me an Invest / Investigate /
Monitor / Pass recommendation.
```

## 3. Founder-led diligence

```text
Assess whether this startup is venture-backable:

Company: Twine
Founders: Jane Doe, John Smith
Category hypothesis: machine identity security

Use any notes in my vault from meetings with the founders, then use ROS market
intelligence if available to benchmark the market and competitor set.
```

## 4. Comparison-ready vault entry

```text
Produce a comparison-ready investment memo for SpecterOps' startup spinout.
Keep the output normalized so I can compare it against other cyber startups in
my vault later. Save the memo into the configured output folder.
```

## 5. Explicit note scope

```text
Analyze ProjectDiscovery as a cybersecurity investment opportunity.
Only use vault notes under:
- 4.Resources/Interactions
- 4.Resources/Companies
- 4.Resources/Categories/Matter

If Return on Security MCP is available, use it for category and competitor
analysis. Otherwise continue and mark that market-intelligence gap explicitly.
```

## 6. Private-context-first memo

```text
I met this company recently and want an IC-style note.
Analyze Token Security as a seed-stage startup. Start with my vault notes, then
separate private context from public evidence and assumptions throughout the
memo.
```

## 7. Red Access Security first pass

```text
Analyze Red Access Security as an early-stage cybersecurity investment.

Start with my vault notes about any meetings, founder conversations, or prior
market maps involving Red Access Security. Then use Return on Security MCP for
market intelligence if it is available.

Produce the full structured cyber VC memo:
- cybersecurity classification
- investment themes
- thesis mapping
- why now
- market analysis
- competitive landscape
- defensibility
- venture scoring
- key risks
- recommendation
- knowledge-base tags
- confidence assessment

Separate facts, reasonable inferences, and assumptions clearly throughout.
Save the final memo back into my vault in the configured output folder.
```

## 8. Slack company invocation

```text
!cyber-vc-analyst Analyze Red Access Security as an early-stage cybersecurity
investment. Start with my vault notes, use Return on Security MCP if available,
separate facts, inferences, and assumptions, and save the final memo back into
4.Resources/Companies.
```

## 9. Slack thematic invocation

```text
!cyber-vc-analyst Analyze the machine identity security theme as a venture
investment area. Use my vault notes plus Return on Security MCP if available.
Focus on market structure, buyer urgency, representative startups, likely
winners, hyperscaler risk, and what would make the theme attractive or weak.
Save the note only if it materially adds to my knowledge base.
```

## 10. Thematic analysis with write-back

```text
Analyze the browser security theme as a cyber venture investment area.

Use my vault notes and ROS market intelligence if available. I want a thematic
memo, not a company memo. Cover market maturity, buyer behavior, representative
startups, major incumbents, likely acquirers, why now, risks, and what signals
would increase conviction.

Save the final note into the configured thematic output folder.
```

## 11. Theme through a named company

```text
Analyze the agentic AI security theme through the lens of Red Access Security
and adjacent startups. This is a thematic memo, not just a company memo.
Use Red Access only as one datapoint in the broader market landscape.
```

## 12. SOC automation thematic invocation

```text
Analyze SOC Automation / AI SOC as a cyber venture investment thesis.

Use my vault notes first, then use Return on Security MCP for market
intelligence if available. Treat this as a thematic memo, not a company memo.
Cover SOC modernization, agentic investigation, SIEM-to-data-lake shifts,
buyer urgency, representative startups, incumbents, hyperscaler risk, and what
would make the theme structurally attractive or weak.

Save the note into the configured thematic output folder.
```

## Notes

- These prompts are intentionally short. The skill should recover the detailed
  structure from `SKILL.md` and `references/output-template.md`.
- If the user already knows the company stage or geography, include it in the
  prompt to reduce assumptions.
- If the user wants a market or category view, explicitly say `thematic memo`
  to avoid falling back to company mode.
