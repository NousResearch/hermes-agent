---
name: weekly-creative-brief
description: >-
  Produce a concise weekly report — executive summary, top signals, visual
  trends, technology radar, community pain points, content opportunities,
  product/service opportunities, competitor movement, and ranked recommended
  actions — synthesized from everything the scouting system found in the last
  seven days. Use this whenever the user asks for a weekly brief, roundup,
  digest, or status report on design/tech scouting, or wants a single
  actionable document instead of a stream of individual signals. Trigger
  proactively at the end of a scouting cycle once enough signals, clusters, and
  pain points have accumulated to synthesize, even if the user just says
  "what's the state of things" or "give me the rundown."
---

# Weekly Creative Brief

Read [../../references/operating-model.md](../../references/operating-model.md) first — this skill synthesizes outputs from every other skill in the plugin, so it should rank and phrase things consistent with the shared operating model.

## Objective
Produce a concise weekly report of the most relevant design, technology, workflow, trend, and opportunity signals.

## Reporting Window
Default to the previous seven days.

## Required Sections

### 1. Executive Summary
Maximum five sentences.

### 2. Top Signals
Include the five most important signals.

For each:
- what happened
- why it matters
- evidence
- recommended action

### 3. Visual Trends
Include up to three:
- trend name
- stage
- evidence
- relevance
- experiment

### 4. Technology Radar
Include:
- test now
- watch
- ignore

### 5. Community Pain Points
Include up to three repeated problems.

### 6. Content Opportunities
Include the three strongest ideas.

### 7. Product or Service Opportunities
Include only opportunities with meaningful evidence.

### 8. Competitor and Market Movement
Include material changes only.

### 9. Recommended Actions
Maximum five actions.

Each action must include:
- owner
- effort
- expected value
- deadline or timing

### 10. Noise Filter
State what appeared important but was excluded and why.

## Ranking
Rank by:
1. business relevance
2. strategic leverage
3. creative value
4. urgency
5. evidence quality

## Output Template
```markdown
# Weekly Creative Intelligence Brief
**Period:** YYYY-MM-DD to YYYY-MM-DD

## Executive Summary
...

## Top Signals
### 1. ...
...

## Visual Trends
...

## Technology Radar
### Test Now
...

### Watch
...

### Ignore
...

## Community Pain Points
...

## Content Opportunities
...

## Product and Service Opportunities
...

## Competitor and Market Movement
...

## Recommended Actions
1. ...

## Noise Filter
...
```

## Rules
- Keep the brief below 1,500 words unless requested otherwise.
- Do not pad sections.
- Omit empty sections.
- Use links and citations.
- Do not repeat the same signal in multiple sections without adding new meaning.
- End with actions, not observations.
