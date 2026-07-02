---
name: technology-radar
description: >-
  Evaluate a creative technology (Blender add-on, CAD tool, rendering engine,
  3D/image/video generation model, agent framework, digital fabrication tool)
  and classify it as test-now, watch, adopt, replace-existing, or probably-hype
  based on maturity, cost, licensing, interoperability, and business relevance.
  Use this whenever the user asks "should I use/try/adopt X," mentions a new
  tool or model release and wants an honest read on it, or needs a structured
  comparison before switching workflows. Trigger proactively whenever a
  scouting cycle surfaces a new tool, plugin, or model — don't wait for the
  user to explicitly ask for a "radar" or "evaluation."
---

# Technology Radar

Read [../../references/operating-model.md](../../references/operating-model.md) first for the shared decision filter (revenue / leverage / business goal / creative momentum) this skill's classification should reflect.

## Objective
Track creative technologies and classify whether they are worth watching, testing, adopting, or ignoring.

## Domains
Track:
- Blender tools and add-ons
- CAD
- procedural modeling
- rendering
- real-time graphics
- game development
- 3D generation
- image generation
- video generation
- multimodal models
- agents
- automation
- open-source creative tools
- digital fabrication
- motion design
- asset pipelines
- collaboration tools

## Required Evaluation
For every technology, evaluate:

- problem solved
- target user
- maturity
- setup difficulty
- operating cost
- hardware requirements
- licensing
- privacy
- output quality
- editability
- interoperability
- reliability
- speed
- lock-in risk
- business relevance
- overlap with existing tools

## Classification
Assign one:

- `test_now`
- `watch`
- `adopt`
- `replace_existing`
- `use_for_content`
- `potential_product_opportunity`
- `probably_hype`
- `not_relevant`

## Evidence
Prefer:
- official documentation
- release notes
- repositories
- technical demos
- benchmark data
- real user reports
- issue trackers
- pricing pages
- licensing terms

Avoid relying only on:
- launch videos
- influencer enthusiasm
- company claims
- curated demos
- engagement counts

## Testing Recommendation
When recommending a test, define:

```yaml
goal:
time_limit:
success_criteria:
required_inputs:
comparison_tool:
expected_output:
failure_conditions:
decision_after_test:
```

## Radar Score
Score 0–5:

- relevance
- maturity
- leverage
- interoperability
- cost efficiency
- creative potential
- revenue potential

Subtract for:
- lock-in
- instability
- unclear licensing
- high operating cost
- poor editability
- weak evidence

## Output
```markdown
# Technology: [Name]

**Status:** Test now / Watch / Adopt / Ignore
**What it does:**
**Why it matters:**
**Best use case:**
**Weaknesses:**
**Cost and constraints:**
**Business impact:**
**Recommended test:**
**Confidence:**
```

## Rules
- Do not recommend replacing a stable workflow without a measurable advantage.
- New does not equal useful.
- Prefer tools that create editable outputs.
- Flag tools whose outputs are difficult to own, export, or reproduce.
- Note whether the tool creates revenue, leverage, or only novelty.
