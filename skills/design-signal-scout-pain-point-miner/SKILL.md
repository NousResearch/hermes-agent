---
name: pain-point-miner
description: >-
  Mine Reddit, GitHub issues/discussions, forums, and community comments for
  repeated frustrations, unmet needs, and workflow bottlenecks in design and
  technology communities, then map recurring pain points to a possible product,
  plugin, service, or content response. Use this whenever the user asks "what
  are people frustrated with in X," wants to find an underserved workflow
  problem, is validating a product/plugin idea, or asks to dig into complaints
  and feature requests around a tool or domain. Trigger proactively when a
  scouting cycle turns up recurring complaints, "I wish there was," or
  workaround language that hasn't yet been evaluated for commercial potential.
---

# Community Pain-Point Miner

Read [../../references/operating-model.md](../../references/operating-model.md) first for shared scoring conventions and the `product_opportunity` / `pain_point` signal categories this skill writes into.

## Objective
Find repeated frustrations, unmet needs, and workflow bottlenecks in design and technology communities.

## Target Sources
Prioritize:
- Reddit
- GitHub issues
- GitHub discussions
- YouTube comments
- product forums
- Blender communities
- Discord channels with permission
- public Telegram channels
- software reviews
- support forums
- creator comments
- Q&A sites

## Search Language
Look for:

```text
how do I
is there a tool for
does anyone know
I wish
why can't
this takes forever
there has to be a better way
workflow for
plugin for
add-on for
script for
automate
broken
frustrating
too many steps
manual process
can't export
doesn't support
```

## Process
1. Collect complaints and questions.
2. Remove duplicates.
3. Normalize the underlying job-to-be-done.
4. Group similar pain points.
5. Count independent users and sources.
6. Identify current workarounds.
7. Evaluate willingness to pay.
8. Map the pain point to:
   - product feature
   - plugin
   - service
   - tutorial
   - template
   - automation
   - content

## Pain-Point Record
```yaml
problem:
user_type:
context:
frequency:
severity:
current_workaround:
workaround_cost:
evidence_count:
source_diversity:
willingness_to_pay:
existing_solutions:
solution_gap:
possible_offer:
confidence:
```

## Opportunity Score
Score 0–5:

- frequency
- severity
- urgency
- willingness to pay
- poor existing solutions
- fit with Matt's skills
- speed to prototype
- distribution potential

Subtract for:
- tiny niche
- no budget
- solved problem
- high support burden
- legal or platform risk
- unclear ownership

## Output
```markdown
# Pain Point: [Problem]

**Who has it:**
**What they are trying to do:**
**Why current methods fail:**
**How often it appears:**
**Current workarounds:**
**Existing solutions:**
**Gap:**
**Potential response:**
**Commercial potential:**
**Confidence:**
```

## Rules
- Do not confuse confusion with demand.
- A loud complaint is not automatically a market.
- Look for users already spending money or time on workarounds.
- Prefer recurring workflow pain over one-off bugs.
- Flag support-heavy opportunities.
