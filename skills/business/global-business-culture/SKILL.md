---
name: global-business-culture
description: "Business-culture briefs by region: trust, deals, etiquette."
version: 1.0.0
author: Eric Wong (pswong661-eng), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Business, Culture, Negotiation, Cross-cultural, Etiquette, International]
    category: business
    related_skills: []
---

# Global Business Culture Skill

Produces a structured business-culture brief for any country or region: how trust is built, how decisions are really made, how contracts and time are treated, what etiquette and taboos apply. Grounded in the documented research frameworks (Meyer, Hofstede, GLOBE, Lewis, Trompenaars) plus practitioner field notes from frontier markets, and it adapts the user's approach to the counterpart's culture instead of exporting one Western playbook worldwide.

## When to Use

- Preparing a cross-border negotiation, sales visit, JV discussion, supplier audit, or M&A management meeting
- Drafting messages to counterparts in another culture (tone, directness, pacing, who is addressed)
- Planning meetings or trips: delegation seniority, seating, gifts, small talk, meal protocol
- Diagnosing a stalled cross-cultural deal ("they went quiet", "they say yes but nothing moves")
- Hiring, managing, or giving feedback across cultures
- Not for: legal/tax structuring advice (flag for counsel) — this skill covers behavior, not law

## Prerequisites

None. Optional: `web_search` for time-sensitive facts (public holidays, Ramadan dates, current visa or entry rules, recent political changes that change etiquette).

## How to Run

1. Identify the country AND the task (negotiate, first meeting, follow-up email, manage staff, collect payment). The task changes which parts of the brief matter.
2. Load the matching region file with `skill_view` (see Quick Reference). Read the within-region variation section — Singapore is not Indonesia; Chile is not Brazil.
3. Produce the Culture Brief using the fixed schema in Procedure.
4. Country not covered? Find its GLOBE cluster, Lewis type, and Meyer/Hofstede anchors in `references/frameworks.md`, extrapolate, then verify with `web_search`.
5. For frontier, informal, or least-developed economies, ALWAYS overlay `references/ldc-playbook.md` on top of the region file.

## Quick Reference

| Context | File | Headline pattern |
|---|---|---|
| Frameworks + fallback method | `references/frameworks.md` | Meyer 8 scales, Hofstede 6, GLOBE 10 clusters, Lewis 3 types |
| US/Canada/UK/AU, Germanic, Nordic, Latin Europe | `references/western.md` | Low-context, task-based trust, contract is the deal |
| China, Japan, Korea, HK/TW/Singapore | `references/east-asia.md` | Reactive, consensus/face, contract begins the relationship |
| ASEAN incl. Laos (deep), Thailand, Vietnam, Indonesia | `references/southeast-asia.md` | Relationship-first, verbal > written, hierarchy + face |
| India, Pakistan, Bangladesh, Sri Lanka, Nepal | `references/south-asia.md` | Hierarchical, relationship-based, jugaad flexibility |
| Gulf, Levant, Egypt, Maghreb, Türkiye | `references/middle-east-north-africa.md` | Wasta, majlis, bargaining, insha'Allah semantics |
| Sub-Saharan Africa | `references/sub-saharan-africa.md` | Ubuntu, greetings + elders first, communal decisions |
| Mexico, Brazil, Andean, Southern Cone | `references/latin-america.md` | Personalismo, simpatía, flexible time |
| Frontier/informal/LDC overlay | `references/ldc-playbook.md` | Relationships substitute for weak contract enforcement |

## Procedure

Produce the Culture Brief — one screen, fixed schema, in this order:

1. **Positioning.** Country, GLOBE cluster, Lewis type, and where it sits on the three scales that matter most for the task (usually Trusting, Communicating, Deciding). Position is RELATIVE to the user's own culture (Meyer's rule: misread comes from the gap, not the absolute).
2. **Trust currency.** Task-based (track record, contracts) or relationship-based (meals, time, family, favors) — and the concrete acts that build it here.
3. **Decision path.** Who actually decides (senior elder? owner's family? consensus?), how long it takes, and what a "yes" really signals here.
4. **Contract attitude.** Contract as final word vs. photograph of the relationship vs. beginning of renegotiation. Match the drafting strategy to it.
5. **Communication + face.** Direct vs. indirect, high vs. low context, how "no" is said, how to disagree without causing face loss.
6. **Meeting protocol.** Greetings order, cards, seating, punctuality expectations (arrival ≠ start time in polychronic cultures), gifts, meals, small talk topics.
7. **Taboos + red lines.** The five things that kill the relationship here — including compliance red lines (see `references/ldc-playbook.md`).
8. **Sequencing recommendation.** What to do in what order for THIS task (e.g., Laos supplier deal: introduction via mutual contact → social meal before agenda → verbal understanding → short written memo → contract as relationship marker, not weapon).

State uncertainty explicitly: mark cluster-level extrapolations vs. country-documented facts. If the user's counterpart is a multinational-trained executive, note that corporate culture may outweigh national culture.

## Pitfalls

- **Cultures are distributions, not boxes.** Generation, sector, education, and time-abroad shift individuals; the brief describes the center of the distribution, not a person. Say so when applying it.
- **The mirror error.** A brief for a German reader dealing with Laos differs from one for a Lao reader dealing with Germany. Always anchor to BOTH sides of the gap.
- **Flexible time ≠ low commitment.** In polychronic cultures internal meetings flex, but a promised delivery to a relationship partner can be sacred. Do not "fix" one with the other's medicine.
- **Relationship-building ≠ facilitation payments.** Meals, respect, and patience build legitimate trust; cash or gifts to officials cross FCPA / UK Bribery Act lines. Never recommend the latter — see `references/ldc-playbook.md`.
- **Yes that isn't yes.** In high-context cultures "yes", "insha'Allah", "we will consider it", and "that would be difficult" encode refusal or non-commitment. Teach the user to read the follow-through, not the word.
- **Don't romanticize or rank.** No culture is "more advanced"; frameworks describe differences, not deficits.

## Verification

Before delivering the brief, confirm every item:

- [ ] All three key Meyer scales positioned, relative to the user's home culture
- [ ] Trust currency stated with concrete acts, not adjectives
- [ ] Real decision-maker identified or explicitly flagged unknown
- [ ] Contract attitude matched to a drafting recommendation
- [ ] At least three specific taboos or face-traps listed
- [ ] Country-documented facts distinguished from cluster extrapolation
- [ ] Compliance flag included when the context is LDC, extractives, or government-touching
- [ ] Sequencing recommendation given for the actual task
