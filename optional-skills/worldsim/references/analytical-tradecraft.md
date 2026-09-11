# Analytical Tradecraft — Intelligence-Grade Analysis

Structured analytic techniques adapted from intelligence community
methodology. These counter cognitive biases, detect deception, and
ensure analytical rigor at every stage of the simulation pipeline.

## Core Principle

A single personality model treated as ground truth is NOT analysis.
Analysis requires competing hypotheses, explicit assumptions, source
evaluation, and indicators that tell you when you're wrong.

## 1. Analysis of Competing Hypotheses (ACH)

After compiling a dossier, ALWAYS generate 2-3 competing personality
hypotheses. Score each against the evidence.

### Template

```
COMPETING HYPOTHESES: @handle

H1 (PRIMARY): {description of most likely personality model}
  Evidence FOR: {list}
  Evidence AGAINST: {list}
  Consistency score: {X/10}

H2 (ALTERNATIVE): {description of alternative model}
  Evidence FOR: {list}
  Evidence AGAINST: {list}
  Consistency score: {X/10}

H3 (CONTRARIAN): {description of model that contradicts surface reading}
  Evidence FOR: {list}
  Evidence AGAINST: {list}
  Consistency score: {X/10}

ASSESSMENT: H1 at {confidence}%, H2 at {X}%, H3 at {X}%
KEY DISCRIMINATORS: {what evidence would shift between hypotheses}
```

### Common Competing Hypotheses

- "Genuinely holds these beliefs" vs "Strategically positioning for career/audience"
- "Personality is consistent across contexts" vs "Heavily performing for platform"
- "Recent shift is authentic" vs "Recent shift is strategic/temporary"
- "Contrarian takes are genuine conviction" vs "Contrarian for engagement/attention"
- "Combative style reflects personality" vs "Combative style is cultivated brand"

### When to Use ACH
- ALWAYS at fidelity 70+
- For any public figure with >50K followers (persona management likely)
- When evidence is contradictory
- When the subject is known for irony/satire

## 2. Key Assumptions Check (KAC)

Every dossier must list its key assumptions and rate their fragility.

### Mandatory Assumptions to Evaluate

| Assumption | Fragility | Notes |
|-----------|-----------|-------|
| Public persona reflects private personality | FRAGILE | Almost always partially false for public figures |
| Recent posts reflect current views | MODERATE | Usually true but crises/pivots happen |
| Cross-platform identity resolution is correct | MODERATE-FRAGILE | Common names = high risk |
| Posts are self-authored | FRAGILE for famous | Ghostwriting, comms teams, staff accounts |
| Stated positions are genuine (not ironic) | FRAGILE for satirists | Must detect irony markers |
| LLM latent knowledge is accurate | MODERATE | Generally good for famous, poor for obscure |
| Social media behavior generalizes to other contexts | FRAGILE | Platform behavior ≠ real behavior |

### Template
```
KEY ASSUMPTIONS: @handle
1. {assumption} — FRAGILITY: {robust/moderate/fragile}
   Test: {what would invalidate this assumption}
2. ...
```

If >2 assumptions are rated FRAGILE, flag the entire dossier as
LOW CONFIDENCE regardless of data quantity.

## 3. Red Hat Analysis (Persona Strategy Detection)

Model the target's strategic self-presentation. Ask:

- **What image are they cultivating?** (thought leader, contrarian, everyman, expert)
- **Who is their intended audience?** (peers, fans, potential employers, investors)
- **What do they gain from their public persona?** (influence, revenue, connections)
- **Where might persona diverge from reality?** (every public figure has gaps)
- **Do they have a comms team / ghostwriter?** (check for: scheduled posting,
  uniform formatting, brand-consistent messaging, never-breaking-character)

### Template for Dossier
```
STRATEGIC SELF-PRESENTATION:
  Cultivated image: {description}
  Target audience: {who they're performing for}
  Incentive structure: {what they gain}
  Possible divergences: {where persona may not equal person}
  Ghostwriting indicators: {present/absent, evidence}
```

## 4. Deception Detection

### Satire / Parody / Irony Detection

CHECK FOR:
- Bio markers: "parody", "satire", "not affiliated", "fan account", "views my own"
- Username patterns: "real{name}", "not{name}", "{name}but{modifier}"
- Absurdist content: internally contradictory statements, surreal humor
- Irony markers: quotes around words, "/s" tags, "love that for us",
  "surely {absurd thing} won't happen", extreme hyperbole
- Tonal inconsistency: serious topic + flippant response pattern
- Account metadata: verified status, follower/following ratio anomalies

WHEN IRONY IS DETECTED:
- Flag that literal interpretation of positions may be INVERTED
- Look for "breaking character" moments where genuine views show
- Cross-reference with serious/long-form content (blog posts, interviews)
  where irony is typically lower
- In simulation: reproduce the ironic style, don't flatten it

### Sockpuppet / Alt Account Detection

INDICATORS:
- Heavy amplification (retweets/reposts) with little original content
- Posting patterns that mirror another account with time offset
- Follower graphs that overlap suspiciously with another account
- Voice analysis mismatch: claimed identity doesn't match writing style
- Account age vs sophistication mismatch

### Professional Persona Management

INDICATORS:
- Perfectly scheduled posting (on-the-hour times, regular intervals)
- No typos, no emotional outbursts, no 3am posting
- Brand-consistent messaging with no deviation
- Content themes match organizational talking points
- Engagement style is uniform (always positive, always professional)

WHEN DETECTED: note in dossier that voice profile may represent a
comms team, not an individual. Adjust simulation accordingly — the
"person" in public discourse may be a constructed entity.

### Persona Authenticity Score

Rate on 1-5 scale:

5 — AUTHENTIC: Consistent voice across platforms and time, includes
    vulnerable/unpolished moments, responds unpredictably to events,
    posts at irregular times, makes typos and corrections.

4 — MOSTLY AUTHENTIC: Generally consistent but some signs of curation.
    Occasional tone shifts that suggest awareness of audience.

3 — CURATED: Clear awareness of personal brand. Strategic topic selection.
    Some genuine moments but overall managed presentation.

2 — HEAVILY MANAGED: Strong indicators of professional management.
    Few if any unguarded moments. Uniform style and messaging.

1 — CONSTRUCTED: Likely ghostwritten or team-operated. Persona may not
    represent any single individual's actual personality.

## 5. Source Reliability Framework

Replace HIGH/MED/LOW with intelligence-grade evaluation.

### Source Reliability (A-F)
- **A — COMPLETELY RELIABLE**: Subject's own verified account, direct quotes in published interviews they reviewed
- **B — USUALLY RELIABLE**: Established journalism quoting the subject, verified tweets, conference transcripts
- **C — FAIRLY RELIABLE**: Aggregator sites paraphrasing, third-party profiles, LinkedIn
- **D — NOT USUALLY RELIABLE**: Anonymous posts attributed to subject, unverified cross-platform matches
- **E — UNRELIABLE**: Scraper artifacts, login-walled content, LLM confabulation
- **F — CANNOT JUDGE**: First-time discovery, unverified handle, cached deleted content

### Information Confidence (1-6)
- **1 — CONFIRMED**: Corroborated by independent sources across platforms/occasions
- **2 — PROBABLY TRUE**: Consistent with known pattern, logically coherent
- **3 — POSSIBLY TRUE**: Single-source, not independently confirmed
- **4 — DOUBTFULLY TRUE**: Inconsistent with some known information
- **5 — IMPROBABLE**: Contradicted by other information, likely outdated or satirical
- **6 — CANNOT JUDGE**: Insufficient basis

### Application
Tag key dossier entries: `"Subject advocates open-source AI" [B2]`
Use combined rating to weight evidence in simulation.

## 6. Temporal Intelligence

### Phase Transition Detection

People go through identifiable life phases that alter behavior:
- Career changes (new job, founding company, getting fired)
- Ideological shifts (political realignment, religious conversion)
- Personal crises (public breakdowns, divorces, health issues)
- Platform migrations (leaving Twitter for Bluesky)
- Growth/maturation (early-career edginess → senior-role diplomacy)

### Detection Method

1. **Timeline construction**: Plot key events and posting pattern changes
2. **Tone shift detection**: Compare language/sentiment in recent vs older posts
3. **Topic shift detection**: What they talked about 2 years ago vs now
4. **Network shift detection**: Who they interact with now vs before
5. **Self-reference detection**: "I used to think..." "I've changed my mind about..."

### Phase-Aware Simulation

When a phase transition is detected:
- Weight post-transition data MUCH higher (2-3x)
- Flag pre-transition data as historical context, not current personality
- Note the transition in the dossier: "Major shift detected around {date}: {description}"
- Consider whether the shift is genuine or performative (ACH)

## 7. Indicators & Warnings (I&W)

After every simulation, list 3 observable indicators that would
invalidate the prediction:

```
INVALIDATION INDICATORS:
1. If @handle {does X instead of Y}, our {trait} estimate is wrong
2. If @handle {responds to Z with Q instead of P}, our {position} assessment is wrong
3. If @handle {interacts with @person in manner M}, our social dynamics model is wrong
```

These serve as:
- Self-correction mechanisms (check after real events)
- Honesty signals (we know what we don't know)
- Learning opportunities (when predictions fail, update the model)

## 8. Counter-Bias Checklist

Run before finalizing any dossier:

- [ ] **Confirmation bias**: Did I search for evidence that CONTRADICTS my model?
- [ ] **Anchoring**: Am I over-weighted on the first information I found?
- [ ] **Availability bias**: Am I over-weighted on viral/memorable moments?
- [ ] **Mirror imaging**: Am I assuming the subject thinks like me?
- [ ] **Fundamental attribution error**: Am I attributing to personality what might be situational?
- [ ] **Recency bias**: Am I ignoring valid older evidence?
- [ ] **Halo effect**: Is one strong trait coloring my assessment of other traits?
- [ ] **Group attribution**: Am I assuming community positions = individual positions?

If any box is checked "yes" or "maybe", revisit that section of the dossier.

## Integration Into Pipeline

### Phase 2 (Dossier Compilation) — ADD:
- Key Assumptions Check (mandatory)
- Red Hat Analysis (strategic self-presentation)
- Deception Detection (persona authenticity score)
- Source reliability tags on key data points

### Phase 2.5 (NEW) — Competing Hypotheses:
- Generate 2-3 competing personality hypotheses
- Score each against evidence
- Carry top 2 into simulation
- Note: simulation uses PRIMARY hypothesis but flags where
  ALTERNATIVE would produce different output

### Phase 5 (Self-Verification) — ADD:
- Counter-bias checklist
- Indicators & Warnings
- Devil's advocacy pass: "What would a critic say is wrong here?"
