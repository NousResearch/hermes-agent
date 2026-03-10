---
name: bayesian-synthesis
description: Use Bayesian reasoning to synthesize evidence and update beliefs. Leverages the Arbiter persona's methodology for calibrated decision-making.
version: 1.0.0
author: Hermes Ouroboros
license: MIT
metadata:
  hermes:
    tags: [Council, Bayesian, Decision-Making, Synthesis, Reasoning]
    related_skills: [multi-perspective-analysis, adversarial-critique]
---

# Bayesian Synthesis

The Arbiter persona uses explicit Bayesian reasoning to synthesize multiple perspectives into a calibrated verdict. Use this approach when you need to make decisions under uncertainty.

## The Bayesian Method

1. **State your prior**: What do you believe before seeing any arguments? (0-100%)
2. **Update on evidence**: For each new piece of evidence, adjust your belief
3. **State your posterior**: Final belief after all updates, with confidence interval
4. **Identify remaining uncertainty**: What would change your mind?

## When to Use

- **Decision-making under uncertainty**: Multiple valid options, unclear trade-offs
- **Forecasting**: Predicting outcomes with limited information
- **Evidence synthesis**: Combining multiple studies or data sources
- **Belief calibration**: Checking if your confidence matches available evidence

## How to Use

### Full Council (recommended for important decisions)
```
council_query(
    question="What is the probability that Ethereum's market cap surpasses Bitcoin's by 2028?",
    context="Consider technical fundamentals, adoption metrics, regulatory landscape, and historical precedent."
)
```

The Arbiter's synthesis will include:
- **PRIOR**: Starting belief before arguments
- **EVIDENCE UPDATES**: How each persona's argument shifts the belief
- **POSTERIOR**: Final calibrated estimate
- **KEY DISAGREEMENTS**: Unresolved tensions
- **FINAL VERDICT**: Actionable recommendation

### Quick Evaluation
```
council_evaluate(
    content="[your analysis here]",
    question="Is this analysis well-calibrated?",
    criteria=["calibration", "evidence_quality", "uncertainty_handling"]
)
```

## Interpreting Bayesian Updates

The Arbiter reports updates like:
```
PRIOR: 40%
Advocate's impact: +15% (strong technical argument, but speculative)
Skeptic's impact: -10% (valid concern about regulatory risk)
Oracle's impact: +5% (base rate of paradigm shifts is low but non-zero)
Contrarian's impact: -5% (alternative framing worth considering)
POSTERIOR: 45% (40 + 15 - 10 + 5 - 5)
```

**Key insight**: The magnitude of each update reflects evidence quality:
- Large updates (>10%): Strong empirical evidence or fatal logical flaw
- Medium updates (5-10%): Credible argument with partial evidence
- Small updates (<5%): Theoretical concern or weak analogy

## Evidence Hierarchy

The Arbiter weights evidence by quality:
1. **Empirical data** (strongest): Controlled studies, historical statistics
2. **Logical argument**: Valid deductive reasoning from established premises
3. **Expert consensus**: Agreement among domain experts
4. **Historical analogies**: Similar past situations and their outcomes
5. **Intuition** (weakest): Gut feeling or aesthetic preference

## Common Pitfalls

- **Anchoring**: Don't let your prior dominate. If evidence is strong, update significantly.
- **Base rate neglect**: Oracle's data matters. Most ventures fail, most predictions are overconfident.
- **Conjunction fallacy**: A specific story isn't more likely just because it's detailed.
- **Confirmation bias**: Pay special attention to the Skeptic's counter-evidence.
