---
name: cognitive-evidence-architecture
description: Evaluate scientific, philosophical, social, and behavioral claims using structured epistemic analysis, evidence quality assessment, causal reasoning, confidence calibration, and alternative hypothesis generation.
version: 1.0.0
author: <ibrhmuyls>
license: MIT

metadata:
  hermes:
    tags:
      - reasoning
      - epistemology
      - critical-thinking
      - research
      - cognitive-science
      - causal-inference

    category: research
---

# Cognitive Evidence Architecture (CEA)

## Overview

Cognitive Evidence Architecture (CEA) is a structured claim evaluation framework.

The purpose of this skill is not to determine whether a claim is true or false.

Instead, it evaluates:

- conceptual clarity
- logical consistency
- evidence quality
- causal support
- alternative explanations
- confidence level

The framework is inspired by contemporary work in:

- analytic epistemology
- cognitive science
- causal inference
- evolutionary theory
- neuroscience

Use this skill whenever a claim requires careful evaluation under uncertainty.

---

## When to Use

Use this skill when:

- evaluating scientific claims
- evaluating social science claims
- evaluating psychological claims
- evaluating economic explanations
- evaluating historical interpretations
- evaluating philosophical arguments
- assessing competing explanations
- distinguishing correlation from causation
- estimating confidence in uncertain conclusions

Do NOT use this skill for:

- mathematical proofs
- legal advice
- medical diagnosis
- live news verification
- simple factual lookups

---

## Procedure

### Step 1 — Extract the Claim

Rewrite the user's statement as a precise evaluable claim.

Example:

Input:

"Social media causes depression."

Claim:

"Social media use increases risk of depressive symptoms."

---

### Step 2 — Conceptual Analysis

Identify:

- undefined terms
- ambiguous concepts
- hidden assumptions

Flag conceptual weaknesses.

---

### Step 3 — Logical Analysis

Evaluate:

- internal consistency
- missing premises
- unsupported inference steps
- logical validity

---

### Step 4 — Evidence Assessment

Classify evidence quality.

Tier 1:
Anecdotes

Tier 2:
Single study

Tier 3:
Multiple studies

Tier 4:
Meta-analysis

Tier 5:
Replicated meta-analyses or strong scientific consensus

---

### Step 5 — Causal Analysis

Determine whether the claim is:

- causal
- correlational
- speculative

Evaluate:

- confounding variables
- reverse causality
- alternative mechanisms

---

### Step 6 — Alternative Hypotheses

Generate at least three plausible competing explanations.

Never stop after the first explanation.

---

### Step 7 — Domain Modules

Apply only if relevant.

#### Evolutionary Module

Check:

- adaptation hypothesis
- byproduct hypothesis
- cultural evolution hypothesis

#### Neuroscience Module

Check:

- causal evidence
- intervention evidence
- correlational limitations

#### Incentive Module

Check:

- incentives
- strategic behavior
- game-theoretic effects

---

### Step 8 — Confidence Calibration

Assign exactly one category:

- Strongly Supported
- Moderately Supported
- Mixed Evidence
- Weakly Supported
- Speculative
- Unsupported

Never output absolute certainty.

---

## Output Format

Use the following structure:

CLAIM

CONCEPTUAL ANALYSIS

LOGICAL ANALYSIS

EVIDENCE QUALITY

CAUSAL ANALYSIS

ALTERNATIVE HYPOTHESES

DOMAIN MODULES

CONFIDENCE CALIBRATION

FINAL EPISTEMIC ASSESSMENT

---

## Pitfalls

Common failure modes:

### Failure 1

Treating correlation as causation.

### Failure 2

Assuming a single explanation.

### Failure 3

Treating one study as scientific consensus.

### Failure 4

Using evolutionary storytelling as evidence.

### Failure 5

Treating neural activation as proof of causality.

### Failure 6

Confusing confidence with truth.

---

## Verification

Before finalizing an assessment verify:

- Claim has been explicitly extracted.
- Evidence tier has been assigned.
- At least three alternative hypotheses were generated.
- Correlation vs causation has been addressed.
- Confidence category has been assigned.
- No absolute truth claims were made.

The assessment is incomplete if any verification item fails.
