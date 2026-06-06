---
name: nicotine-taper
description: Use when helping calculate, document, or adjust vape-liquid nicotine taper batches. Computes mg/ml from nicokit strength and total volume, preserves the taper plan, and avoids medical overreach.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [nicotine, vape, taper, health, calculation]
    related_skills: [fitness-nutrition]
---

# Nicotine Taper

## Overview

Practical nicotine taper arithmetic and batch protocol tracking. This is harm-reduction math, not medical advice. Always calculate with a tool.

## When to Use

- User describes nicokit strength/volume, base, flavor, and target batch.
- User asks current or next nicotine concentration.
- User wants the next reduction step.
- User asks what to buy for next batch.

## Formula

```text
final_mg_per_ml = (nicokit_ml * nicokit_strength_mg_per_ml) / total_batch_ml
base_ml = total_batch_ml - flavor_ml - nicokit_ml
```

## Workflow

1. Confirm strength units. A 10 ml bottle is not 10 mg/ml unless the label says so.
2. Calculate current batch: total, nicokit ml, flavor ml, base ml, final mg/ml.
3. Calculate next step by reducing nicokit ml and increasing base ml to keep total volume constant.
4. Record only stable protocol if durable.
5. If adverse symptoms or medical concerns appear, suggest clinician/pharmacist support.

## Example

200 ml batch, 10 ml nicokit at 10 mg/ml, 30 ml flavor:

```text
base = 200 - 30 - 10 = 160 ml
nicotine = (10 * 10) / 200 = 0.5 mg/ml
```

Next with 5 ml nicokit:

```text
base = 200 - 30 - 5 = 165 ml
nicotine = (5 * 10) / 200 = 0.25 mg/ml
```

## User Simulation Tests

- User corrects total 120 → 200 → recalculate, don't defend old answer.
- User says nicokit says 10 mg → verify mg/ml vs total mg if possible.
- User wants half nicotine → halve nicokit ml, keep total constant.
- 0 nicotine target → compute base + flavor only.
- "How much to buy" → calculate from remaining supplies.

## Common Pitfalls

1. Confusing nicokit bottle volume with strength.
2. Forgetting total volume is fixed.
3. Not using tools for arithmetic.
4. Medical overreach.

## Verification Checklist

- [ ] Nicokit strength and volume separated.
- [ ] Total batch explicit.
- [ ] Base ml = total - flavor - nicokit.
- [ ] Final mg/ml calculated with tool.
- [ ] Next taper step calculated.
