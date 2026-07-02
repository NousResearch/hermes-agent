---
name: trend-clusterer
description: >-
  Group related design/technology signals into coherent, descriptively-named
  themes and track whether each theme is seeding, accelerating, established,
  saturated, or declining, using momentum evidence like source diversity,
  cross-domain appearance, and repository/product activity. Use this whenever
  the user has multiple related signals and asks "is this actually a trend,"
  "group these together," or "what's the momentum on X," or wants to update an
  existing trend cluster with new evidence. Trigger proactively once
  `signal-scout` or `visual-trend-analyzer` has surfaced several signals that
  look related rather than waiting for an explicit "cluster these" request.
---

# Trend Clusterer

Read [../../references/operating-model.md](../../references/operating-model.md) first for the signal record and category taxonomy each cluster's member signals should already carry.

## Objective
Group related signals into coherent themes and track whether those themes are gaining or losing momentum.

## Inputs
- signal records
- embeddings
- tags
- source metadata
- timestamps
- engagement
- prior clusters

## Process
1. Compare new signals with existing clusters.
2. Merge near-duplicates.
3. Create a new cluster only when needed.
4. Require multiple independent sources for trend status.
5. Track first seen, last seen, and velocity.
6. Identify cross-domain movement.
7. Record counter-signals.
8. Update stage and confidence.

## Cluster Record
```yaml
cluster_id:
name:
description:
first_seen:
last_seen:
signal_count:
source_count:
source_types:
domains:
key_attributes:
representative_examples:
counter_examples:
momentum:
stage:
confidence:
relevance:
commercial_value:
creative_value:
recommended_action:
```

## Trend Stages
- `seed`: one or two early signals
- `emerging`: multiple independent signals
- `accelerating`: rapid increase in frequency or cross-domain adoption
- `established`: widely visible
- `saturated`: heavily copied
- `declining`: falling attention or relevance
- `false_signal`: insufficient evidence

## Momentum
Estimate from:
- frequency over time
- source diversity
- creator diversity
- domain diversity
- engagement velocity
- search growth
- repository activity
- product launches
- repeated community discussion

## Naming
Use descriptive names grounded in observable attributes.

Good:
- `Inflated protective forms`
- `Technical translucency`
- `Soft industrial interfaces`

Bad:
- `The future of design`
- `Next-gen aesthetics`
- `Innovation wave`

## Output
```markdown
# Trend Cluster: [Name]

**Stage:**
**Momentum:**
**First seen:**
**Last updated:**

## Core Pattern
...

## Evidence
- ...

## Cross-Domain Appearance
- ...

## Counter-Signals
- ...

## Relevance
...

## Recommended Action
...
```

## Rules
- Do not force unrelated signals together.
- Do not use one viral example as a cluster.
- Preserve disagreement.
- Split clusters that become too broad.
- Archive clusters that stop producing useful information.
