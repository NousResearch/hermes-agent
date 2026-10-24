# Product Architecture Profile

Use for apps, games, tools, workflows, platforms, and creative systems.

## Triggers

product audit, app design, game design, system design, feature coherence, product boundary, experience surface.

## Runes

- `RUNE.NG.PRODUCT.BOUNDARY`
- `RUNE.NG.PRODUCT.SYSTEM_INVENTORY`
- `RUNE.NG.PRODUCT.LOOP_MAP`
- `RUNE.NG.PRODUCT.CONFLICT_SCAN`
- `RUNE.NG.PRODUCT.EXPERIENCE`
- `RUNE.NG.PRODUCT.COST_SURFACE`
- `RUNE.NG.PRODUCT.REGRESSION_SCAN`
- `RUNE.NG.WAYFINDER_HANDOFF`

## Required analysis

- target user and job-to-be-done;
- blocked transition the product completes;
- core mechanism (what must be true for value to exist);
- product boundary (in / out / deferred);
- primary and secondary loops;
- feature interaction and orphan features;
- information architecture;
- emotional and sensory pacing where applicable;
- technical and production burden;
- validation path and acceptance criteria;
- canon-versus-implementation drift risks;
- integration surface and unknown access.

## Conflict scan (fail or flag)

- competing primary loops;
- features that undermine the core mechanism;
- scope that requires undeclared authority or access;
- success metrics that cannot be observed externally.

## Outputs

- `NeonGenieProductPacket` (primary)
- optionally `WayfinderExecutionPacket` when handoff is requested and intent is stable
- `NeonGenieRunReceipt`

Schema: `schemas/product-packet.schema.json`

## CLEAR rules

- Do not invent technical feasibility; mark `NOT_COMPUTABLE` when critical access is unknown.
- Do not silently expand product intent in handoff packets (Gate H).
