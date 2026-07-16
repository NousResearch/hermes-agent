# BETA-003 — Intent and specialist router

`agent.beta.router.route_request()` returns a structured `RoutingDecision` containing intent, selected specialist IDs, rationale, initial risk, delegation need, parallelism, and confidence.

Specialist selection is data-driven: request concepts are matched against each enabled manifest's `capabilities` and `keywords`. The router contains no specialist ID allow-list, so a new manifest can become selectable without router changes.

The router is deterministic and local. It adds no model call, core tool, or mutable prompt state. More semantic routing can replace the scoring later if measured requests exceed these explicit aliases.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_router.py
```
