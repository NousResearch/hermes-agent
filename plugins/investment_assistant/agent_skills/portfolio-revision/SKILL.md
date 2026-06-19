---
name: portfolio-revision
description: Revise an existing portfolio map.
version: 0.1.0
---

# Portfolio Revision

Use this skill when the user asks to adjust an existing target portfolio map.
This stage edits a selected/base map; it does not discover a new theme, rerun
deep research, read current holdings, or create orders.

## Inputs

- user revision request
- base portfolio map
- portfolio architect selection result
- deep-research report when available
- policy constraints

## Process

1. Parse the user's natural-language request into a general patch.
   - Do not hardcode separate workflows for every request.
   - Use generic edits such as `adjust_weight`, `adjust_sleeve`,
     `add_symbol`, `remove_symbol`, `replace_symbol`, `change_style`,
     `preserve_position`, and `request_research`.
   - If magnitude, target symbol, or base map is ambiguous, ask for
     clarification instead of editing.

2. Revise the map from the patch.
   - Preserve cash weight unless the user explicitly asks to change it.
   - Preserve total sleeve weight.
   - Respect single-name limit.
   - Keep required symbols unless the user explicitly asks and the validator
     allows it.
   - Do not introduce unresearched symbols unless the patch marks that more
     research is required.
   - Explain where increased weight is funded from.

3. Output a reviewable revision.
   - Include a revised map.
   - Include change summary.
   - Include tradeoffs and risk delta.
   - Include reduced or removed holdings.
   - Require user confirmation before it becomes selected.

## Boundaries

- No buy/sell/hold recommendations.
- No orders.
- No price targets.
- No current-holdings analysis.
- No new market facts outside supplied artifacts.
