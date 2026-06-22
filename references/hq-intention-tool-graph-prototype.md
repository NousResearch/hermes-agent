# HQ Intention-Tool Graph Prototype

## Overview
Read-only intention-tool compatibility matrix prototype inspired by SING (arXiv:2606.16591). No live pre-dispatch enforcement.

## Why now
- Planner source: `cron_331ecf7312ab_20260621_150248`
- upstream finding: intention-tool graph improved Recall@5 by 59.8% and success rate by 28.9% on 7,471-tool corpus while reducing schema exposure by 99.8% (SING, arXiv:2606.16591)
- current HQ weakness: pre-dispatch relies on static `enabled_toolsets` injection; no dynamic intent-aware tool selection

## Prototype Scope
- read-only analysis of existing HQ tool inventory
- no wiring into live cron/Gateway pre-dispatch
- no schema exposure reduction implementation in this phase

## Design

### Inventory Extraction
- source: `toolsets.py` `_HERMES_CORE_TOOLS` + registered tool schemas
- output: `reports/hq_tool_inventory_latest.json`
- fields per tool:
  - name, toolset, description hash, parameter schema hash

### Intention-Tool Compatibility Matrix
- sample intentions (from recent cron prompts + slash commands):
  - "run a test"
  - "edit a file"
  - "search the web"
  - "read a session"
  - "delegate a subtask"
- mapping: intention → top-k compatible tools by:
  - description semantic overlap (simple keyword/embedding hash)
  - historical tool-call frequency in similar tasks
- output: `references/hq-intention-tool-matrix.md`

### Prototype Limitations
- no live inference budget
- no model-backed embedding (use hash + keyword fallback)
- no pre-dispatch rewrite; reference-only for Review

## Next Steps (approval-gated)
- embed intention-tool matcher into `model_tools.py` pre-dispatch (requires live Gateway change, approval-gated)
- expand intention taxonomy from session DB
- A/B test schema exposure reduction

## Safety
- no secret access
- no live cron/Gateway mutation
- no paid-provider calls
- docs-only / read-only prototype
