# Compatibility Path Lifecycle Contract v1

Date: 2026-03-21  
Status: active (Wave 6 C2 governance substrate)

## Purpose
Prevent indefinite retention of degraded/compatibility paths by requiring explicit inventory and removal governance evidence.

## Required artifacts
- Register schema/template:
  - `docs/ops/schemas/compatibility_path_register.schema.json`
  - `docs/ops/templates/compatibility_path_register.template.json`
- Removal RFC schema/template:
  - `docs/ops/schemas/compatibility_path_removal_rfc.schema.json`
  - `docs/ops/templates/compatibility_path_removal_rfc.template.json`

## Governance requirements
1. Active compatibility paths must appear in a register object.
2. Paths in `removal_in_progress` state must reference a removal RFC.
3. Release evidence bundles must include compatibility lifecycle references.
4. Compatibility exceptions are temporary and must carry owner + target removal wave.

## Lifecycle states
- `active`
- `deprecated`
- `removal_in_progress`
- `removed`

## Operational expectation
For each release that keeps compatibility exceptions active, provide:
- updated register snapshot,
- removal RFC refs for in-progress removals,
- evidence links in release ladder bundle.
