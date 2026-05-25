# Physical Product / 3D-Printable CAD Workflow Reference

Use this reference when an idea is a physical product, 3D-printable object, hardware accessory, or digital STL/3MF product rather than a software app.

## Core shift

Do not force app-centric implementation sections like database, auth, backend, and deployment when they do not apply. Replace them with CAD/manufacturing/product-validation concerns.

## Recommended implementation sections

For physical-product ideas, cover:

- CAD source-of-truth choice
- Parametric model structure
- Dimension/reference data needed
- Model variants/SKUs
- Materials and manufacturing/printing process
- Tolerances and clearances
- Safety, fit, serviceability, and access
- Export/package formats
- Prototype phases
- Physical validation plan
- Marketplace/listing assets if it is a digital product
- Legal/trademark/disclaimer notes when compatibility with branded products matters

## CAD tool defaults

For parametric 3D-printable products:

- Use CadQuery or OpenSCAD when a coding agent should generate repeatable variants.
- Use Fusion 360 or Onshape for human industrial-design refinement and polished commercial geometry.
- Use FreeCAD as an open-source edit/view fallback.
- Use Blender for renders/marketing visuals, not as the dimension-critical source of truth.

## Material planning defaults

Start with a material test matrix, not a single unsupported claim. Include:

- Target material for the first real prototype
- Easier rigid materials for fit gauges only when appropriate
- Secondary materials for later tests
- Print/service constraints and customer printability if files will be sold

Example pattern from a phone-case/AirTag product:

- Target final case material: TPU 95A
- Softer TPU variants: later comfort/flex tests
- PLA/PETG: rough fit gauges only, not final daily-use case
- Professional flexible printing: later if selling physical units

## Prototype phase pattern

1. Reference geometry: product envelope, keep-out zones, mating parts, dummy inserts.
2. Placement/layout comparison: explore all plausible layouts before choosing one.
3. Rough shell/body: simple printable shape, not final surfacing.
4. Functional cavity/mechanism: retention, access, serviceability.
5. Visual variants: hidden/subtle/alternate exterior treatments.
6. Export review package: STL/STEP/3MF plus screenshots/renders.
7. Physical test print or prototype.
8. Tolerance revision based on real fit.
9. Marketplace/manufacturing package.

## Agent handoff guidance

A coding agent can usually build:

- Parametric CAD project skeleton
- YAML/JSON dimension configs
- Variant/export scripts
- Basic STL/STEP exports
- README/print settings/test plan docs

A human or physical test loop is still needed for:

- Ergonomic feel
- Premium surfacing judgment
- Exact tolerances
- Real-world fit and durability
- Commercial-quality validation photos

## Acceptance tests before claiming sellable

For a downloadable physical product, require real-world validation before commercial claims:

- At least one real prototype/test print in the target material
- Fit and function checks against the actual object/device
- Material-specific printability notes
- Clear model compatibility list
- Export files open in common slicers/viewers
- Marketplace readme/disclaimers are present
- Branded compatibility language avoids implying official affiliation
