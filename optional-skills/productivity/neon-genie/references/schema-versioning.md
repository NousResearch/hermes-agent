# Schema versioning policy

**OBSERVED:** Skill package version (`VERSION` / frontmatter) and **artifact schema
versions** are separate lifecycles as of v3.20.

## Identifiers

Every sealed envelope carries:

```json
{
  "schema_id": "neon-genie/run-envelope",
  "schema_version": "1.0.0"
}
```

Artifact entries may include `schema_id` + `schema_version` when known.

| Artifact | schema_id | Current |
|----------|-----------|---------|
| Run envelope | `neon-genie/run-envelope` | `1.0.0` |
| Run receipt | `neon-genie/run-receipt` | `1.0.0` (+ privacy provenance fields) |
| Privacy context | `neon-genie/privacy-context` (file `privacy-context.schema.json`) | `1.0.0` |
| Opportunity packet | `neon-genie/opportunity-packet` | `1.0.0` |
| Product packet | `neon-genie/product-packet` | `1.0.0` |
| Wayfinder handoff | `neon-genie/wayfinder-execution-packet` | `1.0.0` |
| Capital sprint packet | `neon-genie/capital-sprint-packet` | `1.0.0` |

JSON Schema files remain flat under `schemas/*.schema.json` (mirrored to
`references/schemas/`). Directory-per-version trees may be introduced later
without breaking envelope discovery.

## Privacy on envelopes and receipts (3.24.0+)

As of skill **3.24.0**, packaging always co-loads the `privacy` profile and emits
privacy provenance via `scripts/privacy_runtime.py`:

- **Receipt:** top-level fields such as `privacy_mode` (`local_only` default),
  `external_actions`, `telemetry_status: disabled`, plus nested `privacy`
  object conforming to `privacy-context.schema.json`.
- **Envelope:** includes the same privacy context (and often top-level
  `privacy_mode`) so consumers that open `run-envelope.json` first see the
  boundary without digging.

Envelope `schema_version` remains **1.0.0**; privacy is carried as optional
additive properties under the existing envelope major (not a separate 1.1.0
summary-only object). Packaging validation enforces dual-enforcement gates
`NG-PRIV-*` (telemetry off, no successful send under `local_only`, sealed
provenance completeness).

Machine-facing summary: `references/privacy-contract.md`. Human contract:
root `PRIVACY.md` / hub `references/PRIVACY.md`.

## Compatibility rules

| Change | Schema version impact |
|--------|------------------------|
| Add optional field | **minor** (1.0.0 → 1.1.0) |
| Tighten validation of optional data | **minor** if still optional |
| New required field / rename / type change | **major** (1.x → 2.0.0) |
| Remove field | **major** |

## Support window

- Wayfinder (and other consumers) **must** accept the current major schema_version
  for `run-envelope`.
- Previous major: best-effort for one skill minor series, then drop.
- Packaging CLI validates current major only unless `--schema` points at an older file.

## Migrations

When a major envelope change ships:

1. Document in CHANGELOG under Schema.
2. Add `schemas/migrations/README.md` note (or a script) describing field moves.
3. Keep previous `run-envelope` schema file renamed if dual validation is required.

## Canonical consumer path

1. Open `run-envelope.json` in the run directory.
2. Read `primary_artifact.path` for the main packet.
3. Read `receipt_path` / `receipt` for gates, DataRequests, and privacy provenance.
4. If `wayfinder.handoff_path` is set, load that packet for execution planning
   (Wayfinder runtime remains **optional**).
