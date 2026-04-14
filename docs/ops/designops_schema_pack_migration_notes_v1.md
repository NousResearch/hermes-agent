# DesignOps schema-pack migration notes v1 (`XD-102`)

Date: 2026-03-28  
Owner: Architect  
Status: active

## Scope
`XD-102` introduces a versioned schema/template pack for:
- token registry artifacts,
- component spec frontmatter artifacts,
- interaction contract artifacts.

## Migration summary

1. **Token registry now has explicit schema contract**
   - New schema: `ops/openclaw/architecture/schemas/design_token_registry.schema.json`
   - New template: `ops/openclaw/architecture/templates/design_token_registry.template.json`
   - Existing foundation artifact remains valid and is now schema-validated.

2. **Component spec frontmatter remains on v1 contract**
   - Existing schema/template stay canonical:
     - `ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json`
     - `ops/openclaw/architecture/templates/component_spec_template.md`
   - No breaking field rename/removal in this slice.

3. **Interaction contracts now have explicit schema contract**
   - New schema: `ops/openclaw/architecture/schemas/design_interaction_contract.schema.json`
   - New template: `ops/openclaw/architecture/templates/design_interaction_contract.template.yaml`

4. **Pack-level version registry added**
   - `ops/openclaw/architecture/design_schema_pack.v1.json` tracks canonical schema/template pointers.

## Compatibility posture
- Existing design component specs remain compatible.
- Existing token foundation registry remains compatible.
- New interaction contracts should migrate to the v1 structure immediately.

## Promotion guidance
- Promotion packets must reference `design_schema_pack.v1` assets.
- Any breaking changes to token/component/interaction schemas must publish `design_schema_pack.v2.json` (append-only promotion policy).
