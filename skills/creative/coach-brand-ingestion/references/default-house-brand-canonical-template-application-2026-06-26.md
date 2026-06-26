# Default house Brand Design Document canonical application

Use this when installing or resolving the default Brand Design Document for coach agents that have not yet completed a coach-specific brand ingestion exercise.

## Canonical rule
Every coach agent should have access to a polished default **The System Brand Design Document** and `brand-system-map.json` before coach-specific branding exists.

Resolution precedence:
1. **Temporary venue/event override** named in the prompt or registry for the current artifact only.
2. **Coach/profile-specific Brand Design Document** after that coach completes brand ingestion and the brand is installed locally.
3. **The System default Brand Design Document** as the fallback for coaches who have not completed brand ingestion.

The default is a high-quality starting template, not a permanent claim over the coach's brand. Once a coach-specific Brand Design Document exists, it takes precedence for that coach's normal artifacts.

## Local registry convention
Prefer a profile-local registry over Tailnet proof URLs for runtime use:

```text
~/.hermes/profiles/<profile>/home/brand-design-docs/BRAND_REGISTRY.json
~/.hermes/profiles/<profile>/home/brand-design-docs/the-system-default/default-brand-design-doc.pdf
~/.hermes/profiles/<profile>/home/brand-design-docs/the-system-default/default-brand-design-doc.html
~/.hermes/profiles/<profile>/home/brand-design-docs/the-system-default/brand-system-map.json
~/.hermes/profiles/<profile>/home/brand-design-docs/the-system-default/contact-sheet.png
```

Minimum `BRAND_REGISTRY.json` fields:

```json
{
  "schema_version": "brand-registry.v1",
  "default_brand_id": "the-system-default",
  "fallback_policy": "Use the-system-default for coaches/profiles without an installed coach-specific Brand Design Document. Coach/profile-specific Brand Design Documents take precedence after brand ingestion. Temporary venue/event overrides apply only to the current artifact and do not mutate defaults.",
  "brands": {
    "the-system-default": {
      "brand_id": "the-system-default",
      "name": "The System",
      "mode": "default_house_brand",
      "precedence": "fallback_only_until_coach_specific_brand_installed",
      "brand_design_document_pdf": ".../default-brand-design-doc.pdf",
      "brand_design_document_html": ".../default-brand-design-doc.html",
      "brand_system_map": ".../brand-system-map.json",
      "contact_sheet": ".../contact-sheet.png"
    }
  }
}
```

## Runtime behavior
- Before rendering a coach artifact, resolve the active Brand Design Document through the registry.
- If no coach-specific BDD is installed, use `the-system-default`.
- If a coach-specific BDD is installed, use it for normal coach outputs.
- If a temporary venue/event override is named, use it only for that artifact and preserve the coach/profile default.
- Do not depend on a Tailnet proof URL as runtime source truth when local registry files exist.

## Quality gate
Default brand installation is not complete unless:
- the PDF exists locally and is non-empty;
- HTML/source exists locally and is non-empty;
- `brand-system-map.json` exists and parses as JSON;
- the contact sheet exists for visual QA;
- the registry declares precedence correctly: default fallback only, coach-specific BDD wins after ingestion.
