# Brand Design Document local registry + coach-agent handoff

## Session lesson
When a coach agent runs on the same host as the approved Brand Design Documents, do not make the agent depend on a Tailnet/public-proof URL as its runtime source of truth. Tailnet proof links are for operator/Jamess review; profile-local files are better for the agent runtime.

## Preferred pattern
Install approved Brand Design Documents into the target coach profile under a local registry directory, for example:

```text
~/.hermes/profiles/<profile>/home/brand-design-docs/
  BRAND_REGISTRY.yaml
  BRAND_REGISTRY.json
  <default-brand>/...
  <temporary-override>/...
```

The registry should name:
- the default brand root
- any temporary brand overrides
- local PDF/HTML paths
- SHA-256 hashes for each artifact
- explicit rules that temporary overrides do not mutate the default brand root
- the rule that local paths are the runtime source when present

## Example registry shape

```yaml
schema_version: brand-registry.v1
owner_profile: darin
rules:
  - Resolve the active Brand Design Document before creating any branded artifact.
  - Use it as source truth for identity, palette, typography, visual style, and application guidance.
  - If the prompt names a temporary brand override, use it only for that artifact and do not mutate default_brand.
  - If no temporary override is named for Bryan/Darin lesson recap work, default to Hermitage CC.
  - Do not use Tailscale/public-proof URLs as runtime source when local profile paths are available.
default_brand:
  id: hermitage-cc
  brand_design_document_pdf: /Users/.../.hermes/profiles/darin/home/brand-design-docs/hermitage-cc/hermitage-cc-brand-design-doc-YYYYMMDD.pdf
  sha256:
    pdf: <hash>
temporary_brand_overrides:
  congressional:
    status: temporary_venue_brand_override
    brand_design_document_pdf: /Users/.../.hermes/profiles/darin/home/brand-design-docs/congressional/congressional-brand-design-doc-YYYYMMDD.pdf
    reverts_to: hermitage-cc
    mutation_rule: Using this override must not change default_brand.
```

## Why this matters
- no network/proof-server dependency for the coach agent
- deterministic local reads and hashes
- separates operator-visible proof transport from runtime source truth
- prevents temporary event/venue brand links from becoming accidental defaults

## Scope discipline
Copy only approved artifacts into the profile registry. Do not read or copy secrets, tokens, `.env`, session DBs, or unrelated profile state. Do not mutate Workspace, email, or Drive as part of local registry installation.
