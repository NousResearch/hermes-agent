# License and Attribution Boundary

Reader: commercial owner, delivery engineer, legal reviewer.
Next action: prepare a customer-deliverable package without overstating model,
Honcho, or dependency rights.

This document is a packaging checklist, not legal advice.

## Source of Truth

Observed local source facts:

- Repository `LICENSE` is MIT and names Nous Research copyright.
- `pyproject.toml` declares `hermes-agent` with MIT license text.
- Root `package.json` declares MIT.
- Stage package docs require BYO model endpoint, no Dobby weights, and no
  bundled or auto-run Honcho server.

Drift risk: dependency licenses, optional plugin licenses, model terms, and
Honcho terms can change. Recheck primary sources before customer delivery.

## Hermes MIT Notice

Any customer distribution containing Hermes source, substantial portions of the
source, or compiled/package artifacts must include:

- The Hermes MIT license text.
- The Nous Research copyright notice from the root `LICENSE`.
- Any additional third-party notices required by bundled dependencies.

Do not remove or obscure upstream notices when rebranding the package as
Dobby/Hermes.

## Dobby Weights Are Not Bundled

The package must not include Dobby model weights, checkpoints, adapters, merged
weights, quantized weights, or generated model artifacts.

Permitted claim:

- The customer brings a model endpoint compatible with Hermes provider
  configuration.

Forbidden claim:

- The package includes, redistributes, licenses, hosts, or grants rights to
  Dobby weights.

Delivery gate:

- Artifact scan finds no model weight files or model bundles.
- Sales and setup docs say BYO model endpoint.

## BYO Model Endpoint

The customer is responsible for:

- Provider account and API key.
- Model endpoint URL.
- Provider terms, data retention, security, and billing.
- Any model-specific license or acceptable-use obligations.

The package may provide configuration, health checks, and redacted diagnostics.
It must not imply that Hermes or the package vendor grants model usage rights.

## Honcho AGPL Caveat

The default package must not bundle, install, auto-run, or require a Honcho
server. Native Hermes memory is the default package path.

If a customer asks for Honcho:

- Treat Honcho as an optional external component with AGPL-related commercial
  review required before delivery.
- Verify the exact Honcho license and version from primary sources at delivery
  time.
- Keep Honcho artifacts, service configuration, source obligations, and support
  terms separate from the default Dobby/Hermes package unless legal approves a
  specific distribution model.
- Do not claim that V1 includes managed Honcho memory or Honcho server hosting.

Delivery gate:

- Default package artifact contains no Honcho server bundle.
- Default onboarding works without Honcho configuration.
- Any optional Honcho appendix is marked separate and legal-reviewed.

## SBOM Requirement

Every customer-delivered artifact needs an SBOM that covers:

- Hermes package source and version metadata.
- Python dependencies from the resolved environment.
- Node dependencies for any delivered JS package surface.
- Container base image and OS packages, if containers are delivered.
- Optional plugins or MCP servers included for that customer.
- License identifiers and source URLs where available.

The SBOM must be regenerated after dependency, container, plugin, or lockfile
changes. Delivery must block if the SBOM is missing or does not match the
artifact.

The package includes `packaging/dobby-package/SBOM.example.spdx.json` only as a
scaffold. It is explicitly not a delivery SBOM. Customer delivery must replace
that file or ship a separate artifact-specific SBOM generated from the exact
delivered package.

## Attribution Pack

The commercial package should include:

- Hermes MIT license and notice.
- Third-party dependency notices generated from the SBOM.
- Model endpoint responsibility statement.
- Dobby weights not bundled statement.
- Honcho not bundled by default statement.
- Plugin/MCP notices for any customer-enabled optional tools.
- Known license exceptions or legal review notes.

The package includes `packaging/dobby-package/ATTRIBUTION.md` only as a
placeholder checklist. Delivery must replace placeholder text with notices
derived from the delivered SBOM and legal review.

## Commercial No-Go Claims

Do not claim:

- "Includes Dobby model weights."
- "Includes Honcho server."
- "No license obligations."
- "No SBOM needed."
- "The example SBOM or attribution placeholder is delivery-ready."
- "No customer model/provider terms apply."
- "All optional plugins are covered by the Hermes MIT license."
- "Deletion from the package deletes data from Discord or model providers."

## Publish Blockers

- Missing Hermes MIT notice in the customer artifact.
- Missing SBOM.
- Example SBOM or attribution placeholder is presented as customer-ready.
- Dobby weights or Honcho server artifacts appear in the default package.
- Sales copy implies bundled model rights or bundled Honcho.
- Optional plugin, MCP, dependency, or model license review is incomplete for a
  customer-specific delivery.
