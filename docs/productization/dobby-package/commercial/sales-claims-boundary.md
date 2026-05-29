# Sales Claims Boundary

Reader: sales, marketing, solutions, support, implementation owner.
Next action: keep V1 promises aligned with the package's verified boundaries.

## Rule

Only claim what the default package can deliver through local setup, mocked or
dry-run verification, and customer-owned credentials. Anything needing managed
hosting, bundled model rights, write-capable automation, broad integrations, or
legal review is premium, experimental, or no-go.

## Honest V1 Claims

These claims are allowed once their evidence gates are implemented and passing:

| Claim | Required boundary |
|---|---|
| Self-hosted Discord-first operator package for Hermes Agent. | Customer runs it on their host and brings the Discord app. |
| BYO model endpoint. | Customer provides endpoint, API key, billing, and provider terms. |
| Discord command center for operator workflows. | Allowed channels/users configured; no broad integrations by default. |
| Health, quota, and status reporting. | Output is redacted; unavailable provider quota is handled honestly. |
| Signed webhook inbox. | HMAC, timestamp, replay, size, and route gates pass before prompt use. |
| Explicit attachment review. | Metadata is shown before content is read or summarized. |
| Read-only/propose-only repo helper. | No write, commit, push, deploy, or destructive git by default. |
| Native Hermes memory controls. | Consent, status, export, forget, and delete work inside package-owned storage. |
| Reminders and cron delivery through Discord. | Delivery target is customer-configured and logs are redacted. |
| Synthetic demo kit and staging verification. | Demo data is synthetic and does not require live secrets. |
| Redacted support bundle. | Bundle excludes secrets and personal runtime state. |
| SBOM-backed delivery. | SBOM matches the delivered artifact. |

Before evidence exists, phrase these as "planned V1 package requirements" or
"implementation gate", not as shipped capabilities.

## Premium or Custom Claims

These may be sold only as scoped add-ons after separate design, security, legal,
and delivery review:

- Managed hosting or vendor-operated infrastructure.
- Customer-specific plugin or MCP integrations.
- Additional messaging channels beyond Discord.
- Write-capable repo automation.
- Enterprise SSO, SCIM, DLP, SIEM, or audit-log export.
- Dedicated compliance evidence packs.
- Custom model routing, fallback, or budget policy.
- Optional Honcho deployment or migration support.

Premium does not mean safe by default. Each item needs its own threat model,
license review, rollback path, and evidence gate.

## Experimental Claims

These can be shown in demos only when clearly labeled experimental:

- Research scout output quality beyond fixture-backed verification.
- Autonomous multi-step investigation across live web sources.
- Attachment understanding for new file types.
- Plugin/MCP tools that have not completed sandbox review.
- Memory ranking, personality, or long-horizon recall quality.
- Model-specific optimization or cost prediction.

Do not present experimental features as guaranteed production behavior.

## No-Go Claims

Never claim:

- The package bundles Dobby weights.
- The package grants model usage rights.
- The package bundles or requires Honcho server.
- The package deletes data from Discord, model providers, customer backups, or
  optional third-party services.
- The agent can safely commit, push, deploy, or mutate production by default.
- The agent can autonomously purchase, trade, email, post, merge, or deploy
  without operator approval.
- The default package includes broad production GitHub, Notion, Google, or
  Slack OAuth access.
- Browser automation is enabled by default.
- The package is fully autonomous without operator approvals.
- The package is compliant with a named legal or security framework without a
  completed audit.
- The package has zero data retention.
- The package can ingest all company data safely by default.
- Secrets never leave the customer environment if the customer configures a
  third-party model endpoint.
- Optional plugins inherit Hermes MIT license coverage automatically.

## Evidence Language

Use precise evidence language:

- "Verified by mock/golden tests" when tests exist and pass.
- "Verified by staging test" when the customer ran staging with their own
  credentials.
- "Local package control" for export, forget, delete, and rollback inside the
  selected package home.
- "Customer/provider controlled" for Discord retention, model provider logs,
  billing, and external backups.
- "Requires legal review" for Honcho, model rights, customer-specific plugin
  distribution, and compliance claims.

Avoid broad language:

- "Secure by default" without naming defaults.
- "Private" without stating provider and Discord data flows.
- "Fully managed" for self-hosted V1.
- "Enterprise-ready" without support, compliance, and operational scope.

## Publish Blockers

- Website, deck, proposal, or onboarding copy says V1 includes model weights,
  bundled Honcho, managed hosting, write-capable repo actions, or guaranteed
  compliance.
- A shipped claim lacks an evidence gate in the traceability table.
- Sales copy hides customer responsibilities for Discord app, model endpoint,
  privacy policy, staging test, and rollback.
