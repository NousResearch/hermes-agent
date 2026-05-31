# QA Gates — SMB Agent Business Core

## Global gates

- Documentation gate: PRD, ADRs, sprint plan and task graph exist before implementation.
- Architecture gate: each module has clear schema ownership and adapter boundary.
- Test gate: every tool handler has unit tests for success/failure.
- Security gate: secrets only from Infisical/runtime env; no hardcoded keys.
- Data gate: operational records use Agent Core SQL, not memory/chat history.
- Adapter gate: no direct writes to external DBs.
- UX gate: user-facing flows remain conversational and do not require UI unless explicitly configured.

## Commercial/Sales Core QA

- Product creation/update handles duplicate SKU/name.
- Quote totals are deterministic.
- Order conversion preserves quote items.
- Invoice status transitions are valid.
- Payment link is adapter-backed or clearly marked unavailable.
- No invoice is labeled fiscal unless fiscal adapter is configured.

## Accounting Lite QA

- Expense/income totals match report totals.
- Categories are stable and editable.
- Receipts can be referenced without storing secrets.
- Monthly report exports all required summaries.
- Fiscal/accounting limitations are visible in docs and tool descriptions.

## Marketing Core QA

- Brand profile is required before campaign automation.
- Drafts and published assets have separate states.
- Approval is required before external publication unless policy explicitly allows auto-publish.
- External publishing failure does not lose local draft/campaign state.
- Generated images/videos keep provenance and prompt metadata.

## Security / compliance checks

- Payment adapters never log full credentials.
- Social/email OAuth secrets stay in Infisical.
- User approval required for first publication to each external channel.
- PII in contacts/customers is scoped to single tenant.
- Export reports omit secrets and include only business data.
