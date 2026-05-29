# Data Retention and Deletion

Reader: customer operator, support engineer, privacy reviewer.
Next action: decide what data the package may retain, how a customer exports
it, and what "delete" or "forget" means before onboarding.

## Scope

This policy covers customer-owned data handled by the default Dobby/Hermes
package profile:

- Discord commands and responses.
- Signed webhook payload summaries.
- Approved attachment metadata and contents.
- Native Hermes memory in the package `HERMES_HOME`.
- SessionDB entries used by `session_search`.
- Redacted logs, diagnostics, and support bundles.

Out of scope: Discord's own message retention, model provider logs, customer
SIEM retention, cloud backups outside the package host, and optional Honcho or
third-party plugin retention. Those systems need separate customer policies.

## Ownership Model

The customer owns:

- The Discord application, bot token, guild, channels, messages, and roles.
- The model endpoint, model API key, provider account, prompts, responses, and
  provider-side billing metadata.
- The host where the package runs.
- The selected package `HERMES_HOME`.
- Webhook sender systems and payloads.
- Any enabled plugin or MCP server configuration.

The package must not claim ownership of customer data. It provides local
controls for package-owned storage only.

## Storage Surfaces

| Surface | Default retention | Exported | Deleted by package delete | Notes |
|---|---:|---|---|---|
| Volatile command context | Session lifetime | No | Yes, by process stop | Not durable memory. |
| Discord command logs | Operator-configured | Yes, redacted | Yes, if under package home | Avoid message content unless needed. |
| Webhook payload summaries | Operator-configured | Yes, redacted | Yes, if under package home | Raw payload retention should be off unless explicitly enabled. |
| Attachment metadata | Operator-configured | Yes | Yes, if under package home | Metadata may exist before content approval. |
| Approved attachment contents | Operator-configured | Yes, redacted where possible | Yes, if under package home | Never read content before approval. |
| `SOUL.md` | Until edited or deleted | Yes | Yes, if package-owned | Operator-facing identity file. |
| `memories/MEMORY.md` and `memories/USER.md` | Until forgotten or deleted | Yes | Yes, if package-owned | Durable memory requires consent. |
| SessionDB and `session_search` data | Until forgotten or deleted | Yes | Yes, if package-owned | Scope to selected package home. |
| Support bundles | Short-lived by support procedure | Yes | Manual and package cleanup | Must be redacted before sharing. |

## Retention Defaults

V1 should use conservative defaults:

- Fresh package `HERMES_HOME`; no import from personal `~/.hermes`.
- Durable memory writes disabled until operator consent is enabled.
- No raw attachment content stored unless explicitly approved.
- No raw webhook payload retention unless the operator enables it for a route.
- Logs and diagnostics redacted before display or archive.
- Demo and verification fixtures synthetic only.

Implementation slices must turn these defaults into exact config keys and tests.
Until then, sales and onboarding must describe retention as package-local and
customer-configured, not as a managed cloud retention service.

## Export Model

Export must include only package-owned data under the selected `HERMES_HOME` and
the package's configured log/support surfaces. A complete export should include:

- Memory status and consent state.
- `SOUL.md`, package memory files, and SessionDB-derived records.
- Redacted command, webhook, reminder, attachment, and repo-helper audit logs.
- Configuration summary with secret values redacted.
- SBOM and package version metadata when available.

Export must not include:

- Discord bot tokens, model API keys, webhook secrets, plugin credentials, or
  MCP credentials.
- Personal runtime state from another Hermes installation.
- Host files outside the package's allowed paths.
- Raw provider-side logs that the package did not create.

## Forget Model

"Forget" means targeted removal from package-owned memory and search surfaces.
It is the right control when the customer wants one topic, item, or identifier
removed without deleting the whole package home.

Required behavior:

- Show the candidate memory item or search match before deletion.
- Require explicit confirmation.
- Remove matching entries from package memory files and SessionDB-derived search
  data where the implementation owns those records.
- Write a redacted audit entry that records the action, not the forgotten
  secret or sensitive content.

Do not claim:

- Forget deletes Discord messages.
- Forget deletes model provider logs.
- Forget removes knowledge already present in a model's trained weights.
- Forget reaches optional third-party memory systems unless that integration
  has its own verified deletion path.

## Delete Model

"Delete" means full deletion of package-owned local data for the selected
package home. It should be explicit, scoped, and reversible only through
customer backups.

Required behavior:

- Display the target package home and data categories.
- Require an explicit confirmation token.
- Stop package services before destructive local deletion.
- Delete package memory files, SessionDB records, package-owned logs, support
  bundles, replay caches, local attachment stores, and package config snapshots
  if they are under the selected package home.
- Preserve customer data by default when rollback is requested; delete only
  when deletion is the requested operation.

Deletion must not:

- Traverse outside the selected package home.
- Delete unrelated Hermes installations.
- Delete Discord server history.
- Delete model provider records.
- Delete customer backups or SIEM archives outside the package host.

## Support Bundle Redaction

Support bundles exist to diagnose package behavior without copying secrets or
personal runtime data. They must be redacted before leaving the customer host.

Required redactions:

- Discord bot token, model API key, webhook secrets, plugin credentials, MCP
  credentials, session tokens, cookies, private URLs, and authorization headers.
- Values from known secret environment variable names.
- Raw attachment contents unless the customer explicitly includes a redacted
  sample.
- Personal paths outside the package home.
- Full webhook payloads unless route policy permits redacted payload export.

Support bundle contents should prefer:

- Package version and git metadata.
- Redacted config summary.
- Health and preflight results.
- Recent redacted error logs.
- SBOM or dependency summary.
- Reproduction steps using synthetic data.

## Customer-Facing Boundary

The package can offer local export, forget, and delete for package-owned data.
It cannot independently erase data controlled by Discord, the model provider,
webhook senders, optional plugins, customer backups, or customer monitoring
systems. Customer onboarding must state this boundary before production use.

## Publish Blockers

- Durable memory writes happen before consent.
- Export includes secrets, personal runtime state, or host files outside the
  package home.
- Forget/delete can target paths outside package-owned storage.
- Support bundles leave the host without redaction.
- Sales copy implies deletion from Discord, model providers, or optional
  third-party services without a verified integration-specific deletion path.
