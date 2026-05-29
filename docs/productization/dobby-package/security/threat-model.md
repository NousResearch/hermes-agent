# Threat Model

Reader: security reviewer, implementation worker, customer operator.
Next action: use this as the publish gate for V1 package defaults and customer
deployment review.

## Scope

This model covers the default Dobby/Hermes package profile:

- Discord command center.
- Signed webhook inbox.
- BYO model endpoint through Hermes provider configuration.
- Native Hermes memory inside the selected package `HERMES_HOME`.
- Explicit attachment review.
- Read-only/propose-only repo helper.
- Optional plugins and MCP servers only when explicitly enabled.

Out of scope for the default package: bundled Dobby weights, bundled or
auto-run Honcho server, broad messaging integrations, write-capable repo
automation, live deploys, live remote host mutation, and copied personal
runtime state.

Source of truth: the Stage 3 package README, risk register, traceability table,
and ADRs under `docs/productization/dobby-package/`.

## Assets

Protect these assets first:

- Discord bot token, application credentials, guild/channel IDs, user IDs, and
  role membership.
- Discord messages, mentions, command arguments, attachment metadata, and any
  approved attachment contents.
- Model endpoint URL, model API key, provider quota and billing metadata.
- Webhook HMAC secrets, signatures, timestamps, replay caches, and payloads.
- Package `HERMES_HOME`, including `SOUL.md`, `memories/MEMORY.md`,
  `memories/USER.md`, SessionDB, logs, exports, and support bundles.
- Repo helper inputs, repository paths, diffs, branch names, and generated patch
  proposals.
- Plugin and MCP configuration, tool credentials, local filesystem mounts, and
  network egress permissions.
- SBOM, lockfiles, package artifacts, and third-party attribution records.

## Trust Boundaries

| Boundary | Trusted side | Untrusted side | Required gate |
|---|---|---|---|
| Discord gateway | Package command router | Discord users, channels, messages, attachments | Allowed user/channel checks, mention limits, redaction |
| Model endpoint | Hermes request builder | Model provider response and billing surface | BYO endpoint configuration, timeout and quota handling, no secret logging |
| Webhook inbox | HMAC-verified route handler | External senders and payloads | Per-route secret, timestamp, replay, size, schema, and route allowlist |
| Attachment review | Metadata quarantine | Attachment bytes and extracted text | Explicit approval before content read or summarization |
| Repo helper | Read-only inspection and patch proposal | Shell, git, filesystem, remote hosts | Allowlist, no write/commit/push/deploy by default |
| Memory | Package-owned `HERMES_HOME` | Existing personal runtime state and user-provided text | Fresh home, consent, export, forget, delete |
| Plugins/MCP | Disabled default package | Plugin code, external tools, host resources | Allowlist-first sandbox and confirmation gates |
| Supply chain | Reviewed source and SBOM | Dependencies, containers, plugins, model clients | Lockfile review, license review, artifact scan |

## Threats and Required Controls

### Discord and API Data

Threats:

- A user outside the customer-approved audience issues commands.
- A command runs in the wrong channel or leaks output through mentions.
- Discord message content, IDs, or attachments are copied into logs or support
  bundles without redaction.
- Status or quota checks expose model keys, internal URLs, or billing details.

Controls:

- Require an allowlist of Discord guilds, channels, and operator users or roles.
- Keep allowed mentions restrictive by default.
- Redact command diagnostics, logs, support bundles, and health output.
- Treat Discord content as customer-owned data stored only in the selected
  package home unless the operator explicitly exports it.

Publish blockers:

- No preflight check for Discord channel/user scope.
- No redaction scan for diagnostics or support bundles.
- Any example contains usable-looking credentials instead of placeholders.

### Secrets

Threats:

- Bot tokens, model keys, webhook secrets, plugin credentials, or MCP secrets
  appear in examples, logs, exports, support bundles, or test fixtures.
- A weak or placeholder webhook secret is accepted.
- Existing personal `~/.hermes` data is copied into the sellable package.

Controls:

- Examples must use angle-bracket placeholders only.
- Preflight must fail on placeholders, weak webhook secrets, unsafe
  `HERMES_HOME`, missing Discord credentials, and missing model endpoint.
- Support bundles and logs must use redaction before display or archive.
- Never read or copy existing personal runtime state during package setup.

Publish blockers:

- Any secret-shaped token in package docs, examples, fixtures, or generated
  bundles.
- Any setup path that imports `~/.hermes`, session history, logs, or secrets by
  default.

### Prompt Injection

Threats:

- Discord messages, webhook payloads, attachments, memory entries, or repo files
  instruct the agent to reveal secrets, bypass policy, mutate repos, call tools,
  or ignore operator boundaries.
- Model output is treated as authorization for tool execution.

Controls:

- Treat all user, webhook, attachment, memory, and repo text as untrusted input.
- Keep authorization in the command router and policy gates, not in the prompt.
- Require explicit confirmation for sensitive actions and never accept a model
  response as the confirmation token.
- Strip or compartmentalize untrusted content before inserting it into prompts.

Publish blockers:

- A tool executes because model text asked for it without policy-gate approval.
- Prompt templates include secrets or privileged operational instructions that
  untrusted content can override.

### Attachments

Threats:

- Attachment contents are read before the operator reviews metadata.
- Large or malformed files cause resource exhaustion.
- Extracted attachment text contains prompt injection or secrets.

Controls:

- Show metadata first: filename, size, type, source, and scan status.
- Require explicit approval before downloading, extracting, summarizing, or
  sending attachment contents to the model endpoint.
- Enforce size, type, timeout, and storage limits.
- Redact extracted text before logs, support bundles, and previews.

Publish blockers:

- Any default path reads or summarizes attachment bytes before approval.
- No tests for denied, expired, oversized, and approved attachment flows.

### Repo Tools

Threats:

- Repo helper writes files, commits, pushes, deploys, removes branches, or runs
  destructive git commands by default.
- Shell passthrough exposes secrets or host filesystem data.
- A patch proposal includes private data from outside the selected repo.

Controls:

- Default mode is read-only inspection and propose-only patch output.
- Allowlist safe inspection commands and block write, commit, push, deploy,
  remote mutation, and destructive commands.
- Keep repository path scope explicit.
- Require separate customer approval before any future write-capable mode.

Publish blockers:

- Any V1 sales or setup claim implies autonomous commit, push, deploy, or live
  remote mutation.
- No tests proving write and destructive commands are blocked by default.

### Webhooks

Threats:

- Unsigned, stale, replayed, oversized, or wrong-route payloads enter the agent.
- A sender reuses a secret across tenants or routes.
- Payload summaries leak sensitive fields to Discord or logs.

Controls:

- Require HMAC per configured route.
- Verify timestamp freshness, replay cache, body size, content type, and route
  allowlist before prompt construction.
- Redact payload summaries before Discord delivery and logs.
- Keep insecure bypass out of the quickstart.

Publish blockers:

- Any unsigned webhook mode is enabled by default.
- Missing negative tests for bad signature, replay, stale timestamp, and
  oversized payload.

### Memory

Threats:

- Durable memory writes happen before consent.
- Export, forget, or delete reaches outside package-owned `HERMES_HOME`.
- "Forget" is misrepresented as deletion from Discord or the model provider.
- Existing personal memory is imported into the package without explicit action.

Controls:

- Use a fresh package `HERMES_HOME` by default.
- Keep durable memory off until operator consent is enabled.
- Provide status, consent, export, targeted forget, and full delete controls.
- State that package deletion does not delete Discord history or provider logs.

Publish blockers:

- No temp-home tests for consent off, consent on, export, forget, and delete.
- Any default quickstart copies personal memory or session state.

### Plugins and MCP

Threats:

- Optional plugins or MCP servers receive broader filesystem, secret, or network
  access than the customer intended.
- A plugin bypasses package policy gates through host mounts, Docker socket
  access, or unrestricted egress.

Controls:

- Disable optional plugins and MCP servers by default.
- Require allowlisted tool names, commands, arguments, environment variables,
  mount paths, and egress destinations.
- Run non-root in an isolated container or equivalent sandbox.
- Never mount the Docker socket into plugin runtimes.
- Require confirmation for sensitive tool classes.

Publish blockers:

- Any plugin is enabled by default without explicit allowlist review.
- Any sandbox needs host root, Docker socket access, or broad host filesystem
  mounts for normal operation.

### Supply Chain

Threats:

- Model weights, Honcho server artifacts, broad integrations, or unrelated
  generated artifacts are bundled into the package.
- Dependencies include unreviewed licenses or vulnerable packages.
- Artifacts drift from source, lockfiles, or attribution records.

Controls:

- Generate an SBOM for every customer-delivered artifact.
- Scan package artifacts for model weight files, personal paths, generated
  secrets, and unexpected binaries.
- Keep Dobby weights and Honcho server out of the default package.
- Review dependency licenses and maintain third-party notices.

Publish blockers:

- No SBOM for the delivered artifact.
- Any artifact contains Dobby weights, Honcho server bundle, personal runtime
  data, or unreviewed dependency license exposure.

## Residual Risk

- Legal and license review is still required before commercial delivery.
- Exact preflight, redaction, sandbox, and SBOM commands are not defined in this
  document; implementation slices must add executable checks.
- BYO model providers may retain prompts, responses, logs, or billing metadata
  under customer/provider terms outside this package.
