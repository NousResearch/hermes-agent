# Plugin and Tool Sandboxing

Reader: implementation worker, deployment engineer, security reviewer.
Next action: configure optional tools so V1 remains allowlist-first and
fail-closed.

## Default Position

The default package should run without optional plugins or MCP servers. Discord,
signed webhooks, native memory, reminders, attachment review, status, and the
read-only repo helper are the only default surfaces.

Any plugin, MCP server, repo tool, browser tool, shell helper, or customer
integration must be enabled explicitly with a narrow allowlist.

## Allowlist-First Contract

Every enabled tool needs a reviewed entry that names:

- Tool or plugin name.
- Purpose and customer owner.
- Allowed command or API operation.
- Allowed arguments or schema.
- Allowed working directory.
- Allowed read paths.
- Allowed write paths, if any.
- Allowed environment variables.
- Allowed egress destinations.
- Required confirmation class.
- Log and redaction behavior.
- Expiration or review date.

Deny by default when an entry is missing, malformed, expired, or broader than
the requested operation.

## Container and Process Guidance

Preferred runtime isolation:

- Run tools as non-root.
- Use a fresh package home instead of a personal Hermes home.
- Drop Linux capabilities unless a tool has a reviewed need.
- Use read-only filesystem mounts where possible.
- Mount only the package home and explicitly approved repo paths.
- Use a temporary directory for scratch data.
- Set CPU, memory, process, file-size, and timeout limits.
- Keep secrets in the parent process or secret manager and pass only the
  minimum variables required by that tool.

Hard no:

- Do not mount the Docker socket.
- Do not run tool containers privileged.
- Do not mount the host root filesystem.
- Do not give a plugin broad access to `~/.hermes`, SSH keys, browser profiles,
  cloud credentials, shell history, or customer home directories.
- Do not use shell passthrough as a default integration mechanism.

## Egress Limits

Default egress should be restricted to the surfaces needed by the package:

- Discord API for the configured Discord app.
- The configured BYO model endpoint.
- Explicitly configured webhook sender callbacks, if needed.
- Package update or dependency sources only during controlled install/update
  procedures.

Plugin egress must be narrower than package egress. Each plugin should list its
allowed domains, ports, and protocols. Block wildcard egress unless the
customer accepts it as a documented exception.

## Confirmation Classes

Use confirmations for actions that can reveal data, spend money, mutate state,
or cross trust boundaries.

| Class | Examples | Required confirmation |
|---|---|---|
| Data reveal | Attachment content read, memory export, support bundle creation | Operator approval before content leaves quarantine or host |
| Local mutation | File write, memory delete, config change | Target path and action confirmation |
| Repo mutation | Patch apply, commit, branch delete | Out of V1 default; requires separate future approval model |
| Network mutation | Webhook callback, issue creation, remote API write | Out of V1 default unless explicitly allowlisted |
| Cost-bearing | Long model job, high-token research, bulk attachment processing | Budget or quota confirmation |
| External execution | Plugin/MCP tool invocation with host access | Tool-specific approval and audit |

The model must not be treated as the confirmer. Confirmation must come from the
authorized operator or a preapproved customer policy.

## Repo Helper Baseline

The V1 repo helper is read-only and propose-only by default. It may inspect
allowed repository paths and produce patch suggestions, but it must not:

- Write files.
- Stage or commit.
- Push or force-push.
- Create, edit, or merge PRs.
- Deploy.
- Delete branches.
- Run destructive git commands.
- Contact live remote hosts.

Any future write-capable repo helper is a separate commercial and security
decision, not a hidden V1 mode.

## MCP-Specific Rules

MCP servers are treated as external tool providers, even when they run locally.
Before enabling one:

- Review its binary/source origin and license.
- Pin the version or digest.
- Define allowed tools and schemas.
- Restrict filesystem roots.
- Restrict egress.
- Decide whether it may receive Discord content, attachment text, memory, or
  repo data.
- Confirm how it logs and deletes data.

Disable any MCP server that requires broad host access without a customer-owned
business reason and a documented exception.

## Audit and Redaction

Tool audit logs should record:

- Tool name and version.
- Operator, channel, and command ID where applicable.
- Input summary with secrets redacted.
- Allowed policy entry used.
- Confirmation result.
- Output summary with secrets redacted.
- Error class and retry decision.

Audit logs must not store raw secrets, full credentials, unapproved attachment
contents, or unrelated host paths.

## Publish Blockers

- Optional tools are enabled without allowlist entries.
- Normal operation requires root, privileged containers, Docker socket access,
  or broad host mounts.
- Plugin egress is unrestricted by default.
- Confirmation is delegated to model output instead of an authorized operator
  or customer policy.
- Tool logs contain unredacted secrets, attachment contents, or personal runtime
  paths.
