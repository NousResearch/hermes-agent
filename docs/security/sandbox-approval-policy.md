# Sandbox and Approval Policy

This document is a contract for Hermes Agent's current sandbox and approval
posture. It documents existing behavior only. This patch makes no runtime behavior changes and should be safe to review independently of terminal backend code.

## Threat Model

Hermes Agent is a single-tenant personal agent. Its strongest supported boundary
against adversarial model output is **OS-level isolation**. In-process controls
are useful accident-prevention layers, but they are not containment.

The two supported OS-level isolation postures are:

1. **Terminal-backend isolation** — the `terminal`, file, and patch tools run in
   the configured terminal backend. Non-local backends can be Docker,
   Singularity, Modal, Daytona, or SSH.
2. **Whole-process wrapping** — the entire Hermes process tree runs inside a
   container or external sandbox. This is required when the operator expects
   code execution, MCP subprocesses, plugin loading, hook dispatch, and skill
   loading to be confined by the same boundary.

## Approval Gate Policy

The approval gate scans shell strings for common destructive operations and
prompts or denies according to the operator's `config.yaml` policy. The
**approval gate is a heuristic**, not a security boundary.

The default policy is conservative:

```yaml
approvals:
  mode: manual
  timeout: 60
  cron_mode: deny
  mcp_reload_confirm: true
  destructive_slash_confirm: true
command_allowlist: []
```

Policy meanings:

- `approvals.mode: manual` asks the operator for dangerous commands.
- `approvals.mode: smart` may auto-approve lower-risk matches but still keeps
  hardline patterns interactive.
- `approvals.mode: off` disables normal prompts, but hardline safeguards may
  still block or ask for confirmation where the code path enforces that floor.
- `approvals.cron_mode: deny` blocks dangerous commands in unattended cron
  runs instead of waiting for an absent operator.
- `command_allowlist` stores patterns the operator deliberately chose to allow
  permanently.

Hardline patterns are reserved for operations that can destroy broad state or
shut down the host, such as recursive deletion of root filesystems, disk
formatting, system shutdown, forced fork bombs, and writes to Hermes security
configuration.

## Terminal Backend Isolation

The terminal backend is selected in `config.yaml`:

```yaml
terminal:
  backend: local  # local | docker | ssh | modal | daytona | singularity
```

Important properties:

- `local` has no isolation. Commands run with the operator user's host access.
- `docker`, `singularity`, `modal`, and `daytona` run commands in a configured
  sandbox target.
- `ssh` moves command execution to a remote host and relies on that host's OS
  boundary.
- Terminal-backend isolation confines shell/file-tool behavior only. It does
  not confine code paths running inside the Hermes Python process.

Docker's launch-directory mount is opt-in:

```yaml
terminal:
  docker_mount_cwd_to_workspace: false
```

Keep `docker_mount_cwd_to_workspace` disabled unless the sandbox intentionally
needs access to the launch directory. Any `docker_extra_args` that add host
mounts, host networking, privileged mode, or extra capabilities weaken the
sandbox and should be reviewed as operator break-glass choices.

## Whole-Process Wrapping

Use whole-process wrapping when the input surface is untrusted or shared:

- open web ingestion,
- inbound email,
- multi-user messaging channels,
- untrusted MCP servers,
- third-party plugins or skills that have not been reviewed line by line.

Whole-process wrapping must include the Python interpreter and subprocess tree,
not only the terminal backend. Otherwise, code execution, MCP subprocesses,
plugins, hooks, and skill loading remain inside the host trust envelope.

## Path and File Safety Helpers

Path validation helpers such as `tools.path_security.validate_within_dir()` are
in-process guards. They reject traversal and resolved-path escapes in tool
implementations that accept user-controlled file paths. They are important, but
they are still not OS-level containment.

## Operator Review Requirements

Before enabling a weaker posture or installing third-party code, review:

1. `config.yaml` terminal backend and approval settings.
2. `command_allowlist` entries.
3. Docker volume mounts, `docker_extra_args`, and forwarded environment vars.
4. Plugin Python code and dependencies.
5. Skill `SKILL.md`, scripts, templates, and referenced Python code.
6. Whether the deployment should use terminal-backend isolation or
   whole-process wrapping.

## Limitations and Out of Scope

Bypasses of the approval gate, redaction, skills scanning, or other in-process
heuristics are hardening bugs, not isolation escapes. They should be fixed when
useful, but they do not cross the project's declared security boundary unless
they combine with a documented OS-level isolation escape or unauthorized
external-surface access.

## Related Documents

- [`SECURITY.md`](../../SECURITY.md) — full project trust model and disclosure
  scope.
- [`docs/security/network-egress-isolation.md`](network-egress-isolation.md) —
  Docker network egress isolation pattern.
- [`website/docs/user-guide/configuration.md`](../../website/docs/user-guide/configuration.md)
  — terminal backend configuration reference.
