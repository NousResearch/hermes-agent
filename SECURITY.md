# Hermes Agent Security Policy

This document describes Hermes Agent's trust model, names the one
security boundary the project treats as load-bearing, and defines the
scope for vulnerability reports.

## 1. Reporting a Vulnerability

Report privately via [GitHub Security Advisories](https://github.com/NousResearch/hermes-agent/security/advisories/new)
or **security@nousresearch.com**. Do not open public issues for
security vulnerabilities. **Hermes Agent does not operate a bug
bounty program.**

A useful report includes:

- A concise description and severity assessment.
- The affected component, identified by file path and line range
  (e.g. `path/to/file.py:120-145`).
- Environment details (`hermes version`, commit SHA, OS, Python
  version).
- A reproduction against `main` or the latest release.
- A statement of which trust boundary in §2 is crossed.

Please read §2 and §3 before submitting. Reports that demonstrate
limits of an in-process heuristic this policy does not treat as a
boundary will be closed as out-of-scope under §3 — but see §3.2:
they are still welcome as regular issues or pull requests, just not
through the private security channel.

---

## 2. Trust Model

Hermes Agent is a single-tenant personal agent. Its posture is
layered, and the layers are not equally load-bearing. Reporters and
operators should reason about them in the same terms.

### 2.1 Definitions

- **Agent process.** The Python interpreter running Hermes Agent,
  including any Python modules it has loaded (skills, plugins,
  hook handlers).
- **Terminal backend.** A pluggable execution target for the
  `terminal()` tool. The default runs commands directly on the host.
  Other backends run commands inside a container, cloud sandbox, or
  remote host.
- **Input surface.** Any channel through which content enters the
  agent's context: operator input, web fetches, email, gateway
  messages, file reads, MCP server responses, tool results.
- **Trust envelope.** The set of resources an operator has implicitly
  granted Hermes Agent access to by running it — typically, whatever
  the operator's own user account can reach on the host.
- **Stance.** An explicit statement in Hermes Agent's documentation
  or code about how a consuming layer (adapter, UI, file writer,
  shell) should treat agent output — e.g. "the dashboard renders
  agent output as inert HTML."

### 2.2 The Boundary: OS-Level Isolation

**The only security boundary against an adversarial LLM is the
operating system.** Nothing inside the agent process constitutes
containment — not the approval gate, not output redaction, not any
pattern scanner, not any tool allowlist. Any in-process component
that screens LLM output is a heuristic operating on an
attacker-influenced string, and this policy treats it as such.

Hermes Agent supports two OS-level isolation postures. They address
different threats and an operator should choose deliberately.

#### Terminal-backend isolation

A non-default terminal backend runs LLM-emitted shell commands
inside a container, remote host, or cloud sandbox. The file tools
(`read_file`, `write_file`, `patch`) also run through this backend,
since they are implemented on top of the shell contract — they
cannot reach paths the backend doesn't expose.

What this confines: anything the agent does by issuing shell or
file operations. What this does **not** confine: everything the
agent does in its own Python process. That includes the
code-execution tool (spawned as a host subprocess), MCP subprocesses
(spawned from the agent's environment), plugin loading, hook
dispatch, and skill loading (all imported into the agent
interpreter).

Terminal-backend isolation is the right posture when the concern is
LLM-emitted destructive shell or unwanted file-tool writes, and the
operator is otherwise trusted.

#### Whole-process wrapping

Whole-process wrapping runs the entire agent process tree inside a
sandbox. Every code path — shell, code-execution, MCP, file tools,
plugins, hooks, skill loading — is subject to the same filesystem,
network, process, and (where applicable) inference policy.

Hermes Agent supports this in two ways:

- **Hermes Agent's own Docker image and Compose setup.** Lighter-
  weight; the agent runs in a standard container with operator-
  configured mounts and network policy.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)**.
  OpenShell provides per-session sandboxes with declarative policy
  across filesystem, network (L7 egress), process/syscall, and
  inference-routing layers. Network and inference policies are
  hot-reloadable. Credentials are injected from a Provider store
  and never touch the sandbox filesystem.

Under a whole-process wrapper, Hermes Agent's in-process heuristics
(§2.4) function as accident-prevention layered on top of a real
boundary. This is the supported posture when the agent ingests
content from surfaces the operator does not control — the open web,
inbound email, multi-user channels, untrusted MCP servers — and for
production or shared deployments.

Operators running the default local backend with untrusted input
surfaces, or running a terminal-backend sandbox and expecting it to
contain code paths that don't go through the shell, are operating
outside the supported security posture.

### 2.3 Credential Scoping

Hermes Agent filters the environment it passes to its lower-trust
in-process components: shell subprocesses, MCP subprocesses, and
the code-execution child. Credentials like provider API keys and
gateway tokens are stripped by default; variables explicitly
declared by the operator or by a loaded skill are passed through.

This reduces casual exfiltration. It is not containment. Any
component running inside the agent process (skills, plugins, hook
handlers) can read whatever the agent itself can read, including
in-memory credentials. The mitigation against a compromised
in-process component is operator review before install (§2.4,
§2.5), not environment scrubbing.

### 2.4 In-Process Heuristics

The following components screen or warn about LLM behavior. They
are useful. They are not boundaries.

- The **approval gate** detects common destructive shell patterns
  and prompts the operator before execution. Shell is Turing-
  complete; a denylist over shell strings is structurally
  incomplete. The gate catches cooperative-mode mistakes, not
  adversarial output.
- **Output redaction** strips secret-like patterns from display.
  A motivated output producer will defeat it.
- **Skills Guard** scans installable skill content for injection
  patterns. It is a review aid; the boundary for third-party skills
  is operator review before install. Reviewing a skill means
  reading its Python code and scripts, not just its SKILL.md
  description — skills execute arbitrary Python at import time.

### 2.5 Plugin Trust Model

Plugins load into the agent process and run with full agent
privileges: they can read the same credentials, call the same
tools, register the same hooks, and import the same modules as
anything shipped in-tree. The boundary for third-party plugins is
operator review before install — the same rule as skills (§2.4),
called out separately because plugins are architecturally heavier
and often ship their own background services, network listeners,
and dependencies.

A malicious or buggy plugin is not a vulnerability in Hermes Agent
itself. Bugs in Hermes Agent's plugin-install or plugin-discovery
path that prevent the operator from seeing what they're installing
are in scope under §3.1.

### 2.6 External Surfaces

An **external surface** is any channel outside the local agent
process through which a caller can dispatch agent work, resolve
approvals, or receive agent output. Each surface has its own
authorization model, but the rules below apply uniformly.

**Surfaces in Hermes Agent:**

- **Gateway platform adapters.** Messaging integrations in
  `gateway/platforms/` (Telegram, Discord, Slack, email, SMS, etc.)
  and analogous adapters shipped as plugins.
- **Network-exposed HTTP surfaces.** The API server adapter, the
  dashboard plugin, the kanban plugin's HTTP endpoints, and any
  other plugin that binds a listening socket.
- **Editor / IDE adapters.** The ACP adapter (`acp_adapter/`) and
  equivalent integrations that accept requests from a local client
  process.
- **The TUI gateway (`tui_gateway/`).** JSON-RPC backend for the
  Ink terminal UI, reached over local IPC.

**Uniform rules:**

1. **Authorization is required at every surface that crosses a
   trust boundary.** For messaging and network HTTP surfaces, the
   boundary is the network: authorization means an operator-
   configured caller allowlist. For editor and local-IPC surfaces
   (ACP, TUI gateway), the boundary is the host's user account:
   authorization means relying on OS-level access control (file
   permissions, loopback-only binds) and not exposing the surface
   beyond the local user without an explicit network auth layer.
2. **An allowlist is required for every enabled network-exposed
   adapter.** Adapters must refuse to dispatch agent work, resolve
   approvals, or relay output until an allowlist is set. Code paths
   that fail open when no allowlist is configured are code bugs in
   scope under §3.1.
3. **Session identifiers are routing handles, not authorization
   boundaries.** Knowing another caller's session ID does not grant
   access to their approvals or output; authorization is always
   re-checked against the allowlist (or OS-level equivalent).
4. **Within the authorized set, all callers are equally trusted.**
   Hermes Agent does not model per-caller capabilities inside a
   single adapter. Operators who need capability separation should
   run separate agent instances with separate allowlists.
5. **Binding a local-only surface to a non-loopback interface is a
   break-glass operator decision (§3.2).** The dashboard and other
   plugin HTTP servers default to loopback; exposing them via
   `--host 0.0.0.0` or equivalent makes public-exposure hardening
   (§4) the operator's responsibility.

---

## 3. Scope

### 3.1 In Scope

- Escape from a declared OS-level isolation posture (§2.2): an
  attacker-controlled code path reaching state that the posture
  claimed to confine.
- Unauthorized external-surface access: a caller outside the
  configured authorization set (allowlist, or OS-level equivalent
  for local-IPC surfaces) dispatching work, receiving output, or
  resolving approvals (§2.6).
- Credential exfiltration: leakage of operator credentials or
  session authorization material to a destination outside the
  trust envelope, via a mechanism that should have prevented it
  (environment scrubbing bug, adapter logging, transport error
  that flushes credentials to an upstream, etc.).
- Trust-model documentation violations: code behaving contrary to
  what this policy, Hermes Agent's own documentation, or reasonable
  operator expectations would predict — including cases where
  Hermes Agent has documented a stance about how its output should
  be rendered by a consuming layer (dashboard, gateway adapter,
  file writer, shell) and a code path breaks that stance.

### 3.2 Out of Scope

"Out of scope" here means "not a security vulnerability under this
policy." It does not mean "not worth reporting." Improvements to the
in-process heuristics, hardening ideas, and UX fixes are welcome as
regular issues or pull requests — the approval gate can always catch
more patterns, redaction can always get smarter, adapter behavior
can always be tightened. These items just don't go through the
private-disclosure channel and don't receive advisories.

- **Bypasses of in-process heuristics (§2.4)** — approval-gate regex
  bypasses, redaction bypasses, Skills Guard pattern bypasses, and
  analogous reports against future heuristics. These components are
  not boundaries; defeating them is not a vulnerability under this
  policy.
- **Prompt injection per se.** Getting the LLM to emit unusual
  output — via injected content, hallucination, training artifacts,
  or any other cause — is not itself a vulnerability. "I achieved
  prompt injection" without a chained §3.1 outcome is not an
  actionable report under this policy.
- **Consequences of a chosen isolation posture.** Reports that a
  code path operating within its posture's scope can do what that
  posture permits are not vulnerabilities. Examples: shell or file
  tools reaching host state under the local backend; code-execution
  or MCP subprocesses reaching host state under terminal-backend
  isolation that only sandboxes shell; reports whose preconditions
  require pre-existing write access to operator-owned configuration
  or credential files (those are already inside the trust envelope).
- **Documented break-glass settings.** Operator-selected trade-offs
  that explicitly disable protections: `--insecure` and equivalent
  flags on the dashboard or other components, disabled approvals,
  local backend in production, development profiles that bypass
  hermes-home security, and similar. Reports against those
  configurations are not vulnerabilities — that's the flag's job.
- **Community-contributed skills and plugins.** Third-party skills
  (including the community skills repository) and third-party
  plugins are in the operator's review surface, not Hermes Agent's
  trust surface (§2.4, §2.5). A skill or plugin doing something
  malicious is the expected failure mode of one that wasn't
  reviewed, not a vulnerability in Hermes Agent. Bugs in Hermes
  Agent's skill-install or plugin-install path that prevent the
  operator from seeing what they're installing are in scope under
  §3.1.
- **Public exposure without external controls.** Exposing the
  gateway or API to the public internet without authentication,
  VPN, or firewall.
- **Tool-level read/write restrictions on a posture where shell is
  permitted.** If a path is reachable via the terminal tool, reports
  that other file tools can reach it add nothing.

---

## 4. Deployment Hardening

The single most important hardening decision is matching isolation
(§2.2) to the trust of the content the agent will ingest. Beyond
that:

- Run the agent as a non-root user. The supplied container image
  does this by default.
- Keep credentials in the operator credential file with tight
  permissions, never in the main config, never in version control.
  Under OpenShell, use the Provider store rather than an on-disk
  credential file.
- Do not expose the gateway or API to the public internet without
  VPN, Tailscale, or firewall protection. Under OpenShell, use the
  network policy layer to restrict egress.
- Configure a caller allowlist for every network-exposed adapter
  you enable (§2.6).
- Review third-party skills and plugins before install (§2.4,
  §2.5). For skills, this means reading the Python and scripts,
  not just SKILL.md. Skills Guard reports and the install audit
  log are the review surface.
- Hermes Agent includes supply-chain guards for MCP server
  launches and for dependency / bundled-package changes in CI; see
  `CONTRIBUTING.md` for specifics.

---

## 5. Disclosure

- **Coordinated disclosure window:** 90 days from report, or until a
  fix is released, whichever comes first.
- **Channel:** the GHSA thread or email correspondence with
  security@nousresearch.com.
- **Credit:** reporters are credited in release notes unless
  anonymity is requested.
