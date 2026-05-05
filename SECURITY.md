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

Hermes is a single-tenant personal agent. Its posture is layered, and
the layers are not equally load-bearing. Reporters and operators
should reason about them in the same terms.

### 2.1 The Boundary: OS-Level Isolation

**The only security boundary against an adversarial LLM is the
operating system.** Nothing inside the agent process constitutes
containment — not the approval gate, not output redaction, not any
pattern scanner, not any tool allowlist. Any in-process component
that screens LLM output is a heuristic operating on an
attacker-influenced string, and this policy treats it as such.

Hermes supports two OS-level isolation postures. They address
different threats and an operator should choose deliberately.

**Terminal-backend isolation** sandboxes the shell tool. A
non-default terminal backend runs LLM-emitted shell commands inside
a container, remote host, or cloud sandbox. This confines the blast
radius of destructive shell — but only of shell. The Python process
running the agent itself stays on the host, along with every code
path that doesn't go through the shell tool: the code-execution
tool, MCP subprocesses, file tools, plugin loading, hook dispatch,
skill loading. This is the right posture when the concern is
LLM-emitted destructive shell and the operator is otherwise
trusted.

**Whole-process wrapping** sandboxes the agent itself. The agent
runs inside an external runtime that enforces filesystem, network,
process, and inference policies across the entire agent process
tree. [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) is
the reference deployment. Under this posture, every code path in
the agent is subject to the same policy, and the in-process
heuristics in §2.3 become accident-prevention layered on top of a
real boundary. This is the supported posture when the agent
ingests content from surfaces the operator does not control — the
open web, inbound email, multi-user channels, untrusted MCP
servers — and for production or shared deployments.

Operators running the default local backend with untrusted input
surfaces, or running a terminal-backend sandbox and expecting it to
contain code paths that don't go through the shell, are operating
outside the supported security posture.

### 2.2 Credential Scoping

Hermes filters the environment it passes to its lower-trust
in-process components: shell subprocesses, MCP subprocesses, and
the code-execution child. Credentials like provider API keys and
gateway tokens are stripped by default; variables explicitly
declared by the operator or by a loaded skill are passed through.

This reduces casual exfiltration. It is not containment. A
component with code-execution primitives can always reach
filesystem-resident credentials that the agent process itself can
read.

### 2.3 In-Process Heuristics

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
  is operator review before install.

### 2.4 Gateway Authorization

When the gateway integrates with a messaging platform, each platform
adapter authenticates callers against an operator-configured
allowlist. **An allowlist is required for every enabled adapter.**
Adapters should refuse to dispatch agent work, resolve approvals, or
relay output until an allowlist is set; code paths that fail open
when no allowlist is configured are code bugs in scope under §3.1.
Within the allowlist, all authorized callers are equally trusted.
Session identifiers are routing handles, not authorization
boundaries.

### 2.5 Agent-Loaded Content

Hermes chooses, by design, to load and execute content from specific
on-disk locations at its own initiative — skills, hooks, plugins,
operator-configured shortcuts. Content placed in these locations
becomes code the agent runs on its next session, hook dispatch, or
command invocation.

Hermes does not claim these locations are protected files.
Filesystem-level protection is whatever the OS provides under the
operator's chosen isolation posture (§2.1). What Hermes commits to
is narrower and different: **attacker-influenced input must not be
chainable into a write that Hermes would later load and execute on
its own initiative**. The concern is not what the filesystem
allows; it is what Hermes loads.

---

## 3. Scope

### 3.1 In Scope

- Escape from a declared OS-level isolation posture (§2.1): an
  attacker-controlled code path reaching state that the posture
  claimed to confine.
- Unauthorized gateway access: a caller outside the configured
  allowlist dispatching work, receiving output, or resolving
  approvals (§2.4).
- Credential exfiltration: leakage of operator credentials or
  session authorization material to a destination outside the
  operator's trust envelope.
- Untrusted input chaining into agent-loaded content: an untrusted
  input surface chains into a write whose target is a location
  Hermes loads and executes on its own initiative (§2.5).
- Output integrity failures into external platforms: agent output
  rendered on a receiving platform with unintended authority —
  broadcast-mention passthrough, content that fetches attacker
  resources for every recipient, markup injection into hosted UIs.
- Trust-model documentation violations: code behaving contrary to
  what this policy states, where an operator relying on the policy
  would reasonably expect otherwise.

### 3.2 Out of Scope

"Out of scope" here means "not a security vulnerability under this
policy." It does not mean "not worth reporting." Improvements to the
in-process heuristics, hardening ideas, and UX fixes are welcome as
regular issues or pull requests — we can always make the approval
gate catch more patterns, make redaction smarter, or tighten adapter
behavior. These items just don't go through the private-disclosure
channel and don't receive advisories.

- **Bypasses of in-process heuristics (§2.3)** — approval-gate regex
  bypasses, redaction bypasses, Skills Guard pattern bypasses, and
  analogous reports against future heuristics. These components are
  not boundaries; defeating them is not a vulnerability under this
  policy.
- **Prompt injection that does not chain to a §3.1 outcome.** Getting
  the LLM to emit unusual text or "ignore previous instructions" is
  not itself a vulnerability; it becomes one only when it results in
  something §3.1 describes.
- **Consequences of a chosen isolation posture.** Reports that a
  code path operating within its posture's scope can do what that
  posture permits are not vulnerabilities. Examples: shell tools
  reaching host state under the local backend; code-execution or
  file tools reaching host state under terminal-backend isolation
  that only sandboxes shell; reports whose preconditions require
  pre-existing write access to operator-owned configuration or
  credential files (those are already inside the operator's trust
  envelope).
- **Public exposure without external controls.** Exposing the
  gateway or API to the public internet without authentication,
  VPN, or firewall.
- **Documented break-glass settings.** Disabled approvals, local
  backend in production, development profiles that bypass
  hermes-home security, and similar operator-selected trade-offs.
- **Tool-level read/write restrictions on a posture where shell is
  permitted.** If a path is reachable via the terminal tool, reports
  that other file tools can reach it add nothing.

---

## 4. Deployment Hardening

The single most important hardening decision is matching isolation
(§2.1) to the trust of the content the agent will ingest. Beyond
that:

- Run the agent as a non-root user. The supplied container image
  does this by default.
- Keep credentials in the operator credential file with tight
  permissions, never in the main config, never in version control.
  Under OpenShell, use its Provider store rather than an on-disk
  credential file.
- Do not expose the gateway or API to the public internet without
  VPN, Tailscale, or firewall protection. Under OpenShell, use the
  network policy layer to restrict egress.
- Configure a caller allowlist for every gateway adapter you enable
  (§2.4).
- Review third-party skills before install. Skills Guard reports and
  the install audit log are the review surface.
- The OSV malware database is consulted before launching
  ecosystem-resolved MCP servers. Additional supply-chain guards
  on dependency and bundled-package changes run in CI; see
  `CONTRIBUTING.md` for specifics.

---

## 5. Disclosure

- **Coordinated disclosure window:** 90 days from report, or until a
  fix is released, whichever comes first.
- **Channel:** the GHSA thread or email correspondence with
  security@nousresearch.com.
- **Credit:** reporters are credited in release notes unless
  anonymity is requested.
