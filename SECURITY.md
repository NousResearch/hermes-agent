# Hermes Agent Security Policy

This document outlines the security protocols, trust model, and deployment hardening guidelines for the **Hermes Agent** project.

## 1. Vulnerability Reporting

Hermes Agent does **not** operate a bug bounty program. Security issues should be reported via [GitHub Security Advisories (GHSA)](https://github.com/NousResearch/hermes-agent/security/advisories/new) or by emailing **security@nousresearch.com**. Do not open public issues for security vulnerabilities.

### Required Submission Details
- **Title & Severity:** Concise description and CVSS score/rating.
- **Affected Component:** Exact file path and line range (e.g., `tools/approval.py:120-145`).
- **Environment:** Output of `hermes version`, commit SHA, OS, and Python version.
- **Reproduction:** Step-by-step Proof-of-Concept (PoC) against `main` or the latest release.
- **Impact:** Explanation of what trust boundary was crossed.

---

## 2. Trust Model

The core assumption is that Hermes is a **personal agent** with one trusted operator.

### Operator & Session Trust
- **Single Tenant:** The system protects the operator from LLM actions, not from malicious co-tenants. Multi-user isolation must happen at the OS/host level.
- **Gateway Security:** Authorized callers (Telegram, Discord, Slack, etc.) receive equal trust. Session keys are used for routing, not as authorization boundaries.
- **Execution:** Defaults to `terminal.backend: local` (direct host execution). Container isolation (Docker, Modal, Daytona) is opt-in for sandboxing.

### Dangerous Command Approval
The approval system (`tools/approval.py`) is a core security boundary. Terminal commands, file operations, and other potentially destructive actions are gated behind explicit user confirmation before execution. The approval mode is configurable via `approvals.mode` in `config.yaml`:
- `"on"` (default) — prompts the user to approve dangerous commands.
- `"auto"` — auto-approves after a configurable delay.
- `"off"` — disables the gate entirely (break-glass; see Section 3).

### Output Redaction
`agent/redact.py` strips secret-like patterns (API keys, tokens, credentials) from all display output before it reaches the terminal or gateway platform. This prevents accidental credential leakage in chat logs, tool previews, and response text. Redaction operates on the display layer only — underlying values remain intact for internal agent operations.

### Skills vs. MCP Servers
- **Installed Skills:** High trust. Equivalent to local host code; skills can read environment variables and run arbitrary commands.
- **MCP Servers:** Lower trust. MCP subprocesses receive a filtered environment (`_build_safe_env()` in `tools/mcp_tool.py`) — only safe baseline variables (`PATH`, `HOME`, `XDG_*`) plus variables explicitly declared in the server's `env` config block are passed through. Host credentials are stripped by default. Additionally, packages invoked via `npx`/`uvx` are checked against the OSV malware database before spawning.

### Code Execution Sandbox
The `execute_code` tool (`tools/code_execution_tool.py`) runs LLM-generated Python scripts in a child process with API keys and tokens stripped from the environment to prevent credential exfiltration. Only environment variables explicitly declared by loaded skills (via `env_passthrough`) or by the user in `config.yaml` (`terminal.env_passthrough`) are passed through. The child accesses Hermes tools via RPC, not direct API calls.

### Subagents
- **No recursive delegation:** The `delegate_task` tool is disabled for child agents.
- **Depth limit:** `MAX_DEPTH = 2` — parent (depth 0) can spawn a child (depth 1); grandchildren are rejected.
- **Memory isolation:** Subagents run with `skip_memory=True` and do not have access to the parent's persistent memory provider. The parent receives only the task prompt and final response as an observation.

---

## 3. Out of Scope (Non-Vulnerabilities)

The following scenarios are **not** considered security breaches:
- **Prompt Injection:** Unless it results in a concrete bypass of the approval system, toolset restrictions, or container sandbox.
- **Public Exposure:** Deploying the gateway to the public internet without external authentication or network protection.
- **Trusted State Access:** Reports that require pre-existing write access to `~/.hermes/`, `.env`, or `config.yaml` (these are operator-owned files).
- **Default Behavior:** Host-level command execution when `terminal.backend` is set to `local` — this is the documented default, not a vulnerability.
- **Configuration Trade-offs:** Intentional break-glass settings such as `approvals.mode: "off"` or `terminal.backend: local` in production.

---

## 4. Deployment Hardening & Best Practices

### Filesystem & Network
- **Production sandboxing:** Use container backends (`docker`, `modal`, `daytona`) instead of `local` for untrusted workloads.
- **File permissions:** Run as non-root (the Docker image uses UID 10000); protect credentials with `chmod 600 ~/.hermes/.env` on local installs.
- **Network exposure:** Do not expose the gateway or API server to the public internet without VPN, Tailscale, or firewall protection. SSRF protection is enabled by default across all gateway platform adapters (Telegram, Discord, Slack, Matrix, Mattermost, etc.) with redirect validation. Note: the local terminal backend does not apply SSRF filtering, as it operates within the trusted operator's environment.

### Skills & Supply Chain
- **Skill installation:** Review Skills Guard reports (`tools/skills_guard.py`) before installing third-party skills. The audit log at `~/.hermes/skills/.hub/audit.log` tracks every install and removal.
- **MCP safety:** OSV malware checking runs automatically for `npx`/`uvx` packages before MCP server processes are spawned.
- **CI/CD:** GitHub Actions are pinned to full commit SHAs. The `supply-chain-audit.yml` workflow blocks PRs containing `.pth` files or suspicious `base64`+`exec` patterns.

### Credential Storage
- API keys and tokens belong exclusively in `~/.hermes/.env` — never in `config.yaml` or checked into version control.
- The credential pool system (`agent/credential_pool.py`) handles key rotation and fallback. Credentials are resolved from environment variables, not stored in plaintext databases.

---

## 5. Disclosure Process

- **Coordinated Disclosure:** 90-day window or until a fix is released, whichever comes first.
- **Communication:** All updates occur via the GHSA thread or email correspondence with security@nousresearch.com.
- **Credits:** Reporters are credited in release notes unless anonymity is requested.
