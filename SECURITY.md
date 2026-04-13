# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in **hermes-agent**, please report
it privately. Do **not** open a public GitHub issue.

- **Primary:** file a GitHub private vulnerability report via the repository's
  Security tab.
- **Alternate:** email the maintainer listed in `pyproject.toml`'s `authors`
  field, with the subject prefixed `[SECURITY]`.

We will acknowledge receipt within **72 hours** and aim to provide a
remediation plan or preliminary patch within **14 days** for Critical/High
severity issues. Lower-severity issues may be batched with routine releases.

## Supported Versions

Only the latest released tag of `hermes-agent` receives security fixes.
Users on older tags should upgrade. If that is infeasible due to a
compatibility blocker, note it in the report — we will consider a
back-ported patch case-by-case.

## Scope

In scope:
- `hermes-agent` (Python gateway, CLI, agent loop, cron, tools)
- `hermes-companion` (Swift menu-bar app)
- `hermes-webui`

Out of scope:
- Bugs in third-party dependencies (report to those projects; we track CVE
  advisories via `uv.lock` and renovate).
- Missing hardening that is not exploitable in the documented threat model
  (see below).

## Threat Model

`hermes-agent` is primarily deployed on a **trusted developer workstation**
and, optionally, behind a reverse-proxy on a single-tenant LAN. The
documented threat actors are:

1. **A malicious LLM response** — tool arguments originate from the model
   and must be validated before reaching subprocess/SQL/file APIs.
2. **A malicious fetched web page / RSS feed / email** — untrusted content
   may contain prompt-injection payloads; the agent must not treat it as
   system instructions.
3. **A co-tenant on the same host (multi-user macOS / shared Linux)** —
   file permissions on `~/.hermes/` must exclude other local users.
4. **A network peer reaching the gateway** — when the gateway is exposed
   beyond loopback, bearer-token auth must hold and failed-auth attempts
   must be rate-limited.

Out-of-scope adversaries (currently):
- A compromised OS, kernel, or root user.
- An attacker with local read access to `~/.hermes/.env` (treated as
  already-compromised).
- Side-channel attacks (cache timing, RF emissions, etc.).

## Hardening Controls

Documented defensive controls in the codebase (see
`~/.claude/skills/code-review/SKILL.md` audit report for the full list):

- Path-traversal blocking for tools accepting file paths
  (`HERMES_READ_SAFE_ROOT`, `HERMES_WRITE_SAFE_ROOT`)
- Cron script-output guardrail scanner + untrusted-data sentinel (F-004)
- Session ownership binding via `owner_fingerprint` (F-003)
- 5-fail/60s per-IP auth lockout (F-009)
- Secret redaction across 30+ token formats in logs (`agent/redact.py`)
- ContextVar-isolated approval session keys (`tools/approval.py`)
- Loopback-default gateway binding

## Disclosure

After a fix lands, we credit reporters in the release notes unless
anonymity is requested.
