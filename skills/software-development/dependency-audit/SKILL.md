---
name: dependency-audit
description: "Use when the user asks to audit, scan, or check project dependencies for security vulnerabilities, outdated packages, or license compliance issues. Runs ecosystem-native audit tools (pip-audit, npm audit, cargo audit, bundler-audit), queries the OSV API for CVEs and malware advisories, summarises findings by severity, and produces a prioritised remediation plan with exact upgrade commands."
version: 1.0.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, dependencies, audit, vulnerability, cve, npm, pip, cargo, license]
    related_skills: [github-code-review, systematic-debugging, requesting-code-review]
---

# Dependency Audit Skill

## Overview

This skill performs a comprehensive security and health audit of a project's
third-party dependencies. It combines ecosystem-native tooling with the Google
OSV database to surface:

- **Known CVEs and malware** (CRITICAL / HIGH / MEDIUM / LOW)
- **Outdated packages** with available upgrades
- **License compliance** issues (copyleft in a proprietary project, etc.)
- A **prioritised remediation plan** with ready-to-run upgrade commands

It works across the four most common ecosystems — Python (pip), Node.js (npm/
yarn/pnpm), Rust (cargo), and Ruby (bundler) — and can be combined in
polyglot monorepos.

The OSV malware check re-uses the logic in `tools/osv_check.py` (already
present in this repo) and extends it from MCP-package-only coverage to
all declared dependencies.

---

## When to Use

Trigger this skill when the user says things like:

- "Audit our dependencies"
- "Check for vulnerable packages"
- "Are there any CVEs in this project?"
- "Scan for outdated or insecure libraries"
- "Run a security scan on the repo"
- "Check license compliance"
- Before merging a PR that bumps a dependency
- During a security review or pentest preparation

---

## Step-by-Step Protocol

### 1. Detect Ecosystems

Identify which lock-files / manifests are present in the project root (or the
directory the user specified):

| File | Ecosystem | Audit tool |
|------|-----------|-----------|
| `requirements.txt`, `pyproject.toml`, `Pipfile.lock`, `poetry.lock` | Python | `pip-audit` |
| `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml` | Node.js | `npm audit` / `yarn audit` / `pnpm audit` |
| `Cargo.lock` | Rust | `cargo audit` |
| `Gemfile.lock` | Ruby | `bundle audit` |

Run detection with:

```bash
ls requirements.txt pyproject.toml Pipfile.lock poetry.lock \
   package-lock.json yarn.lock pnpm-lock.yaml \
   Cargo.lock Gemfile.lock 2>/dev/null
```

### 2. Install Missing Audit Tools (if needed)

Only install tools that are absent. Check with `which <tool>` first.

```bash
# Python
pip install pip-audit --quiet

# Node — npm audit is bundled with npm; for yarn:
# yarn global add improved-yarn-audit

# Rust
cargo install cargo-audit --quiet

# Ruby
gem install bundler-audit --quiet && bundle-audit update
```

### 3. Run Ecosystem Audits

Run each applicable tool and capture JSON output for structured parsing.

**Python:**
```bash
pip-audit --format json --output pip-audit-results.json
# or for poetry projects:
pip-audit --format json --requirement <(poetry export --without-hashes) \
  --output pip-audit-results.json
```

**Node.js (npm):**
```bash
npm audit --json > npm-audit-results.json
```

**Node.js (yarn v1):**
```bash
yarn audit --json > yarn-audit-results.json
```

**Node.js (pnpm):**
```bash
pnpm audit --json > pnpm-audit-results.json
```

**Rust:**
```bash
cargo audit --json > cargo-audit-results.json
```

**Ruby:**
```bash
bundle-audit check --update --format json > bundler-audit-results.json
```

### 4. OSV API Cross-Reference (Optional Deep Check)

For dependencies not caught by native tools (e.g. indirect/transitive deps or
packages on ecosystems without a native auditor), query the OSV API directly.
This mirrors the logic in `tools/osv_check.py`.

```python
import json, urllib.request

def osv_query(package: str, ecosystem: str, version: str | None = None) -> list:
    payload = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.osv.dev/v1/query", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read()).get("vulns", [])
```

Batch queries by reading the lock-file directly when the native tool is
unavailable (e.g. `requirements.txt` without `pip-audit`).

### 5. License Compliance Check

```bash
# Python
pip-licenses --format=json --output-file pip-licenses.json

# Node.js
npx license-checker --json > npm-licenses.json
```

Flag packages whose license matches any of the user's prohibited list
(default: `GPL-2.0`, `GPL-3.0`, `AGPL-3.0`, `LGPL-2.1` when building
proprietary software). Ask the user for their policy if unknown.

### 6. Consolidate and Prioritise Findings

Merge all results into a single severity-ranked table:

| Severity | CVE / Advisory ID | Package | Installed | Fixed In | Action |
|----------|------------------|---------|-----------|----------|--------|
| CRITICAL | CVE-2024-XXXXX | package-name | 1.2.3 | 1.2.4 | `pip install package-name==1.2.4` |
| HIGH | MAL-2024-XXXXX | other-pkg | 0.9.1 | — | Remove / replace |
| MEDIUM | CVE-2023-YYYYY | third-pkg | 2.0.0 | 2.1.0 | `npm install third-pkg@2.1.0` |
| LOW | … | … | … | … | … |

Severity mapping:

- **CRITICAL** — CVSS ≥ 9.0 or any MAL-* (malware) advisory
- **HIGH** — CVSS 7.0–8.9
- **MEDIUM** — CVSS 4.0–6.9
- **LOW** — CVSS < 4.0 or informational

### 7. Generate Remediation Plan

Produce a ready-to-run upgrade script tailored to the ecosystem(s) found.

```bash
# Example output for a Python+Node project
# --- CRITICAL / HIGH fixes ---
pip install "django>=4.2.14"          # CVE-2024-38875
pip install "cryptography>=42.0.4"    # CVE-2024-26130
npm install tough-cookie@4.1.4        # CVE-2023-26136

# --- MEDIUM fixes ---
pip install "pillow>=10.3.0"          # CVE-2024-28219

# --- After upgrading, re-run audit to confirm clean ---
pip-audit && npm audit
```

If an upgrade is not available (zero-day or abandoned package), recommend:

1. Assess actual exploitability in the project's context
2. Apply a monkey-patch / workaround if documented in the advisory
3. Replace with a maintained alternative

### 8. Present Final Report

Structure the report as:

```
## Dependency Audit Report — <project name> — <date>

### Summary
- Ecosystems scanned: Python, Node.js
- Total packages audited: 142
- Vulnerabilities found: 3 CRITICAL, 1 HIGH, 2 MEDIUM, 4 LOW
- License issues: 0
- Outdated (non-vulnerable): 17

### Critical & High (action required)
...table...

### Remediation Script
...bash block...

### License Issues
None found.

### Outdated Packages (no known CVE)
...optional table, collapsible...
```

---

## Tips and Edge Cases

**Monorepos:** If the project has multiple `package.json` files (e.g.
`apps/web/package.json`, `packages/core/package.json`), run `npm audit` in
each directory separately, or use `--workspaces` if available.

**Poetry / uv projects:** Export a flat requirements file first:
```bash
poetry export --without-hashes -f requirements.txt | pip-audit -r /dev/stdin
# or with uv:
uv export --no-hashes | pip-audit -r /dev/stdin
```

**Private registries:** If the project uses a private npm/PyPI registry,
native audit tools may still work but OSV queries will fail for private
packages — that is expected and safe to skip.

**CI integration:** After fixing, suggest adding the audit step to CI:
```yaml
# GitHub Actions example
- name: Dependency audit
  run: pip-audit && npm audit --audit-level=high
```

**False positives:** Some advisories affect only specific configurations or
code paths. If the user confirms a finding is unexploitable, document the
exception with a justification and suggest adding it to the audit ignore list
(`.pip-audit-ignore`, `.nsprc`, etc.).

---

## Related Skills

- `github-code-review` — review a PR that bumps a dependency before merging
- `systematic-debugging` — investigate a runtime crash caused by a patched vuln
- `requesting-code-review` — request team sign-off after applying fixes
