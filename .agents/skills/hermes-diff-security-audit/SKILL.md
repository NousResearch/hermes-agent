---
name: hermes-diff-security-audit
description: Pre-commit diff security and compliance audit for Hermes Desktop/Webapp. Scans staged and unstaged diffs for leaked secrets, tokens, credentials, or personal data; inspects React, TSX, and CSS diffs for OWASP/XSS sinks and CSP regressions; and verifies strict adherence to repository rules and technical boundaries.
---

Based on: [https://skills.sh/getsentry/skills/security-review](https://skills.sh/getsentry/skills/security-review) (Apache License 2.0)

# Hermes Diff Security Audit

Pre-commit diff security and compliance audit designed specifically for the Hermes Desktop/Webapp codebase (`apps/desktop/`).

## When to Use
- Before finalizing changes or preparing a pull request review.
- When auditing staged and unstaged git diffs for compliance and security regressions.
- When checking React/TSX components and Tailwind/CSS files for dangerous DOM sinks or unsafe configuration.

---

## Actionable Steps

### 1. Inspect Staged and Unstaged Diff Scope
- Run `git diff` and `git diff --cached` to isolate all modified lines in the current workspace.
- Ensure no changes touch forbidden files (`package.json`, `pnpm-lock.yaml`, lockfiles, or files outside `apps/desktop/`).
- Verify the current branch is not `main` (must be on a dedicated feature/fix branch).

### 2. Scan for Secrets, Tokens, and PII
- Check every added line for accidental inclusion of:
  - API keys (`sk-`, `ghp_`, bearer tokens, private keys).
  - Passwords, connection strings, or internal credentials.
  - Hardcoded paths containing user home directories or personal names.
  - Personal identifiable information (PII) or telemetry logs containing user data.

### 3. Audit React/TSX & CSS Sinks (OWASP/XSS)
- Inspect all modified `.tsx`, `.ts`, and `.css` files for high-confidence security risks:
  - **XSS Sinks**: Flag any usage of `dangerouslySetInnerHTML`, unsanitized `customCSS` injection, or unsafe DOM manipulation (`innerHTML`, `outerHTML`).
  - **External Links**: Verify all external anchor tags have `rel="noopener noreferrer"` and `target="_blank"`.
  - **CSP Regressions**: Check for inline `<script>` tags, inline `style=` attributes injected from dynamic user data, or unsafe eval usage (`eval()`, `new Function()`, `setTimeout` with string arguments).

### 4. Verify Project Compliance (`.ai/regras.md` & `.ai/teto.md`)
- **No lockfile modifications**: Confirm zero changes to package managers or dependencies.
- **No force pushes or main branch commits**: Confirm working branch integrity.
- **No credential access**: Confirm no attempts to read `.env` or personal credential stores.

---

## What this skill does NOT do
- Never reads `.env`, secret files, or credential stores.
- Never bypasses authentication walls or login screens.
- Never installs third-party vulnerability scanners or runs untrusted external scripts.
- Never commits changes or pushes to remote repositories automatically.
