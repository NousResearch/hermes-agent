---
name: code-security
description: Security analysis skill for code changes. Wraps existing security tools — static scan patterns, injection detection, credential audit. Auto-loads requesting-code-review for pre-commit security gates.
version: 1.0.0
metadata:
  hermes:
    tags: [security, code-review, audit, scanning]
    related_skills: [requesting-code-review, clawsec-suite]
adoption_status: provisional
---

# Code Security

A lightweight bridge skill that loads the security scanning patterns from `requesting-code-review` and advisory monitoring from `clawsec-suite`. This skill exists so profiles that need security awareness (wesker, octacon) can reference it in `always_skills` without loading the full pre-commit review pipeline.

## What this provides

When loaded alongside `clawsec-suite` and `requesting-code-review`, this skill enables:

- Static security scan of code diffs (hardcoded secrets, injection, eval, unsafe deserialization)
- Credential exposure detection in configs and scripts
- Advisory feed monitoring via clawsec-suite
- Security-focused code review gate (Gate 4 in the 5-gate framework)

## Quick security checks

For any code change, run:

```bash
# Hardcoded secrets
git diff --cached | grep "^+" | grep -iE "(api_key|secret|password|token|passwd)\s*=\s*['\"][^'\"]{6,}['\"]"

# Shell injection
git diff --cached | grep "^+" | grep -E "os\.system\(|subprocess.*shell=True"

# Dangerous eval/exec
git diff --cached | grep "^+" | grep -E "\beval\(|\bexec\("

# Unsafe deserialization
git diff --cached | grep "^+" | grep -E "pickle\.loads?\("

# SQL injection (string formatting in queries)
git diff --cached | grep "^+" | grep -E "execute\(f\"|\.format\(.*SELECT|\.format\(.*INSERT"
```

## Third-Party Code Adoption

When adopting open-source tools/skills as Hermes assets, use **clean-room rewrite only**. See `references/third-party-code-adoption.md` for the full policy — covers licensing risk, prompt injection vectors, and process.

Quick rule: no LICENSE file → clean-room only. Security-sensitive domain (CLI, file writes, network, LLM interaction) → clean-room always.

## Related files

- `requesting-code-review` — full pre-commit security + quality gate pipeline
- `clawsec-suite` — advisory feed monitoring and cryptographic verification
- `references/third-party-code-adoption.md` — clean-room adoption policy with rationale and check table
