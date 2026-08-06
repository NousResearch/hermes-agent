---
name: production-audit
description: "Local-evidence production readiness audit for shipped apps, pre-launch reviews, and post-merge checks. Use when asking 'is this production-ready', 'what breaks in prod', or before a launch."
version: 1.0.0
author: Hermes Agent (adapted from community)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [production, audit, launch, readiness, safety]
    related_skills: [verification-loop, safety-guard, gstack-review]
---

# Production Audit

Use this skill when the user asks whether an application is ready to ship, what could break in production, or what must be fixed before a launch.

## When to Use

- "Is this production-ready?", "What would break in prod?", "What did we miss?"
- A feature was merged and needs a pre-deploy or post-merge risk pass
- A public launch, demo, customer rollout, or investor walkthrough is close
- CI is green but the user wants production risk, not only test status

## When Not to Use

- During active implementation when the right lens is line-level secure coding
- For pure libraries, templates, docs-only repos unless checking packaging readiness
- When the user asks for a formal compliance audit

## How It Works

Build the audit from local and user-authorized evidence in this order:

1. **Establish the release surface** — What's being shipped? Which services, endpoints, features?
2. **Read recent changes** — `git log`, `git diff` against base branch
3. **Inspect boundaries** — Runtime, auth, data, payment, background jobs, AI, deployment
4. **Check CI, tests, migrations, env docs, rollback path**
5. **Produce ship/block recommendation** with specific fixes

## Evidence Checklist

Start with cheap, local signals:

```bash
git status --short --branch
git log --oneline --decorate -20
git diff --stat origin/main...HEAD
```

Then check:

| Area              | What to check                                                       |
| ----------------- | ------------------------------------------------------------------- |
| **Auth**          | Are new endpoints protected? Token expiry? Session handling?        |
| **Data**          | Migrations reversible? Backups configured? No accidental data loss? |
| **Secrets**       | Any hardcoded keys? `.env.example` updated? Rotation procedures?    |
| **Errors**        | Error handling on every boundary? User-safe messages? Logging?      |
| **Performance**   | New N+1 queries? Unbounded collections? Missing indexes?            |
| **Dependencies**  | New packages vetted? Version pinned? Security advisories?           |
| **Config**        | Feature flags? Environment-specific config? Graceful degradation?   |
| **Observability** | Logging on key paths? Metrics? Alerting thresholds?                 |
| **Rollback**      | Can this be rolled back? Data migration reversible? Downtime?       |

## Output Format

```
PRODUCTION AUDIT
===============
Release Surface: [what's being shipped]
Risk Level: [LOW / MEDIUM / HIGH]

Critical (BLOCK deployment):
- [Finding] → [Fix]

Important (fix before next release):
- [Finding] → [Fix]

Advisory (nice to have):
- [Finding] → [Suggestion]

Verdict: [SHIP / SHIP WITH CAUTION / BLOCK]
```

## Hermes Integration

- Use `terminal` for git and build commands
- Use `read_file` to inspect config, migrations, environment files
- Use `search_files` to find secrets, error handlers, N+1 patterns
- Use `delegate_task` for parallel boundary inspection
- Combine with `verification-loop` for test/build checks
- Combine with `safety-guard` for destructive operation awareness
