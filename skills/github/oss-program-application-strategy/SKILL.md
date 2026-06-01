---
name: oss-program-application-strategy
description: "Evaluate OSS program fit with live evidence."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [open-source, grants, applications, github, maintainer, strategy]
    related_skills: [github-repo-management, github-pr-workflow, github-issues]
---

# OSS Program Application Strategy Skill

Use this skill when a user asks whether and how to apply to an open-source support program. It compares the program's stated intent with the user's credible public work and recommends a portfolio shape.

This skill does not treat social posts or old research notes as eligibility rules. Official pages and current repository evidence win.

## When to Use

- Use when the user is comparing an upstream contribution, fork, companion toolkit, or new project for an OSS program.
- Use when a program offers maintainer credits, grants, API credits, security tooling, or contributor support.
- Use when the user needs a grounded application narrative and immediate action plan.
- Do not use for programs unrelated to open-source contribution or maintainer work.

## Prerequisites

- A target program, sponsor, or program category is known.
- The user's relevant repos, handles, or candidate project ideas are available.
- Web or GitHub evidence can be checked live before final recommendations.

## How to Run

Research in three layers:

1. Official program intent and eligibility.
2. Public ecosystem evidence from GitHub and project docs.
3. Community sentiment and prior art, clearly labeled as non-authoritative.

Use `terminal`, web search, or the GitHub connector to refresh facts that drift.

## Quick Reference

| Option | Prefer When |
| --- | --- |
| Upstream PRs | The target project is active and accepts focused fixes. |
| Fork | The fork has a distinct audience or safety architecture. |
| Companion toolkit | The value is workflow integration, docs, templates, or automation. |
| New project | Existing tools cannot express the user's thesis. |

## Procedure

1. Read the official program page, terms, announcement, and linked docs.
2. Extract who the program supports and what work it rewards.
3. Map each user option into the program's own language.
4. Compare current GitHub evidence: activity, issues, stars, forks, contributors, license, and recent releases.
5. Check prior art and sentiment without overstating it.
6. Recommend a portfolio shape and write the application narrative.
7. Give an immediate plan with one or two public proof points the user can ship.

## Pitfalls

- Do not recommend a fork just because the program name matches the upstream tool.
- Do not quote stale GitHub counts without re-checking.
- Do not persist transient search failures as skill rules.
- Do not fabricate metrics or eligibility language.
- Do not underplay maintainer operations; review, triage, security, and release work are often the strongest story.

## Verification

- Official criteria are linked or quoted from current sources.
- Ecosystem metrics are current or explicitly marked as unavailable.
- The recommendation explains why one portfolio shape beats the others.
- The final narrative names concrete maintainer outcomes, not only tools used.

## References

- `references/codex-for-open-source-positioning.md`
