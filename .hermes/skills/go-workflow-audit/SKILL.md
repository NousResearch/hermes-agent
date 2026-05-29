---
name: go-workflow-audit
description: AUDIT support skill for repo-contained go-workflow runs. Run selectable quality gates for code, tests, architecture, API, performance, accessibility, UX, and security-style checks.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, audit]
---

# go-workflow support skill: AUDIT

## Purpose

Run selectable quality gates for code, tests, architecture, API, performance, accessibility, UX, and security-style checks.

## Commands / modes

`code`, `test`, `arch`, `api`, `perf`, `a11y`, `ux`, `security`, `ship`, `diff`, `re-audit`

## Inputs

- claimed task acceptance
- diff or target path
- verification output
- docs impact
- selected audit profiles

## Outputs

- profile-specific findings
- severity/ship verdict
- required fixes or explicit pass evidence

## Allowed mutations

- review notes only; fixes must return to BUILD/VERIFY before finish

## Required evidence

- audit_profile_selected
- audit_evidence_recorded

## Failure / stop conditions

- unreviewed risky code path
- security/data/auth/perf concern without disposition
- missing selected audit evidence

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase audit --evidence key=value` before depending on this support gate.
