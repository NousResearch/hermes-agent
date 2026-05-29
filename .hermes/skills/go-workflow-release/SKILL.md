---
name: go-workflow-release
description: RELEASE support skill for repo-contained go-workflow runs. Separate release/version/tag/notes behavior from generic docs-ledger work and make final publish evidence explicit.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, release]
---

# go-workflow support skill: RELEASE

## Purpose

Separate release/version/tag/notes behavior from generic docs-ledger work and make final publish evidence explicit.

## Commands / modes

`dry-run`, `version`, `notes`, `tag`, `publish`

## Inputs

- completed task ledgers
- version policy
- release scope
- git/PR status
- explicit publish flag for external side effects

## Outputs

- version bump or no-release rationale
- tag/release notes plan
- publish URL or dry-run evidence

## Allowed mutations

- configured version files
- release notes
- tags/releases only when explicitly requested

## Required evidence

- release_scope_confirmed
- version_or_no_release_recorded
- final_ship_evidence_recorded

## Failure / stop conditions

- verification/docs/devil/antislop gates not passed
- unclear version bump
- implicit GitHub release side effect

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase release --evidence key=value` before depending on this support gate.
