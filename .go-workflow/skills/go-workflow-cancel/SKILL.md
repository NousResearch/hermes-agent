---
name: go-workflow-cancel
description: CANCEL support skill for repo-contained go-workflow runs. Gracefully stop, supersede, or cancel obsolete workflow tasks without corrupting queue or git state.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, cancel]
---

# go-workflow support skill: CANCEL

## Purpose

Gracefully stop, supersede, or cancel obsolete workflow tasks without corrupting queue or git state.

## Commands / modes

`cancel-task`, `supersede`, `rollback-claim`, `preserve-notes`

## Inputs

- task id or scope to cancel
- current claim status
- dirty files
- superseding decision

## Outputs

- cancelled/superseded task state
- rollback or preserved handoff
- clear open-risk note

## Allowed mutations

- task status/notes
- ledger note
- targeted revert only when explicitly safe

## Required evidence

- cancel_scope_identified
- state_preserved_or_reverted
- next_action_recorded

## Failure / stop conditions

- unclear whether user wants cancel vs pause
- uncommitted valuable work would be lost
- claimed task by another agent

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase cancel --evidence key=value` before depending on this support gate.
