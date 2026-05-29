---
name: go-workflow-better
description: BETTER support skill for repo-contained go-workflow runs. Post-run improvement loop: turn a completed task/session into better workflow rules, tasks, tests, docs, or skill updates.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, better]
---

# go-workflow support skill: BETTER

## Purpose

Post-run improvement loop: turn a completed task/session into better workflow rules, tasks, tests, docs, or skill updates.

## Commands / modes

`reflect`, `add-task`, `patch-skill`, `add-gate`, `decline`

## Inputs

- completed ledger
- verification failures
- review feedback
- user correction
- repo contract

## Outputs

- improvement tasks
- skill/doc patches
- tests or gates to prevent recurrence

## Allowed mutations

- tasks.yaml/goals.yaml
- docs
- skills
- tests when the improvement task is claimed

## Required evidence

- improvement_source_recorded
- reusable_change_captured_or_declined

## Failure / stop conditions

- one-off task progress being mistaken for durable knowledge
- unverified process change

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase better --evidence key=value` before depending on this support gate.
