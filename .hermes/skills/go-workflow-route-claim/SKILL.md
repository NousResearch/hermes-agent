---
name: go-workflow-route-claim
description: ROUTE/CLAIM phase for repo-contained go-workflow runs. Select exactly one executable task and write an exclusive handoff.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, route-claim]
---

# go-workflow phase: ROUTE/CLAIM

## Purpose

Select exactly one executable task and write an exclusive handoff.

## Inputs

- ready task
- dependency status
- agent id

## Outputs

- exclusive claim
- handoff markdown
- lease metadata

## Allowed mutations

- .go-workflow/tasks.yaml claim fields
- .go-workflow/runs/*-handoff.md

## Required evidence

- claim written
- handoff path
- scope read/modify listed

## Failure / stop conditions

- task not ready
- dependencies not done
- overlapping active claim

## Handoff contract

Worker reads handoff and stays inside scope.modify.
