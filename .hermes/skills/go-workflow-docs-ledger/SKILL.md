---
name: go-workflow-docs-ledger
description: DOCS/LEDGER phase for repo-contained go-workflow runs. Update docs, task state, evidence, and run ledgers.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, docs-ledger]
---

# go-workflow phase: DOCS/LEDGER

## Purpose

Update docs, task state, evidence, and run ledgers.

## Inputs

- accepted changes
- docs policy
- evidence

## Outputs

- updated docs
- updated tasks.yaml/tasks.md
- run ledger

## Allowed mutations

- docs/update paths
- .go-workflow/tasks.yaml
- tasks.md
- .go-workflow/runs/*-ledger.md

## Required evidence

- docs changed or checked/no-change reason
- ledger path
- task evidence

## Failure / stop conditions

- docs drift
- missing evidence
- human cockpit stale

## Handoff contract

Task record and docs describe what changed and why.
