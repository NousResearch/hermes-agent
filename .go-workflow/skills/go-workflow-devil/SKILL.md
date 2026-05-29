---
name: go-workflow-devil
description: DEVIL phase for repo-contained go-workflow runs. Run adversarial review for risky or multi-file changes.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, devil]
---

# go-workflow phase: DEVIL

## Purpose

Run adversarial review for risky or multi-file changes.

## Inputs

- diff
- task acceptance
- verification evidence
- docs decision

## Outputs

- adversarial verdict
- fix list or approval

## Allowed mutations

- review notes; fixes only after returning to BUILD/VERIFY

## Required evidence

- verdict A/B/C/D or explicit not-required reason

## Failure / stop conditions

- verdict C/D
- unreviewed risky auth/data/scheduling change

## Handoff contract

Either safe-to-ship verdict or concrete fixes.
