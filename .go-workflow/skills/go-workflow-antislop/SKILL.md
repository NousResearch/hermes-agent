---
name: go-workflow-antislop
description: ANTISLOP phase for repo-contained go-workflow runs. Remove sloppy artifacts before shipping.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, antislop]
---

# go-workflow phase: ANTISLOP

## Purpose

Remove sloppy artifacts before shipping.

## Inputs

- diff
- docs
- final report draft

## Outputs

- cleaned comments/docs/tasks
- secret/slop scan result

## Allowed mutations

- small cleanup inside already allowed paths

## Required evidence

- diff check
- no placeholders/secrets note

## Failure / stop conditions

- placeholder TODOs
- AI-ish filler
- secret-like strings

## Handoff contract

Change set is clean enough to ship.
