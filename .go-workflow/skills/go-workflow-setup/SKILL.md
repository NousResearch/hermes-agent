---
name: go-workflow-setup
description: SETUP phase for repo-contained go-workflow runs. Establish repository context before planning or editing.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, setup]
---

# go-workflow phase: SETUP

## Purpose

Establish repository context before planning or editing.

## Inputs

- AGENTS.md
- .go-workflow/config.yaml
- .go-workflow/goals.yaml
- .go-workflow/tasks.yaml
- .go-workflow/gates.yaml
- git status

## Outputs

- current repo state
- available task queue
- known dirty/untracked files

## Allowed mutations

- none; read-only unless explicit setup/hygiene task

## Required evidence

- validation command output
- git status summary

## Failure / stop conditions

- missing workflow files
- dirty state that overlaps intended task
- unknown repo contract

## Handoff contract

Repo context is explicit enough for PLAN without guessing.
