---
name: go-workflow-ship
description: SHIP phase for repo-contained go-workflow runs. Commit, push or PR, and report final git evidence.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, ship]
---

# go-workflow phase: SHIP

## Purpose

Commit, push or PR, and report final git evidence.

## Inputs

- verified diff
- ledger
- git status
- commit policy

## Outputs

- commit/PR/push evidence
- final branch state

## Allowed mutations

- git index/history for relevant files only

## Required evidence

- commit or PR link
- push/ahead-behind status
- final git status

## Failure / stop conditions

- failing checks
- unrelated staged files
- behind remote without safe rebase

## Handoff contract

Final user report includes tasks, commits, verification, docs, and open risks.
