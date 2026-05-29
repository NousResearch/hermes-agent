---
name: go-workflow-git
description: GIT ROUTER support skill for repo-contained go-workflow runs. Keep repository state safe: status hygiene, selective staging, ship, CI repair, PR/check inspection, and clean return to main.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, git]
---

# go-workflow support skill: GIT ROUTER

## Purpose

Keep repository state safe: status hygiene, selective staging, ship, CI repair, PR/check inspection, and clean return to main.

## Commands / modes

`status`, `stage`, `commit`, `ship`, `fix-ci`, `pr-checks`, `return-main`

## Inputs

- git status
- claimed task scope.modify
- commit policy
- remote/upstream state
- CI/PR state when present

## Outputs

- clean or explained git state
- focused commit/PR
- CI/check evidence
- main/default-branch return evidence

## Allowed mutations

- git index/history for in-scope paths only
- branches/PRs when the task commit policy allows it

## Required evidence

- git_status_clean_or_explained
- selective_staging_used
- ci_or_pr_status_checked

## Failure / stop conditions

- unrelated dirty files would be mixed
- behind remote without safe rebase
- failing required checks
- cannot return to main/default branch

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase git --evidence key=value` before depending on this support gate.
