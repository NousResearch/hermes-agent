---
name: go-workflow-verify
description: VERIFY phase for repo-contained go-workflow runs. Run the task and repository checks that prove the change.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, verify]
---

# go-workflow phase: VERIFY

## Purpose

Run the task and repository checks that prove the change.

## Inputs

- task verification list
- repo gates
- changed files

## Outputs

- command results
- failure fixes or blocked status

## Allowed mutations

- fixes needed to satisfy acceptance within scope

## Required evidence

- task verification output
- repo validation output

## Failure / stop conditions

- failing required check
- missing dependency
- unreproducible result

## Handoff contract

All required checks have explicit pass/fail evidence.
