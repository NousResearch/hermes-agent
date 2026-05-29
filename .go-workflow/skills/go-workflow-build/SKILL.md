---
name: go-workflow-build
description: BUILD phase for repo-contained go-workflow runs. Change only the claimed task's allowed modify scope.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, build]
---

# go-workflow phase: BUILD

## Purpose

Change only the claimed task's allowed modify scope.

## Inputs

- handoff
- scope.modify
- acceptance criteria

## Outputs

- focused code/docs changes

## Allowed mutations

- only paths listed in scope.modify

## Required evidence

- changed path list
- scope compliance note

## Failure / stop conditions

- needed file outside scope
- unrelated dirty file
- unsafe data loss risk

## Handoff contract

Changes are ready for mechanical verification.
