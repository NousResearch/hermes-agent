---
name: go-workflow-plan
description: PLAN phase for repo-contained go-workflow runs. Turn intent and repo state into bounded requirements and acceptance checks.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, plan]
---

# go-workflow phase: PLAN

## Purpose

Turn intent and repo state into bounded requirements and acceptance checks.

## Inputs

- claimed or requested task
- goals.yaml
- tasks.yaml
- project docs

## Outputs

- requirement/acceptance mapping
- docs impact
- verification plan

## Allowed mutations

- tasks.yaml/goals.yaml only when the task is queue planning

## Required evidence

- task has acceptance
- task has verification
- dependencies known

## Failure / stop conditions

- ambiguous source of truth
- missing acceptance
- unresolved dependency

## Handoff contract

A bounded task is ready to be claimed or executed.
