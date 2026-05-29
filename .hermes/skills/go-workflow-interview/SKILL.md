---
name: go-workflow-interview
description: INTERVIEW support skill for repo-contained go-workflow runs. Ambiguity gate for work that cannot be safely planned without a small number of targeted questions.
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, interview]
---

# go-workflow support skill: INTERVIEW

## Purpose

Ambiguity gate for work that cannot be safely planned without a small number of targeted questions.

## Commands / modes

`score`, `ask`, `assume`, `block`

## Inputs

- user request
- repo contract
- known tasks/goals
- ambiguity score

## Outputs

- minimum clarifying questions
- assumptions
- blocked/waiting task state when needed

## Allowed mutations

- tasks.yaml status/notes only when recording a real waiting state

## Required evidence

- ambiguity_assessed
- questions_minimized_or_assumptions_recorded

## Failure / stop conditions

- choice changes implementation path materially
- missing credentials
- risk of data loss

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase interview --evidence key=value` before depending on this support gate.
