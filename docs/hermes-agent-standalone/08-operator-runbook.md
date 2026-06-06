---
title: Hermes Agent Operator Runbook
tags:
  - hermes-agent
  - runbook
status: active
updated: 2026-05-20
---

# 08 Operator Runbook

## Normal Work Request

1. Open `http://127.0.0.1:9119/chat` or run `hermes`.
2. Name the project and goal.
3. Hermes Agent loads `/workspace-40` or the relevant skill directly.
4. Hermes Agent reads the target context pack.
5. Hermes Agent creates or updates a `workspace-40` kanban task.
6. Hermes Agent performs the work in the requested project.
7. Hermes Agent runs the project's verification gate.
8. Hermes Agent writes an Obsidian report when the result should become reusable knowledge.

## Hard Boundary

- Do not call HermesNous `7422`.
- Do not call Hermes Labs `7421`.
- Do not copy `.env` values into docs, vault notes, or skills.
- Do not symlink the HermesAgent vault to legacy vaults.

## Required Delivery Evidence

- Changed files.
- Commands/tests run.
- Localhost or VPS status.
- Numeric phase compliance.
