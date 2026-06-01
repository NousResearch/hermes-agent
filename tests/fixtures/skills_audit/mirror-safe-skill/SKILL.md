---
name: safe-audit-fixture
description: "Minimal safe skill for operator security audit tests."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
---

# Safe Audit Fixture

Use `read_file` and `search_files` to explore the repository. Use `patch` only
on files the operator explicitly names.

## Procedure

1. Read the target file with `read_file`.
2. Summarize findings for the operator in the chat transcript.

## Verification

Confirm the operator received a summary before ending the turn.
