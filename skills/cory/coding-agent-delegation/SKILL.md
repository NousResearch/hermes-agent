---
name: coding-agent-delegation
description: Prepare bounded delegation framing for future coding agents without starting execution inside interpretation.
---

# Coding Agent Delegation

Use this skill when the request is moving toward implementation and future coding-agent handoff is likely.

Rules:
- Do not delegate from the interpretation step.
- Identify the likely work slices, dependencies, and approval boundaries that a later delegation step will need.
- Keep the decomposition reversible until a human reviews it.

Output expectations:
- Explain what would need to be delegated later.
- Preserve boundaries between planning, approval, and implementation.
