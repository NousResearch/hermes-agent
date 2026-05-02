---
name: governed-change-analysis
description: Treat policy, routing, or workflow changes as governed proposals that need approval, not direct mutation.
---

# Governed Change Analysis

Use this skill when the request changes how Cory or the control plane should behave.

Rules:
- Frame the request as a proposal to governance, workflow, or policy.
- Identify impact, risk, and what explicit human approval is required.
- Do not act as if the new behavior is already canonical.

Output expectations:
- Summarize the proposed change in review-ready language.
- Explain who or what would be affected.
- Surface unresolved policy questions when the requested behavior is underspecified.
