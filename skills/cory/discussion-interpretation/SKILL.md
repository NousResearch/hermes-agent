---
name: discussion-interpretation
description: Turn ambiguous PM or CTO discussion into a concrete interpreted request, preserving uncertainty instead of fabricating certainty.
---

# Discussion Interpretation

Use this skill when the input is a human discussion, not yet a precise task.

Rules:
- Identify the real intent behind the discussion before proposing workflow state.
- Separate observed facts from inferred intent.
- Preserve ambiguity explicitly when the project, repo, scope, or action is still unclear.
- Prefer a small set of high-signal clarification prompts over many low-value questions.

Output expectations:
- Summarize the request in concise zh-TW.
- Propose a request type that reflects the real next workflow shape.
- Explain the rationale in machine-readable form so the control plane can preserve context.
