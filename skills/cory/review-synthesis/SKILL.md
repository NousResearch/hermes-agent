---
name: review-synthesis
description: Produce approval-facing synthesis that helps humans review scope, tradeoffs, and risk without rereading the whole request history.
---

# Review Synthesis

Use this skill when the interpretation is meant to support human review or approval.

Rules:
- Compress the important signal without dropping the decision boundary.
- Highlight risk, impact, and the main tradeoff.
- Keep human-facing language readable in zh-TW.

Output expectations:
- A concise review-ready summary.
- The most important risks or dependencies.
- Clear reasoning for the proposed next workflow state.
