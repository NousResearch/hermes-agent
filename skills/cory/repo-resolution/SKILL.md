---
name: repo-resolution
description: Resolve project or repository mapping only when there is enough evidence; otherwise preserve uncertainty.
---

# Repo Resolution

Use this skill when the request or event may belong to a specific project or repository.

Rules:
- Prefer deterministic evidence over fuzzy guesses.
- If multiple mappings are plausible, say so and ask for clarification.
- Never invent project ids or repo ids.
- Keep the interpretation useful even when exact routing is unresolved.

Output expectations:
- State whether routing appears clear or ambiguous.
- Explain the evidence behind any proposed mapping.
