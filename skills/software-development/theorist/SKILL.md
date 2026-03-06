---
name: theorist
description: Maintain a per-repo THEORY.MD document capturing the operating theory behind the current work — why the system works this way, what's been tried, and where uncertainty remains. Inspired by blader/theorist (MIT, by Siqi Chen).
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Theory, Documentation, Strategic Thinking, Project Understanding, Context]
    homepage: https://github.com/blader/theorist
    related_skills: []
---

# Theorist — Operating Theory Maintenance

Maintain a per-repo `THEORY.MD` document that captures the _operating theory_ behind the current work. Unlike AGENTS.md (static project rules) or MEMORY.md (operational notes), THEORY.MD is a **living strategic narrative** that answers "why does this system work this way?" and "where does uncertainty remain?"

## What THEORY.MD Contains

**Problem thesis**: What problem is being solved and why it matters — the structural reason the problem exists, not just symptoms.

**Operating theory**: Current mental model of how the system works, what the leverage points are, what has been tried and learned.

**Systematic strategy**: The higher-order approach — not tasks but principles connecting the changes.

**Key discoveries and pivots**: Moments where understanding shifted, what the old theory was, what broke it, what replaced it.

**Open questions**: What's still unknown, where the theory might be wrong, what would change the approach.

## What THEORY.MD Is NOT

- ❌ Not a changelog (no timestamped entries — holistic rewrites only)
- ❌ Not a plan/todo list (no checkboxes or step-by-step instructions)
- ❌ Not a postmortem (present tense of ongoing work)
- ❌ Not a status report (no "today I did X")

## When to Use This Skill

Load this skill when:
- Starting a new project or returning to an existing one
- The user asks "what's the current theory?" or "update the operating theory"
- You need to understand the strategic context behind code decisions
- Significant understanding shifts occur (root cause found, strategy pivot, new uncertainty)
- After completing investigate/implement/verify loops

## Session Behavior

**At session start**: Read THEORY.MD silently if it exists, orient to the work.

**During work**: Update when _understanding_ shifts (root cause found, strategy pivot, new uncertainty), NOT on every code change.

**Update cadence**: 
- After each investigate/implement/verify loop
- Every ~10 minutes of active work
- When 2-3 learnings accumulate
- Trivial sessions (one-liner fix, config change) — no-op the document

**Update style**: Holistic rewrites of relevant sections, keeping the full document coherent. Never append. Rewrite the entire document or relevant sections to maintain narrative flow.

## Practical Constraints

- One THEORY.MD per repo, at repo root (`THEORY.MD` or `theory.md`)
- Max ~200 lines — tighten prose if longer
- Rewrite holistically, never append
- Tone: thoughtful engineer explaining mental model to a peer, direct, specific, no filler

## How to Maintain THEORY.MD

### Reading THEORY.MD

```python
# Check if THEORY.MD exists at repo root
theory_path = Path.cwd() / "THEORY.MD"
if not theory_path.exists():
    theory_path = Path.cwd() / "theory.md"  # case-insensitive fallback

if theory_path.exists():
    content = read_file(file_path=str(theory_path))
    # Use this to orient yourself to the project's strategic context
```

### Updating THEORY.MD

When understanding shifts:

1. **Read the current THEORY.MD** (if it exists)
2. **Identify which sections need revision** based on new understanding
3. **Rewrite those sections holistically** — don't append, synthesize
4. **Maintain narrative coherence** — the document should read as a unified explanation
5. **Keep it concise** — max ~200 lines total

Example update triggers:
- Found root cause of a bug → update "Operating theory" section
- Discovered a better approach → note the pivot in "Key discoveries"
- New uncertainty emerges → add to "Open questions"
- Strategy changes → rewrite "Systematic strategy"

### Creating THEORY.MD

If THEORY.MD doesn't exist and the work warrants it:

1. **Assess the project**: Is this sustained development work (not a one-off fix)?
2. **Create initial theory**: Write sections covering:
   - Problem thesis
   - Operating theory (current understanding)
   - Systematic strategy
   - Open questions
3. **Place at repo root**: `THEORY.MD` (case-insensitive matching supported)

## Example Structure

```markdown
# Operating Theory

## Problem Thesis

[What problem is being solved and why it matters — structural reasons, not symptoms]

## Operating Theory

[Current mental model: how the system works, leverage points, what's been tried]

## Systematic Strategy

[Higher-order approach: principles connecting changes, not task lists]

## Key Discoveries and Pivots

[Understanding shifts: old theory → what broke it → new theory]

## Open Questions

[What's unknown, where theory might be wrong, what would change approach]
```

## Integration with Other Context Files

THEORY.MD complements other context files:

| File | Purpose | Author |
|------|---------|--------|
| AGENTS.md | Project rules, conventions, build commands | Human |
| SOUL.md | Agent persona, tone, identity | Human |
| .cursorrules | IDE-specific rules | Human |
| MEMORY.md | Operational notes, facts | Agent |
| **THEORY.MD** | **Strategic understanding, mental model** | **Agent** |

AGENTS.md tells you HOW to work; THEORY.MD tells you WHY the system is shaped this way.

## Notes

- THEORY.MD is auto-loaded by Hermes at session start (if present), so it's always available as context
- The document evolves with understanding — old theories aren't deleted, they're noted as pivots
- Quality depends on synthesis — focus on coherent narrative, not exhaustive detail
- Some teams may want THEORY.MD tracked in git, others may prefer .gitignore — both are valid

## References

- Original concept: [blader/theorist](https://github.com/blader/theorist) (MIT, v1.3.0, by Siqi Chen)
- Issue: [#381](https://github.com/NousResearch/hermes-agent/issues/381)
