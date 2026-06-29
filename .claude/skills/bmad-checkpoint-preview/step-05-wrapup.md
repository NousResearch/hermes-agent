# Step 5: Wrap-Up

Display: `Orientation → Walkthrough → Detail Pass → Testing → [Wrap-Up]`

## Follow Global Step Rules in SKILL.md

## PROMPT FOR DECISION

```
---

Review complete. What's the call on this {change_type}?
- **Approve** — ship it (I can help with interactive patching first if needed)
- **Rework** — back to the drawing board (revert, revise the spec, try a different approach)
- **Discuss** — something's still on your mind
```

HALT — do not proceed until the user makes their choice.

## ACT ON DECISION

- **Approve**: Acknowledge briefly. If the human wants to patch something before shipping, help apply the fix interactively. If reviewing a PR, offer to approve via `gh pr review --approve` — but confirm with the human before executing, since this is a visible action on a shared resource.
- **Rework**: Ask what went wrong — was it the approach, the spec, or the implementation? Help the human decide on next steps (revert commit, open an issue, revise the spec, etc.). Help draft specific, actionable feedback tied to `path:line` locations if the change is a PR from someone else.
- **Discuss**: Open conversation — answer questions, explore concerns, dig into any aspect. After discussion, return to the decision prompt above.

## On Complete

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete`

If the resolved `workflow.on_complete` is non-empty, follow it as the final terminal instruction before exiting.
