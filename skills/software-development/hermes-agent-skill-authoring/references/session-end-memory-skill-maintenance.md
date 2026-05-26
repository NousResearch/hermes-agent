# Session-End Memory + Skill Maintenance

Use when the user asks to review a conversation and update memory/skills, or after a complex session produces durable lessons.

## Goal

Preserve reusable learning without turning the skill library into a flat pile of one-session artifacts.

## Required order

1. **Memory first for user facts/preferences**
   - Save who the user is, durable preferences, and expectations about how the agent should behave.
   - Keep it compact; replace an existing related memory if the memory store is near its limit.
   - Do not save one-off task progress, PR numbers, issue numbers, or temporary artifacts.

2. **Patch currently-loaded skills first**
   - Look at skills loaded/consulted during the session.
   - If one governs the durable lesson, patch that skill before searching for another home.
   - Frustration/correction about style or approach is a skill signal, not only a memory signal.

3. **Use class-level owners and references**
   - Prefer umbrella/router or executor owners over new top-level skills.
   - Put session-specific detail, audit notes, API excerpts, and postmortems into `references/` under an owner skill.
   - Add a one-line pointer in `SKILL.md` so future agents find the support file.

4. **Create a new skill only as a last resort**
   - The trigger must be durable and distinct.
   - The tool/workflow surface must differ materially from existing owners.
   - The name must describe a class of work, not today’s issue, PR, error string, or codename.

## What to avoid saving as skills

Do not harden transient setup failures or environment state into durable rules:

- missing binaries or unconfigured credentials
- post-migration path mismatches
- “tool X is broken” negative claims
- one-off task narratives
- resolved retry-only failures

If a setup issue revealed a durable fix, save the fix under the relevant setup/troubleshooting skill, not a blanket refusal.

## Output shape

End with a short report:

- Memory: added/replaced/skipped + why.
- Skills: patched/wrote support file/skipped + why.
- Mention overlapping skills if noticed; do not perform broad consolidation unless it was the task.
