---
name: capability-authoring
description: Guide for creating and evolving learned capabilities
---

# Capability Authoring

When your owner wants you to learn a new ability, you create a capability together. This guide tells you how to write, format, and register it. The quality bar for the prompt body lives in the prompt-quality canon, which your "Author to the standard" standing order has you load before you write. The shipped copy is `references/prompt-quality-canon.md`. This guide points at the canon rather than restating it, so the standard cannot drift.

## Capability Types

A capability can take several forms:

### Prompt (default)
A markdown file with guidance on what to achieve. Best for judgment-based tasks where you need flexibility — brainstorming, analysis, coaching, review.

```
capabilities/
└── blog-ideation.md
```

### Script
A Python or bash script for deterministic tasks — calculations, file processing, data transformation, API calls. Create the script alongside a short markdown file that describes when and how to use it.

```
capabilities/
├── weekly-stats.md          # When to run, what to do with results
└── weekly-stats.py          # The actual computation
```

### Multi-file
A folder with multiple files for complex capabilities — mini-workflows with multiple steps, reference materials, templates.

```
capabilities/
└── pitch-builder/
    ├── pitch-builder.md     # Main guidance
    ├── structure.md         # Pitch structure reference
    └── examples.md          # Example pitches for tone
```

### External Skill Reference
Point to an existing installed skill rather than reinventing it. If you discover a skill that would serve your owner well, suggest it — but always ask before installing.

```markdown
## Learned
| Code | Name | Description | Source | Added |
|------|------|-------------|--------|-------|
| [PR] | Create PRD | Product requirements | External: `bmad-create-prd` | 2026-03-25 |
```

## Prompt File Format

Every capability prompt file should have this frontmatter:

```markdown
---
name: {kebab-case-name}
description: {one line — what this does}
code: {2-letter menu code, unique across all capabilities}
added: {YYYY-MM-DD}
type: prompt | script | multi-file | external
---
```

Author the body against the canon you loaded. A capability body usually carries the outcome you want, the context that constrains it (preferences and domain knowledge), how to draw on MEMORY.md and BOND.md to personalize, and what to capture in the session log after use. Hold each of those to the canon's tests rather than to a rule restated here.

## Creating a Capability (The Flow)

1. Owner says they want you to do something new
2. Explore what they need through conversation — don't rush to write
3. Draft the capability prompt and show it to them
4. Refine based on feedback
5. Save to `capabilities/` (file or folder depending on type)
6. Update CAPABILITIES.md — add a row to the Learned table
7. Update INDEX.md — note the new file under "My Files"
8. Confirm: "I'll remember how to do this next session. You can trigger it with [{code}]."

## Scripts

When a capability needs deterministic logic (math, file parsing, API calls), write a script:

- **Python** preferred for portability
- Keep scripts focused — one job per script
- The companion markdown file says WHEN to run the script and WHAT to do with results
- Scripts should read from and write to files in the sanctum
- Never hardcode paths — accept sanctum path as argument

## Refining Capabilities

Capabilities evolve. After use, if the owner gives feedback:

- Update the capability prompt with refined context
- Add to the "Owner Preferences" section if one exists
- Log the refinement in the session log

A capability that's been refined 3-4 times is usually excellent. The first draft is rarely the best.

## Retiring Capabilities

Whether a capability still earns its place is a canon question, so apply the canon's retirement test rather than a rule restated here. When it no longer earns its place:

- Remove its row from CAPABILITIES.md
- Keep the file (don't delete — the owner might want it back)
- Note the retirement in the session log
