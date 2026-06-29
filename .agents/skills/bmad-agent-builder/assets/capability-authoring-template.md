---
name: capability-authoring
description: How to author, register, and evolve learned capabilities
---

# Capability Authoring

When your owner wants you to learn a new ability, you create a capability together. The mechanics are below; first, the one thing that decides whether the capability is any good.

## Write the destination, not the route

Know your own default. Asked to author a capability, you will script it — numbered steps, question lists, a template with mandatory sections — because elaborate scaffolding feels like diligence and reads like quality. That instinct is the central defect to resist. A script is your imagined transcript of one good session; real sessions diverge from it, and a capability that scripts the path spends your future self's intelligence on compliance instead of the problem.

Write the destination instead. A capability prompt holds four things: the **outcome** (the artifact or change that must exist when it has done its job), the **consumer** (who must act on that outcome, and what they can or cannot be assumed to know), the **bar** (what the consumer needs to be true of it), and the **non-inferables** — what your future self cannot infer on its own: owner specifics worth pulling from MEMORY.md and BOND.md, wiring like paths and formats, and any rule with real consequences behind it. Then stop. The outcome and its consumer imply the process. Do not restate your stance: your persona is already in the room when a capability runs, and it supplies the voice and the relationship — the capability only adds what this ability needs on top.

A complete capability body, not an excerpt:

```text
The outcome is a pitch the owner can deliver tomorrow: claims they can
defend, one through-line, no slide that exists out of fear. You are
stress-testing the argument, not polishing words — wordsmithing comes
last. Push where it is weak: the number that will not survive a
question, the benefit with no evidence, the ask that got buried.
Check MEMORY.md for what this owner's audiences have punished before.
```

Everything a scripted version would add — a pitch-structure walkthrough, a ten-question intake, a slide template — subtracts adaptivity. The owner who arrives with a finished deck gets pressure-testing instead of an intake interview precisely because nothing scripted the opening.

This section is the working standard, synced from the prompt-quality canon. For the full canon — the cut tests, the two-version comparison, the retirement test — load your copy at `references/prompt-quality-canon.md`.

## Capability Types

A capability can take several forms.

### Prompt (default)
A markdown file with guidance on what to achieve. Best for judgment-based tasks where you need flexibility.

```
capabilities/
└── {example-capability}.md
```

### Script
A Python or bash script for deterministic tasks such as calculations, file processing, data transformation, or API calls. Create the script alongside a short markdown file that says when to run it and what to do with the results.

```
capabilities/
├── {example-script}.md          # When to run, what to do with results
└── {example-script}.py          # The actual computation
```

Keep scripts to one job each, have them read and write within the sanctum, and never hardcode paths — accept the sanctum path as an argument.

### Multi-file
A folder with multiple files for a more involved capability, such as a mini-workflow with several steps plus reference material or templates.

```
capabilities/
└── {example-complex}/
    ├── {example-complex}.md     # Main guidance
    ├── structure.md             # Reference material
    └── examples.md              # Examples for tone/format
```

### External Skill Reference
Point to an existing installed skill rather than reinventing it. If you discover a skill that would serve your owner well, suggest it, and always ask before installing.

```markdown
## Learned
| Code | Name | Description | Source | Added |
|------|------|-------------|--------|-------|
| [XX] | Skill Name | What it does | External: `skill-name` | YYYY-MM-DD |
```

## Prompt File Frontmatter

Every capability prompt file carries this frontmatter:

```markdown
---
name: {kebab-case-name}
description: {one line, what this does}
code: {2-letter menu code, unique across all capabilities}
added: {YYYY-MM-DD}
type: prompt | script | multi-file | external
---
```

The body is the capability prompt itself, written to the standard above.

## Creating a Capability (The Flow)

1. Owner says they want you to do something new.
2. Explore what they need through conversation; don't rush to write.
3. Draft the capability and show it to them.
4. Refine based on feedback.
5. Save to `capabilities/` as a file or folder depending on type.
6. Register it in CAPABILITIES.md by adding a row to the Learned table.
7. Register it in INDEX.md by noting the new file under "My Files".
8. Confirm: "I'll remember how to do this next session. You can trigger it with [{code}]."

## Refining and Retiring

When you refine a capability after feedback, update the file in place and log the refinement in the session log. When a capability is no longer useful, remove its row from CAPABILITIES.md but keep the file so the owner can bring it back, and note the retirement in the session log. Whether a capability still earns its place is the canon's retirement test: when it stops beating what you would do bare, retire it rather than patch it.
