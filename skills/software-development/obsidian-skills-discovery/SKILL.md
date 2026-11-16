---
name: obsidian-skills-discovery
description: "Discover and use Obsidian skills from kepano repo."
version: 1.0.0
author: Hermes
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [obsidian, skills, discovery, markdown, canvas, cli]
    related_skills: [hermes-agent-skill-authoring, hermes-find-skills]
---

# Obsidian Skills Discovery

Discover and use the 5 agent skills from the kepano/obsidian-skills repository for working with Obsidian vaults: markdown, bases, canvas, CLI, and web extraction.

## Overview

The kepano/obsidian-skills repo provides 5 skills following the Agent Skills spec (agentskills.io). They work with any skills-compatible agent (Claude Code, Codex, OpenCode, Hermes). This skill teaches how to discover, install, and use them.

**Does NOT cover:** Writing Obsidian plugins, general Obsidian usage without skills, or skills from other repos.

## When to Use

- "What Obsidian skills are available?"
- "How do I install obsidian-skills?"
- "Create an Obsidian note with wikilinks/callouts"
- "Make a .canvas or .base file"
- "Search my vault from the command line"
- "Extract clean markdown from a web page"

## Prerequisites

- Agent with skills support (Hermes, Claude Code, Codex, OpenCode)
- Obsidian installed (for CLI skill: Obsidian must be running)
- Node.js + npm (for defuddle: `npm install -g defuddle`)
- Obsidian vault path known

## How to Run

Invoke through Hermes tools:

1. **List skills** — `skills_list(category="obsidian")` (if categorized) or browse repo
2. **View skill** — `skill_view(name="obsidian-markdown")`
3. **Install repo** — `terminal` with `npx skills add https://github.com/kepano/obsidian-skills`
4. **Use skill** — Load skill, then follow its procedure

## Quick Reference

| Skill | Purpose | Key Command/Tool |
|-------|---------|------------------|
| obsidian-markdown | Create/edit .md with wikilinks, callouts, embeds | `skill_view + write_file` |
| obsidian-bases | Create/edit .base views with filters/formulas | `skill_view + write_file` |
| json-canvas | Create/edit .canvas nodes/edges/groups | `skill_view + write_file` |
| obsidian-cli | Read/search/create notes via `obsidian` CLI | `terminal: obsidian search query="..."` |
| defuddle | Extract clean markdown from URLs | `terminal: defuddle parse <url> --md` |

## Procedure

### Step 1: Install the Skills Repo

```bash
npx skills add https://github.com/kepano/obsidian-skills
```

Or for Hermes user-local skills:
```bash
git clone https://github.com/kepano/obsidian-skills ~/.hermes/skills/obsidian-skills
```

**Completion:** Skills appear in `skills_list()` or `~/.hermes/skills/obsidian-skills/skills/`.

### Step 2: Load a Specific Skill

```python
skill_view(name="obsidian-markdown")
```

**Completion:** Full SKILL.md content with references returned.

### Step 3: Follow the Skill's Procedure

Each skill has its own workflow. Examples:

**obsidian-markdown** — Create note:
1. Add frontmatter (title, tags, aliases)
2. Write content with wikilinks `[[Note]]`, embeds `![[Image.png]]`, callouts `> [!note]`
3. Save as `.md` in vault via `write_file`

**json-canvas** — Create canvas:
1. Generate 16-char hex IDs
2. Build nodes array (type: text/file/link/group, with x,y,width,height)
3. Build edges array (fromNode, toNode, optional sides/label)
4. Write `.canvas` JSON via `write_file`

**obsidian-cli** — Search vault:
```bash
obsidian search query="project" limit=10
```

**defuddle** — Extract article:
```bash
defuddle parse https://example.com/article --md -o output.md
```

**Completion:** Target file created / vault operation completed / markdown saved.

### Step 4: Verify in Obsidian

Open the vault in Obsidian and confirm:
- `.md` renders with wikilinks/callouts working
- `.base` opens as interactive view
- `.canvas` opens as visual canvas
- CLI commands reflect in UI
- Extracted markdown is clean

## Pitfalls

1. **CLI needs running Obsidian** — `obsidian` commands fail if app not open
2. **Vault targeting** — Use `vault="Name"` prefix if multiple vaults open
3. **ID collisions** — Canvas/base IDs must be unique (16-char hex)
4. **YAML quoting** — Base formulas/filters need careful quoting (see skill refs)
5. **Defuddle not for .md URLs** — Use WebFetch directly for raw markdown URLs
6. **Skill cache** — New skills need session restart to appear in `skills_list`

## Verification

Run `skills_list()` and confirm all 5 skills appear:
- obsidian-markdown
- obsidian-bases
- json-canvas
- obsidian-cli
- defuddle

Then `skill_view(name="obsidian-markdown")` returns full skill with references.