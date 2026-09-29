---
name: hermes-find-skills
description: "List available Hermes skills by category or keyword."
version: 1.0.0
author: Hermes
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, discovery, search, hermes-agent]
    related_skills: [hermes-agent-skill-authoring, hermes-skill-authoring-workflow]
---

# Hermes Find Skills

Discover and list available Hermes skills from both in-repo and user-local skill directories. Filter by category, tag, or keyword. Provides structured output for programmatic use.

## Overview

This skill wraps the `skills_list` tool to provide a consistent interface for discovering Hermes skills. It searches both the in-repo skills tree (`skills/`) and user-local skills (`~/.hermes/skills/`), returning structured metadata including descriptions, tags, and related skills.

Does NOT install, create, or modify skills — use `hermes-agent-skill-authoring` for that.

## When to Use

- "What skills are available for X?"
- "List all skills in category Y"
- "Find skills tagged with Z"
- "Show me skills related to skill-name"
- Need to programmatically discover skills before delegating

## Prerequisites

- Active Hermes session (skill loader initialized)
- No additional installs or credentials required

## How to Run

Invoke through Hermes tools:

1. **List all skills** — `skills_list()`
2. **Filter by category** — `skills_list(category="software-development")`
3. **Get full skill details** — `skill_view(name="skill-name")`

## Quick Reference

| Action | Tool | Parameters |
|--------|------|------------|
| List all | `skills_list` | `{}` |
| By category | `skills_list` | `{"category": "devops"}` |
| View details | `skill_view` | `{"name": "skill-name"}` |
| View linked file | `skill_view` | `{"name": "skill", "file_path": "references/x.md"}` |

## Procedure

### Step 1: List All Available Skills

```python
skills_list()
```

Returns all skills across categories with name, description, category.

**Completion:** JSON returned with `skills` array.

### Step 2: Filter by Category (Optional)

```python
skills_list(category="software-development")
```

Returns only skills in that category.

**Completion:** Filtered list returned.

### Step 3: Get Full Skill Details

```python
skill_view(name="hermes-agent-skill-authoring")
```

Returns full SKILL.md content, frontmatter, linked files, metadata.

**Completion:** Complete skill object returned.

### Step 4: Access Linked References/Scripts

```python
skill_view(name="hermes-agent-skill-authoring", file_path="references/template.md")
```

Returns content of referenced file.

**Completion:** File content returned.

## Pitfalls

1. **Current session cache** — New skills added this session won't appear until next session. Read file directly if needed.
2. **Category names** — Must match exact directory names under `skills/` (e.g., `mlops/evaluation` not `mlops-eval`).
3. **Plugin skills** — Use `plugin:skill` format (e.g., `superpowers:writing-plans`) for plugin-provided skills.
4. **No search by tag** — `skills_list` only filters by category. Use `skill_view` + client-side filter for tags.
5. **User-local vs in-repo** — Both trees merged in results. Check `path` field to distinguish.

## Verification

Run `skills_list()` and confirm:
- Returns `skills` array with at least one entry
- Each entry has `name`, `description`, `category`
- Known skills (e.g., `hermes-agent-skill-authoring`) appear in results