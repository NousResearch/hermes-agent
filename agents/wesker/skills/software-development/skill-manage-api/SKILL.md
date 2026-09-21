---
name: skill-manage-api
description: "Quick-reference for skill_manage tool: actions, required params, and common pitfalls. Load before calling skill_manage to avoid rejected calls."
version: 1.0.0
author: Ops Lead
license: MIT
metadata:
  hermes:
    tags: [skills, skill-manage, tool-api, reference]
    related_skills: [hermes-agent-skill-authoring, kanban-worker]
---

# skill_manage API Quick Reference

## Actions and Their Required Parameters

| Action | Required Params | Optional Params | What It Does |
|---|---|---|---|
| `create` | `name`, `content` | `category` | Creates a new skill under `~/.hermes/skills/[category]/[name]/SKILL.md`. Category optional — defaults to uncategorised. |
| `edit` | `name`, `content` | — | Rewrites the entire SKILL.md. Supply full new content. |
| `patch` | `name`, `old_string`, `new_string` | `file_path`, `replace_all` | Targeted find-and-replace on a skill file. **Omit `file_path` to patch SKILL.md root.** Use `file_path=` for supporting files only. |
| `delete` | `name` | `absorbed_into` | Removes the skill. Set `absorbed_into=""` to prune; set `absorbed_into="umbrella-name"` when merging content elsewhere. |
| `write_file` | `name`, `file_path`, `file_content` | — | Creates/overwrites a supporting file inside the skill directory. `file_path` must be under: `assets/`, `references/`, `scripts/`, or `templates/`. |
| `remove_file` | `name`, `file_path` | — | Deletes a supporting file from the skill directory. Same subdirectory constraints as `write_file`. |

## Critical: Patch Mode and `file_path`

This is the most common footgun:

```python
# ✅ Patches SKILL.md root — OMIT file_path
skill_manage(action='patch', name='my-skill', old_string='old text', new_string='new text')

# ❌ Rejected: file_path='SKILL.md' is not in allowed subdirectories
skill_manage(action='patch', name='my-skill', file_path='SKILL.md', old_string='old text', new_string='new text')

# ✅ Patches a supporting file — include file_path
skill_manage(action='patch', name='my-skill', file_path='references/guide.md', old_string='old text', new_string='new text')
```

**Rule:** On `patch`, `file_path` is exclusively for files under `assets/`, `references/`, `scripts/`, `templates/`. Omit it entirely to target SKILL.md.

## Common Mistakes

1. **`skill_manage(action='show')`** — Wrong tool. Use `skill_view(name='...')` instead.
2. **`skill_manage(action='write_file')` without `file_path`** — Required. Must include subdirectory (e.g. `references/guide.md`).
3. **`skill_manage(action='patch')` without `new_string`** — Required, even for deletion (set to `""`).
4. **`skill_manage(action='')`** — Action string cannot be empty.
5. **`skill_manage(action='create')` for in-repo skills** — Creates locally-only. In-repo skills (inside `/home/bb/hermes-agent/skills/`) need `write_file` + `git add`.
6. **`old_string` doesn't match** — The API uses fuzzy matching (9 strategies) and suggests alternatives. Read the error's suggestions and pick the closest match, adjusting whitespace or indentation.

## Related

- `kanban-worker` skill — for all kanban tool pitfalls
- `hermes-agent-skill-authoring` skill — for in-repo skill authoring workflow
