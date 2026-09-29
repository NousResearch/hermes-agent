---
name: hermes-skill-authoring-workflow
description: "End-to-end workflow for authoring Hermes skills in-repo."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, authoring, workflow, skill-md, hermes-agent]
    related_skills: [hermes-agent-skill-authoring, plan, requesting-code-review]
---

# Hermes Skill Authoring Workflow

End-to-end process for creating production-ready Hermes skills in the repository. Covers source gathering, requirement application, SKILL.md authoring to repo standards, validation, and commit.

## Overview

This workflow captures the disciplined process for authoring a new Hermes skill from scratch: gather every source the user named, apply every requirement/constraint they specified, write a compliant SKILL.md to `skills/<category>/<name>/`, validate against the repo's validator, and commit. It mirrors the exact steps used to create this skill.

**Does NOT cover:** Creating user-local skills via `skill_manage(action='create')` (those go to `~/.hermes/skills/`). This workflow targets the in-repo tree that ships with Hermes.

## When to Use

- User asks to "learn a skill from" a request and save it
- User asks to add a skill "in this branch / repo / commit"
- You need to codify a reusable workflow discovered during a session
- You're committing a skill that should ship with the Hermes package

## Prerequisites

- Active session in the Hermes repo root (`/opt/hermes` or equivalent)
- Write access to `skills/<category>/`
- `git` available for commit
- Python 3 + `yaml` module for local validation

## How to Run

Invoke through Hermes tools:

1. **Gather sources** — `web_extract` for URLs, `read_file`/`search_files` for local paths, `session_search` for conversation history
2. **Author skill** — `write_file` to `skills/<category>/<name>/SKILL.md`
3. **Validate locally** — Python one-liner (see Verification)
4. **Commit** — `terminal` with `git add` + `git commit`

## Quick Reference

| Step | Tool | Target |
|------|------|--------|
| Fetch URL | `web_extract` | `urls: ["https://..."]` |
| Read local file | `read_file` | `path: "path/to/file"` |
| Search repo | `search_files` | `pattern: "..."` |
| Search history | `session_search` | `query: "..."` |
| Write skill | `write_file` | `path: "skills/<cat>/<name>/SKILL.md"` |
| Validate | `terminal` | Python validator one-liner |
| Commit | `terminal` | `git add ... && git commit -m "..."` |

## Procedure

### Step 1: Gather Every Named Source

Collect **all** sources the user referenced — URLs, file paths, directories, "what we just did", pasted notes. Do not stop after the first.

- **URLs** → `web_extract(urls=[...])` (max 5 per call)
- **Local files/dirs** → `read_file(path=...)` or `search_files(target="files", pattern="...")`
- **Conversation history** → `session_search(query="...")`
- **Pasted text** → use as-is

**Completion:** Every source mentioned in the request has been fetched/read and its content is in context.

### Step 2: Extract & Apply All Requirements

Parse the request for **requirements, constraints, focus areas, scope limits, naming preferences**. Treat prose after a source link as binding instructions for that source (e.g., "focus on auth flow, skip deprecated" = gather the URL AND honor the focus).

**Completion:** A checklist of every explicit/implicit requirement extracted from the request.

### Step 3: Survey Peer Skills in Target Category

```bash
ls skills/<category>/
```

Read 2–3 peer `SKILL.md` files (`read_file`) to match tone, structure, frontmatter shape, and section conventions.

**Completion:** Category confirmed, 2+ peers read, structure internalized.

### Step 4: Draft SKILL.md to Repo Path

Write to `skills/<category>/<name>/SKILL.md` via `write_file`.

**Frontmatter (mandatory fields):**
```yaml
---
name: lowercase-hyphenated           # ≤64 chars
description: "Use when <trigger>. <one-line behavior>."  # ≤1024 chars
version: 1.0.0
author: Hermes Agent                 # literal string
license: MIT
platforms: [linux, macos, windows]   # only if OS-bound primitives used
metadata:
  hermes:
    tags: [Capitalized, Relevant, Tags]
    related_skills: [peer-skill-1, peer-skill-2]
---
```

**Body sections (in order, omit only if genuinely empty):**
1. `# <Human Title>` + 2–3 sentence intro (what it does, what it doesn't, key dependencies)
2. `## When to Use` — bullet triggers + "Don't use for:" counter-triggers
3. `## Prerequisites` — exact env vars, installs, credentials
4. `## How to Run` — canonical Hermes-tool invocation
5. `## Quick Reference` — flat command/endpoint list
6. `## Procedure` — numbered steps with copy-paste commands + completion criteria
7. `## Pitfalls` — known limits, rate limits, false alarms
8. `## Verification` — single command/check that proves it worked

**Hermes-tool framing:** Always frame actions through Hermes tools (`terminal`, `read_file`, `write_file`, `search_files`, `patch`, `web_extract`, `web_search`, `vision_analyze`, `browser_navigate`, `delegate_task`, `image_generate`, `text_to_speech`, `cronjob`, `memory`, `skill_view`, `execute_code`). Do NOT name raw shell utils (`cat`, `grep`, `sed`, `curl`).

**Completion:** File written, frontmatter valid, all required sections present, description ≤1024 chars and starts with "Use when".

### Step 5: Local Validation

Run this Python one-liner via `terminal`:

```bash
python3 -c "
import yaml, re, pathlib, sys
p = pathlib.Path('skills/<category>/<name>/SKILL.md')
c = p.read_text()
assert c.startswith('---'), 'missing leading ---'
m = re.search(r'\n---\s*\n', c[3:])
assert m, 'missing closing ---'
fm = yaml.safe_load(c[3:m.start()+3])
assert 'name' in fm and 'description' in fm, 'missing name/description'
assert len(fm['description']) <= 1024, f'description too long: {len(fm[\"description\"])}'
assert len(c) <= 100_000, f'file too large: {len(c)}'
print('OK:', fm['name'], len(fm['description']), 'chars desc,', len(c), 'chars total')
"
```

**Completion:** Script prints `OK: <name> ...` with no assertion errors.

### Step 6: Commit to Repo

```bash
git add skills/<category>/<name>/
git commit -m "feat(skills): add <name> skill

<one-line summary of capability>"
```

**Completion:** `git status` shows clean working tree, new skill committed on active branch.

### Step 7: Note Session Cache Limitation

The current Hermes session's skill loader is initialized at startup. `skill_view`/`skills_list` will NOT see the new skill until a new session starts. This is expected — verify in a fresh session or by reading the file directly.

**Completion:** User informed of cache behavior.

## Pitfalls

1. **Using `skill_manage(action='create')` for in-repo skill** — writes to `~/.hermes/skills/`, not the repo. Use `write_file`.
2. **Leading whitespace before `---`** — validator checks `content.startswith("---")`.
3. **Generic description** — must start with "Use when ..." and describe trigger class, not single task.
4. **Missing author/license/metadata** — not validator-enforced but every peer has it; omitting looks half-finished.
5. **Duplicating a peer** — always `ls skills/<category>/` and read 2–3 peers first.
6. **Expecting current session to see new skill** — it won't; skill loader is cached.
7. **Letting sediment accumulate** — when adding a rule, remove the old wording it replaces.
8. **No-op prose** — "be careful", "be thorough" rarely change behavior; replace with checkable criteria.
9. **Linking user-local skills in `related_skills`** — works for you, breaks for fresh clones. Prefer in-repo links.
10. **Description >1024 chars** — hard validator limit; count before saving.

## Verification

Run the validation one-liner (Step 5) and confirm:

```
OK: <skill-name> <desc-len> chars desc, <total-len> chars total
```

Then verify `git log -1 --oneline` shows the commit.

---

*This workflow was distilled from the actual process used to author the `hermes-agent-skill-authoring` and `hermes-skill-authoring-workflow` skills themselves.*