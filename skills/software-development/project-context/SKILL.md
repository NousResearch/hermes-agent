---
name: project-context
description: "Use when starting work in a project directory or switching between projects. Auto-detects and loads HERMES.md from the working directory (recursing up to 3 parent levels). Inspired by Claude Code CLAUDE.md."
version: 1.0.0
author: Hermes Agent (adapted from Claude Code project context pattern)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [project, context, HERMES.md, workspace, conventions, configuration]
    related_skills: [plan, compact]
---

# Project Context — Auto-Detect Project-Level Configuration

## Overview

Every project has its own conventions, tools, and structure. Hermes should adapt to each project automatically — just like Claude Code does with CLAUDE.md.

This skill teaches Hermes to detect and load `HERMES.md` from the project root, enabling project-specific behavior without the user having to re-explain conventions every session.

**Core principle:** The working directory defines the project context. Switch directories, switch contexts.

## When to Use

- **Automatically** at the start of every session
- When the user changes working directory (`cd` to another project)
- When you're about to work on project files and haven't loaded project context yet
- When the user says "this project uses..." — that info should go into HERMES.md

## Detection Algorithm

```
function find_hermes_md(workdir):
    search_paths = [
        workdir / "HERMES.md",
        workdir / ".hermes" / "project.md",
        parent(workdir) / "HERMES.md",       # 1 level up
        parent(parent(workdir)) / "HERMES.md", # 2 levels up
        parent(parent(parent(workdir))) / "HERMES.md", # 3 levels up
    ]
    for path in search_paths:
        if file_exists(path):
            return path
    return None
```

### Step 1: Check for HERMES.md

At session start (and whenever workdir changes), check with `read_file`:

```
read_file("HERMES.md")  # in current workdir
```

If it exists, read and apply it.

### Step 2: If Not Found, Search Upward

```
# Check parent directories (search_files for HERMES.md)
search_files("HERMES.md", target="files", path="..")
search_files("HERMES.md", target="files", path="../..")
search_files("HERMES.md", target="files", path="../../..")
```

### Step 3: Apply Project Context

When HERMES.md is found, its contents become project-level directives that guide all work in that directory. It overrides global defaults where specified.

### Step 4: If Not Found, Offer to Create

If no HERMES.md exists in any searched path, and the user seems to be working in a structured project:

```
"I notice this project doesn't have a HERMES.md yet. 
I can create one with the project's conventions, testing 
commands, and structure. This will be auto-loaded in 
future sessions. Create it?"
```

## HERMES.md File Format

A good HERMES.md should contain:

```markdown
# Project Name

## Overview
[One paragraph about what this project is]

## Tech Stack
- Language: [Python 3.12 / TypeScript 5.x / etc.]
- Framework: [FastAPI / React 19 / etc.]
- Database: [PostgreSQL / SQLite / etc.]

## Commands
- Test: `pytest tests/ -q`
- Lint: `ruff check src/`
- Build: `npm run build`
- Run: `python -m uvicorn main:app`

## Conventions
- [Coding style preferences]
- [File naming conventions]
- [Branch naming patterns]

## Key Files
- `src/main.py` — entry point
- `config/settings.yaml` — configuration
- `docs/architecture.md` — architecture overview

## Known Issues
- [Any recurring problems and workarounds]
```

## Common Pitfalls

1. **Not re-loading on directory change** — when the user `cd`s to another project, re-run the detection algorithm.
2. **Loading HERMES.md but ignoring it** — if HERMES.md says "use pytest -n 4", actually use that command.
3. **Overwriting existing HERMES.md** — always read it first, suggest additions, never replace wholesale.
4. **Creating HERMES.md without asking** — always confirm before creating project files.
5. **Recursing too deep** — stop at 3 parent levels. If not found by then, the project likely doesn't need one.

## Verification Checklist

- [ ] HERMES.md found and loaded (or confirmed absent)
- [ ] Project commands match what HERMES.md specifies
- [ ] Coding conventions from HERMES.md are applied
- [ ] Switch detection working: changing dir loads new context
