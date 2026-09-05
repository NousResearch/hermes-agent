---
name: codebase-onboarding
description: 'Analyze an unfamiliar codebase and generate a structured onboarding guide with architecture map, key entry points, conventions, and a starter guide. Use when joining a new project or exploring a repo for the first time.'
version: 1.0.0
author: Hermes Agent (adapted from ECC)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [onboarding, architecture, codebase, documentation]
    related_skills: [repo-scan, code-tour, blueprint]
---

# Codebase Onboarding

Systematically analyze an unfamiliar codebase and produce a structured onboarding guide. Designed for developers joining a new project or setting up an agent in an existing repo for the first time.

## When to Use

- First time opening a project
- Joining a new team or repository
- "Help me understand this codebase"
- "Onboard me" or "Walk me through this repo"

## How It Works

### Phase 1: Reconnaissance

Gather raw signals without reading every file. Run these in parallel:

1. **Package manifest detection** — `package.json`, `go.mod`, `Cargo.toml`, `pyproject.toml`, `pom.xml`, `build.gradle`
2. **Framework fingerprinting** — `next.config.*`, `vite.config.*`, Django settings, FastAPI main, Rails config
3. **Entry point identification** — `main.*`, `index.*`, `app.*`, `server.*`, `cmd/`, `src/main/`
4. **Directory structure snapshot** — Top 2 levels, ignoring `node_modules`, `vendor`, `.git`, `dist`, `build`
5. **Config and tooling** — `.eslintrc*`, `tsconfig.json`, `Makefile`, `Dockerfile`, `docker-compose*`, `.github/workflows/`
6. **Test structure** — `tests/`, `test/`, `__tests__/`, `*_test.go`, `*.spec.ts`, `pytest.ini`, `jest.config.*`

### Phase 2: Architecture Mapping

Identify:

- **Tech Stack** — Language(s), framework(s), database(s), build tools
- **Module Map** — How code is organized, key directories, dependency flow
- **Entry Points** — Main function, API routes, CLI commands, background jobs
- **Data Flow** — How data moves through the system
- **External Dependencies** — APIs, services, databases, caches

### Phase 3: Onboarding Guide

Write to `docs/onboarding.md`:

```markdown
# [Project Name] — Onboarding Guide

## Quick Start

[Commands to install, build, and run]

## Architecture Overview

[High-level diagram in text, key components]

## Key Directories

| Directory | Purpose |
| --------- | ------- |

## Important Files

[Critical config files, entry points, shared utilities]

## Conventions

[Naming, file organization, commit style, PR process]

## Testing

[Test frameworks, how to run, coverage expectations]

## Common Tasks

[How to add a route, how to add a migration, how to deploy]

## Gotchas

[Non-obvious things that trip up newcomers]
```

## Hermes Integration

- Use `search_files` and `read_file` for reconnaissance
- Use `list_dir` (terminal: `Get-ChildItem` / `ls`) for directory structure
- Use `write_file` to save the onboarding guide
- Use `delegate_task` to run Phase 1 checks in parallel
- Combine with `code-tour` for step-by-step walkthroughs
- Combine with `blueprint` for project planning after onboarding
