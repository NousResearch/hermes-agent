---
name: codebase-exploration
description: Quickly map, understand, and navigate unfamiliar codebases. Use when entering a new project, debugging unfamiliar code, or onboarding to a repository.
version: 1.0.0
author: vominh1919
license: MIT
metadata:
  hermes:
    tags: [codebase, exploration, onboarding, navigation, architecture, reverse-engineering]
    related_skills: [systematic-debugging, writing-plans, test-driven-development]
---

# Codebase Exploration

Rapidly understand any codebase — structure, conventions, data flow, and key patterns — without reading every file.

## When to Use

- Entering an unfamiliar project or repository
- Preparing to fix a bug or add a feature in new territory
- Onboarding to a team's codebase
- Reverse-engineering a library or framework
- Before writing implementation plans for unfamiliar areas

## The Exploration Protocol

Complete each phase before moving to the next. Resist the urge to read random files.

---

## Phase 1: Landscape Survey (5 minutes)

Get the big picture before diving into details.

### 1.1 Project Metadata

```bash
# Read the entry points first
read_file("README.md")
read_file("AGENTS.md")          # If exists — often has contributor guidance
read_file("CONTRIBUTING.md")    # If exists
read_file("pyproject.toml")     # or package.json, Cargo.toml, go.mod
read_file("Makefile")           # or Taskfile.yml
```

### 1.2 Directory Structure

```bash
# Top-level layout (depth 2)
terminal(command="find . -maxdepth 2 -type f | head -80")
terminal(command="find . -maxdepth 2 -type d | sort")

# File type distribution
terminal(command="find . -type f -name '*.py' | wc -l")  # adjust extension
terminal(command="find . -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15")
```

### 1.3 Size and Complexity

```bash
# Lines of code by directory
terminal(command="find . -name '*.py' -not -path '*/\.*' | head -500 | xargs wc -l | sort -rn | head -20")

# Or use pygount if available
terminal(command="pygount --format summary . 2>/dev/null || echo 'pygount not installed'")
```

**Record your findings:**
- Primary language and framework
- Project structure pattern (monorepo, microservice, monolith, library)
- Key directories and their purposes
- Entry points (main files, CLI commands, API routes)

---

## Phase 2: Dependency and Flow Mapping (10 minutes)

### 2.1 Import Graph

Trace how modules connect:

```bash
# Who imports what (Python)
search_files("^from \\w+ import|^import \\w+", path="src/", file_glob="*.py", output_mode="content")

# Who imports a specific module
search_files("from mymodule import|import mymodule", path="src/", file_glob="*.py")

# Dependency chain — find the core
search_files("^from \\.", path="src/", file_glob="*.py", output_mode="count")
```

### 2.2 Configuration and Environment

```bash
# Config files
search_files(target="files", pattern="config|settings|env", file_glob="*.{py,yaml,yml,json,toml,ini}")

# Environment variables
search_files("os\\.environ|os\\.getenv|process\\.env|ENV\\[", path="src/")

# API keys and secrets patterns
search_files("API_KEY|SECRET|TOKEN|PASSWORD", path="src/", file_glob="*.py")
```

### 2.3 Entry Points and Interfaces

```bash
# CLI entry points
search_files("__main__|argparse|click|typer", path="src/", file_glob="*.py")

# API routes
search_files("@app\\.|@router\\.|@api_view|@route", path="src/", file_glob="*.py")

# Test entry points
search_files("def test_", path="tests/", file_glob="test_*.py", output_mode="count")
```

---

## Phase 3: Pattern Recognition (10 minutes)

### 3.1 Code Conventions

```bash
# Naming patterns — are there prefixes, suffixes?
search_files("class \\w+", path="src/", file_glob="*.py", output_mode="content", limit=20)

# Error handling pattern
search_files("except |raise |try:", path="src/", file_glob="*.py", output_mode="content", limit=15)

# Logging pattern
search_files("logger\\.|logging\\.|console\\.log", path="src/", file_glob="*.py", limit=10)
```

### 3.2 Architecture Patterns

Look for these common patterns:

| Pattern | Indicators |
|---------|-----------|
| **Registry** | `registry.register`, `@register`, class decorators |
| **Factory** | `create_*`, `build_*`, `make_*` functions |
| **Observer/Event** | `on_*`, `emit_*`, `subscribe`, `dispatch` |
| **Strategy** | `strategy`, `handler`, `backend`, `provider` |
| **Plugin** | `load_plugin`, `discover`, `entry_points` |
| **Middleware** | `middleware`, `pipeline`, `chain` |

```bash
# Check for common patterns
search_files("registry|register\\(", path="src/")
search_files("factory|create_|build_", path="src/")
search_files("handler|strategy|backend|provider", path="src/")
search_files("middleware|pipeline|chain", path="src/")
```

### 3.3 Testing Patterns

```bash
# Test structure
terminal(command="find tests/ -type f -name '*.py' | head -20")

# Test framework
search_files("import pytest|import unittest|from django.test", path="tests/")

# Fixtures and helpers
search_files("@pytest\\.fixture|def setUp|conftest", path="tests/")

# Mocking pattern
search_files("Mock|patch|mock", path="tests/", file_glob="*.py", output_mode="count")
```

---

## Phase 4: Targeted Deep Dive (when needed)

Once you know WHERE to look, read the relevant files completely.

### 4.1 Trace a Feature

```bash
# Find all references to a concept
search_files("concept_name", path="src/")

# Read the main implementation
read_file("src/module/feature.py")

# Check its tests
search_files("test_.*feature", path="tests/")
```

### 4.2 Follow the Data

```bash
# Where is data created?
search_files("data = |result = |response =", path="src/module/", output_mode="content")

# Where is it transformed?
search_files("def process_|def transform_|def parse_", path="src/")

# Where is it stored/persisted?
search_files("\\.save|\\.write|INSERT INTO|UPDATE .* SET", path="src/")
```

### 4.3 Understand the Build/Deploy

```bash
# CI/CD
read_file(".github/workflows/ci.yml")  # or similar
read_file("Dockerfile")
read_file("docker-compose.yml")

# Build system
read_file("Makefile")  # or Taskfile.yml
terminal(command="grep -E '^[a-z-]+:' Makefile | head -20")
```

---

## Exploration Report Template

After exploration, summarize findings:

```
## Codebase Report: [Project Name]

### Overview
- Language: [primary language]
- Framework: [if any]
- Structure: [monorepo/library/app]
- Entry points: [main files, CLI commands, API routes]

### Architecture
- Pattern: [MVC/hexagonal/plugin-based/etc.]
- Key modules: [list with one-line descriptions]
- Data flow: [how data moves through the system]

### Conventions
- Naming: [snake_case/camelCase, prefix patterns]
- Error handling: [how errors are raised/caught]
- Testing: [framework, patterns, coverage areas]
- Config: [how configuration is managed]

### For Contributors
- Start here: [specific files to read first]
- Test command: [how to run tests]
- Lint/format: [how to check code style]
- Key patterns: [what to follow when adding code]
```

---

## Tips

- **Read README + AGENTS.md first** — they often contain contributor shortcuts
- **Follow imports, not file names** — the actual flow may differ from directory structure
- **Start with tests** — they show how the code is meant to be used
- **Find the registry/config** — it reveals the plugin/module system
- **Check git log** — recent commits show what's actively worked on
- **Look for FIXME/TODO** — they reveal known issues and future plans

```bash
# Quick wins
search_files("TODO|FIXME|HACK|XXX", path="src/")
terminal(command="git log --oneline -20")
terminal(command="git log --diff-filter=A --name-only --oneline -10")
```

## Integration with Other Skills

- **systematic-debugging** — use exploration first to find the right code, then debug
- **writing-plans** — exploration findings feed directly into implementation plans
- **test-driven-development** — understand test patterns before writing new tests
- **github-contributions** — explore a repo before attempting fixes
