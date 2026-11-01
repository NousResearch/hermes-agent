---
name: create-readme
license: MIT
version: 1.2.1
author: afonsoft
platforms: [linux, macos, windows]
description: "Generate or update README.md and CHANGELOG.md."
metadata:
  hermes:
    tags: [Documentation, README, Changelog]
  version: "1.2.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Create README and CHANGELOG

## Overview

Transforms a raw repository into a well-documented project by generating professional `README.md` and `CHANGELOG.md` files based on empirical evidence from the codebase, git history, and configuration files. Every claim in the generated documentation must be traceable to a file, commit, or config in the repository.

## When to Use

- When a project lacks a README or CHANGELOG.
- When existing documentation is outdated, incomplete, or inconsistent.
- Before shipping a new version to ensure the changelog is up to date.
- When restructuring a project and the existing documentation no longer reflects the architecture.
- When onboarding new contributors and the README doesn't answer "what is this, how do I run it, how do I test it?"

- User asks or mentions this skill in English (e.g., "use /create-readme", "run create-readme").
- O usuário pede ou menciona esta skill em português (ex.: "use /create-readme", "execute create-readme").

## When NOT to Use

- **API reference docs** — use OpenAPI/Swagger, TypeDoc, Sphinx, or docfx instead.
- **License files** — copy the appropriate license text directly (MIT, Apache-2.0, etc.).
- **Internal code documentation** — use docstrings, JSDoc, TSDoc, or XML doc comments.
- **Architecture diagrams** — use the `architecture` skill (which routes to `drawio-architecture`, `mermaid-architecture`, and optional `archify`) for visual and Markdown system design.
- **Agent harness setup** — use `create-agent-harness` for CLAUDE.md/AGENTS.md and skill scaffolding.

## Process

### Phase 1: Discovery (Evidence Gathering)

Before writing a single line, analyze the target repository:

**Structure analysis:**
- Root directory tree and top-level files (`ls -la`, `find . -maxdepth 2 -type f`)
- Identify entry points (`main.ts`, `Program.cs`, `__main__.py`, `index.js`)
- Detect monorepo vs. single-package layout

**Stack detection:**
- **Node.js**: `package.json` (name, version, scripts, dependencies, engines)
- **.NET**: `*.csproj`, `*.sln` (TargetFramework, PackageReferences, SDK version)
- **Python**: `pyproject.toml`, `setup.py`, `requirements.txt`, `Pipfile`
- **Java**: `pom.xml`, `build.gradle` (groupId, artifactId, Java version)
- **Go**: `go.mod` (module path, Go version, requires)
- **Rust**: `Cargo.toml` (name, edition, dependencies)
- **Docker**: `Dockerfile`, `docker-compose.yml` (base image, exposed ports, services)
- **CI/CD**: `.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`, `azure-pipelines.yml`

**History analysis:**
- `git log --oneline -n 50` — recent features and fixes
- `git tag --sort=-creatordate | head -10` — recent releases
- `git log --since="last tag" --oneline` — unreleased changes

**Existing docs:**
- Current `README.md` and `CHANGELOG.md` (if any) — reuse valid content
- `docs/` directory — reference but don't duplicate
- `LICENSE` file — extract license type

**Requirement**: Output a "Discovery Summary" containing:
```markdown
## Discovery Summary
- **Stack**: [languages, frameworks, versions]
- **Architecture**: [monorepo/single, layers, patterns]
- **CI/CD**: [platforms, pipelines, badges available]
- **Entry points**: [main files]
- **Test command**: [how to run tests]
- **Recent releases**: [last 3 tags]
- **Unreleased changes**: [commits since last tag]
- **Gaps**: [what's missing from current docs]
```

Wait for confirmation before proceeding to Phase 2.

### Phase 2: README.md Authoring

Generate the README following this strict order. Skip sections where no evidence exists — do not invent content.

#### 2.1 Title & Badges
- Project name from `package.json`/`*.csproj`/`pyproject.toml` or directory name
- Badges: CI status (from workflow file), license (from LICENSE file), version (from package manifest), language coverage (if configured)
- Badge format: `[![Name](url)](link)`

#### 2.2 Project Description
- **One-sentence summary** (what it does, for whom)
- **Rich paragraph** (functional and strategic overview)
- Evidence: derive from code comments, existing docs, commit messages — never guess

#### 2.3 Repository Structure
```text
project-root/
├── src/              # Source code
├── tests/            # Test suite
├── docs/             # Documentation
├── .github/workflows # CI/CD pipelines
└── package.json      # Node.js manifest
```
- One-line description per top-level directory
- Only include directories that actually exist

#### 2.4 Tech Stack
| Layer | Technology | Version |
|-------|-----------|---------|
| Language | TypeScript | 5.x |
| Framework | Next.js | 15.x |
| Database | PostgreSQL | 16 |
| CI | GitHub Actions | — |

- Versions from package manifests, not guesses
- Include runtime requirements (Node version, Python version, .NET version)

#### 2.5 Architecture
- Layers and patterns (Clean Architecture, DDD, MVC, microservices)
- Mermaid diagram if the system has 3+ interacting components
- Evidence: derive from directory structure and dependency graph

#### 2.6 System Flow
- Mermaid sequence/flow diagram for the main use case
- Or textual step-by-step description for simple systems
- Only include if the flow is non-obvious from the code

#### 2.7 Getting Started
```bash
# Prerequisites
node >= 20.x

# Install
npm install

# Configure
cp .env.example .env  # then edit values

# Run
npm run dev
```
- Prerequisites with specific versions
- Environment variables (names only, never values — link to `.env.example`)
- Install and run commands from `scripts` in package manifest or Makefile

#### 2.8 Tests & Coverage
```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
npm run test:coverage # Coverage report
```
- Commands from package manifest scripts or Makefile
- Coverage threshold if configured (`.nycrc`, `jest.config`, `coverlet`)
- Badge if coverage reporting is set up

#### 2.9 Business & Technical Views
- **Business**: strategic goals, target users, problem solved
- **Technical**: key design decisions, trade-offs, constraints
- Only include if evidence exists in docs, comments, or commit messages

#### 2.10 License & Status
- License type from `LICENSE` file
- Project status: active, maintained, experimental, deprecated (from recent commit activity)
- Link to full license text

#### 2.11 Links
- Internal references (`docs/`, `CHANGELOG.md`, contributing guide)
- External references (homepage, demo, API docs) if they exist

#### 2.12 Language routing and `docs/` structure

Create a bilingual documentation layout:

- `README.md` — English (en-us), always the primary landing page.
- `README.pt-br.md` — Portuguese (pt-br) translation of `README.md`.
- `docs/en/` — auxiliary English documents generated from repository evidence.
- `docs/pt/` — same documents in pt-br.

Rules for `docs/`:

- Place every auxiliary document in both `docs/en/` and `docs/pt/`.
- Each README links to its counterpart at the top: `[Português](README.pt-br.md)` in `README.md`; `[English](README.md)` in `README.pt-br.md`.
- Each README links to `docs/en/` or `docs/pt/` for deeper docs.
- Do not duplicate `README.md` content inside `docs/`. Put extended/auxiliary content there (e.g., `ARCHITECTURE.md`, `CONTRIBUTING.md`, `API.md`, `INSTALL.md`).
- Generate a doc only when evidence exists in the repo. Never invent auxiliary docs.
- Keep the same filename in both folders (e.g., `docs/en/CONTRIBUTING.md` and `docs/pt/CONTRIBUTING.md`).

Evidence-based docs to consider:

- `ARCHITECTURE.md` — if architecture patterns or diagrams were identified.
- `CONTRIBUTING.md` — if `.github/CONTRIBUTING.md` or commit conventions exist.
- `API.md` — if the project exposes a REST/GraphQL API (link to OpenAPI/Swagger if present).
- `INSTALL.md` — if setup has platform-specific steps beyond the README.

**Rules:**
- Reuse existing content where applicable — don't rewrite what's already correct
- Only include sections where concrete evidence exists
- `README.md` is always en-us; `README.pt-br.md` is the pt-br translation
- Auxiliary docs live under `docs/en/` and `docs/pt/` and must not duplicate README content
- Never hardcode secrets, API keys, or environment variable values

### Phase 3: CHANGELOG.md Authoring

Follow the [Keep a Changelog](https://keepachangelog.com/) and [SemVer](https://semver.org/) standards:

#### 3.1 Structure
```markdown
# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- New feature X (commit abc123)

### Fixed
- Bug Y in module Z (commit def456)

## [1.2.0] - 2025-01-15

### Added
- Feature A

### Changed
- Updated dependency B to v2.0

[Unreleased]: https://github.com/user/repo/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/user/repo/releases/tag/v1.2.0
```

#### 3.2 Categories
- `Added` — new features
- `Changed` — changes in existing functionality
- `Deprecated` — soon-to-be removed features
- `Removed` — removed features
- `Fixed` — bug fixes
- `Security` — vulnerability fixes

#### 3.3 Source
- Populate `[Unreleased]` by analyzing `git log --since="last tag" --oneline`
- Map commit prefixes (`feat:`, `fix:`, `breaking:`) to changelog categories
- For past versions, use `git log v1.1.0..v1.2.0 --oneline` between tags
- If no tags exist, create `[Unreleased]` from the last 50 commits

#### 3.4 Linking
- Link version numbers to GitHub compare URLs: `https://github.com/{owner}/{repo}/compare/v{prev}...v{curr}`
- Link tags to release pages: `https://github.com/{owner}/{repo}/releases/tag/v{version}`

### Phase 4: Delivery & Verification

#### 4.1 Branching
Create a dedicated branch:
```bash
git checkout -b feature/{YYYYMMDD}-readme-changelog
```

#### 4.2 Linting
- Run a Markdown linter if available (`markdownlint`, `remark`, `vale`)
- Check for broken internal links
- Verify Mermaid diagram syntax (if included)

#### 4.3 Commit
Use Conventional Commits:
```bash
git commit -m "docs(readme): update README and CHANGELOG

- Add tech stack table from package.json analysis
- Add getting started section with verified commands
- Populate CHANGELOG [Unreleased] from git log since v1.2.0

Generated with [Devin](https://devin.ai)"
```

#### 4.4 Reporting
Provide a summary of changes:
```markdown
## Summary
- **README.md**: [created/updated] — en-us primary, added sections X, Y, Z
- **README.pt-br.md**: [created/updated] — pt-br translation
- **docs/en/** and **docs/pt/**: [created/updated] — auxiliary docs by language
- **CHANGELOG.md**: [created/updated] — added [Unreleased] with N entries
- **Evidence**: all claims traced to files/commits in Discovery Summary
```
Do not open the PR automatically — let the human reviewer decide.

## Common Mistakes

| Mistake | Impact | Fix |
|----------|--------|-----|
| **Inventing info** | README claims a feature that doesn't exist | Every claim must trace to a file, commit, or config |
| **Generic templates** | README doesn't reflect the actual architecture | Use Discovery Summary to tailor every section |
| **Ignoring history** | Changelog doesn't match git commits | Use `git log` between tags as the source of truth |
| **Hardcoded secrets** | API keys or passwords in README | Use env var names only; link to `.env.example` |
| **Stale badges** | CI badge points to wrong workflow | Verify badge URL matches actual workflow filename |
| **Manual PRs** | PR opened without review | Summarize changes first; let human open the PR |
| **Wrong language** | README in English for a pt-BR project | Match the language of existing documentation |
|| **Only one language** | Repository misses pt-br users | Always generate `README.pt-br.md` and `docs/pt/` alongside the English versions |
|| **Docs duplicate README** | `docs/en/README.md` or `docs/pt/README.md` copies the README | Use `docs/` only for extended/auxiliary documents, not another README |

## Verification Checklist

- [ ] Discovery Summary was produced and confirmed before generation
- [ ] README includes all required sections based on available evidence
- [ ] No section contains invented or unverifiable information
- [ ] Tech stack versions match package manifests exactly
- [ ] Getting Started commands were verified against `scripts`/Makefile
- [ ] No secrets, API keys, or environment variable values are present
- [ ] CHANGELOG follows the "Keep a Changelog" format with all 6 categories
- [ ] CHANGELOG `[Unreleased]` section matches `git log` since last tag
- [ ] Version links point to valid compare/release URLs
- [ ] Mermaid diagrams (if any) have valid syntax
- [ ] Branch naming follows the project's convention
- [ ] Commit message follows Conventional Commits
- [ ] `README.pt-br.md` is generated as a pt-br translation of `README.md`
- [ ] `docs/en/` and `docs/pt/` exist for auxiliary docs
- [ ] Language switcher links connect `README.md` and `README.pt-br.md`
- [ ] No `README.md` content is duplicated inside `docs/`

## References

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Shields.io](https://shields.io/) — badge generation
- [makeareadme.com](https://www.makeareadme.com/) — README best practices
