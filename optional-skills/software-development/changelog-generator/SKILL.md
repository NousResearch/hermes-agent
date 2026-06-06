---
name: changelog-generator
description: |
  Generate a structured CHANGELOG.md from git history — categorizes commits
  into features, bug fixes, breaking changes, and more. Works with any repo,
  any language, any version scheme.
version: 0.1.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
category: software-development
triggers:
  - "generate changelog"
  - "generate changelog for [version]"
  - "update changelog"
  - "write changelog since [tag]"
  - "changelog from [tag] to [tag]"
  - "what changed in [version]"
  - "release notes for [version]"
toolsets:
  - terminal
  - file
metadata:
  hermes:
    tags: [Changelog, Git, Release, Documentation, Versioning, Automation]
    related_skills: [code-wiki, git-workflow]
---

# Changelog Generator

Generate a clean, structured `CHANGELOG.md` from git history. Reads commit
messages, groups them by type (features, fixes, breaking changes, etc.), and
writes a [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)-compliant
markdown file — ready to publish with your release.

Works with any repo, any language. No external services. Uses only `terminal`
and `file` tools.

---

## When to Use

- User says "generate changelog", "update CHANGELOG.md", "write release notes"
- User points at a version tag or commit range and asks what changed
- Before cutting a release and the changelog is stale or missing

Do NOT use for:
- Single-commit summaries — just describe the commit directly
- Project roadmaps or future plans — that's a different document
- Repos with no git history (nothing to read from)

---

## Prerequisites

- `git` on PATH and the working directory must be inside a git repo.
- No env vars, no external services, no extra dependencies.

---

## Quick Reference

| Step | Action |
|---|---|
| 1 | Detect repo root and existing CHANGELOG.md |
| 2 | Resolve version range (tag-to-tag, tag-to-HEAD, or full history) |
| 3 | Fetch and parse commits in range |
| 4 | Categorize commits by type |
| 5 | Group breaking changes separately |
| 6 | Write / prepend the new section to CHANGELOG.md |
| 7 | Report output path and summary to user |

---

## Procedure

### Step 1 — Resolve repo and version range

```bash
# Confirm we're in a git repo
git rev-parse --show-toplevel

# List available tags (newest first)
git tag --sort=-version:refname | head -20
```

Ask the user for the target version if not provided (e.g. `v1.2.0`). Determine
the range:

- **Tag to tag:** `git log v1.1.0..v1.2.0`
- **Tag to HEAD:** `git log v1.1.0..HEAD`
- **Last N commits:** `git log -N`
- **Full history:** `git log` (warn the user this may be large)

If the repo uses no tags, default to the last 50 commits.

### Step 2 — Fetch commits

```bash
git log v1.1.0..v1.2.0 \
  --pretty=format:"%H|%s|%an|%ae|%ad" \
  --date=short
```

Each line gives: `hash|subject|author_name|author_email|date`

Also fetch the merge commits to detect PR numbers:
```bash
git log v1.1.0..v1.2.0 --merges --pretty=format:"%s"
```

### Step 3 — Categorize commits

Parse each commit subject using these prefix rules (Conventional Commits
compatible, but also handles non-prefixed repos):

| Category | Prefixes / Patterns |
|---|---|
| **Breaking Changes** | `!` after type (`feat!`, `fix!`), or `BREAKING CHANGE:` in body |
| **Added** | `feat:`, `feat(...):`  , `add:`, `new:` |
| **Fixed** | `fix:`, `fix(...):`  , `bugfix:`, `patch:` |
| **Changed** | `refactor:`, `change:`, `update:`, `improve:`, `perf:` |
| **Removed** | `remove:`, `delete:`, `drop:` |
| **Security** | `security:`, `sec:`, commits mentioning CVE/vuln/injection |
| **Deprecated** | `deprecate:`, `deprecated:` |
| **Documentation** | `docs:`, `doc:` |
| **Tests** | `test:`, `tests:` |
| **Chores** | `chore:`, `ci:`, `build:`, `release:` |
| **Other** | Everything that doesn't match above |

For non-prefixed repos, fall back to keyword scanning of the subject:
- Contains "fix", "bug", "error", "crash" → Fixed
- Contains "add", "new", "feature", "implement" → Added
- Contains "remove", "delete", "drop" → Removed
- Contains "update", "refactor", "improve", "perf" → Changed
- Contains "doc", "readme", "comment" → Documentation
- Everything else → Other

Strip the prefix from the displayed line (show `Add dark mode support` not
`feat: Add dark mode support`).

### Step 4 — Detect breaking changes

A commit is a breaking change if:
- Subject contains `!` after the type: `feat!:` or `feat(scope)!:`
- Commit body or footer contains `BREAKING CHANGE:` (run
  `git log --format="%B" <hash>` for the full body)
- Subject explicitly mentions "breaking", "incompatible", "removed API",
  "dropped support"

List breaking changes in their own section at the top — they must be
impossible to miss.

### Step 5 — Build the changelog section

Format per [Keep a Changelog](https://keepachangelog.com/en/1.0.0/):

```markdown
## [1.2.0] - 2026-06-06

### Breaking Changes
- Removed `--legacy-auth` flag; use `--auth-token` instead (#312)

### Added
- Dark mode support for the web dashboard (#298)
- New `batch_export` API endpoint for bulk session downloads (#301)

### Fixed
- Session list not refreshing after profile switch (#305)
- Crash when log file is empty (#308)

### Changed
- Improved startup time by 40% via lazy-loading providers (#295)

### Security
- Pinned Starlette to >=0.46.2 to address CVE-2026-48710 (#310)

### Documentation
- Added API reference for the new batch endpoints (#303)
```

Rules:
- Each line starts with `-` followed by a capital letter
- Include the PR/issue number at the end when detectable from the commit
  (look for `(#N)` or `#N` in the subject, or `Merge pull request #N`)
- Keep subjects concise — trim boilerplate like "this commit", "we now"
- Omit chores and test-only changes by default (include with `--all` intent)
- If a section has no entries, omit it entirely

### Step 6 — Check for existing CHANGELOG.md

```bash
ls CHANGELOG.md 2>/dev/null && head -5 CHANGELOG.md
```

- **File exists:** prepend the new section after the `# Changelog` header
  and before the previous first release section
- **File does not exist:** create it with the standard header:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
```

Then append the new version section below `[Unreleased]`.

### Step 7 — Write and report

Write the final file using the `write_file` tool. Then report:

```
Changelog updated: CHANGELOG.md

Version:  v1.2.0
Range:    v1.1.0..HEAD
Commits:  47 processed

  Breaking Changes  2
  Added            11
  Fixed             9
  Changed           8
  Security          1
  Documentation     4
  Chores           12  (omitted by default)

Output: /path/to/project/CHANGELOG.md
```

---

## Output Variants

Users may ask for different output formats. Support these without re-fetching:

| Request | Output |
|---|---|
| "GitHub release notes" | Markdown without the `## [version]` header — paste directly into GitHub Releases |
| "Just the summary" | Bullet list of highlights only (top 5-10 items across all categories) |
| "Include chores" | Re-render with chore/ci/build commits included |
| "Plain text" | Strip markdown formatting for email/Slack |
| "Since last week" | Use `--since="7 days ago"` instead of a tag range |

---

## Edge Cases

**No conventional commit prefixes:**
Use keyword fallback (Step 3). Mention to the user that adopting
[Conventional Commits](https://www.conventionalcommits.org/) would improve
future changelogs.

**Merge-only repo (squash merges):**
Read merge commit subjects only. These typically contain PR titles which are
more descriptive than individual commits.

**Monorepo with scopes:**
Group by scope: `feat(web):`, `fix(cli):`, `fix(gateway):`. Create a section
per package/scope instead of a flat list. Ask the user which scopes to include
if the list is large.

**Very large range (500+ commits):**
Warn the user. Offer to limit to the last 100, or to a specific subdirectory:
```bash
git log v1.0.0..HEAD -- path/to/subpackage/
```

**No tags in repo:**
Use `git log --oneline | wc -l` to count total commits. Default to last 50.
Tell the user how to create a tag for future use:
```bash
git tag -a v1.0.0 -m "Initial release"
```

---

## What This Skill Does NOT Cover

- Semantic version bumping — deciding *what* the new version number should
  be. Use `conventional-commits` tooling or ask the user.
- Publishing the release to GitHub/npm/PyPI — do that separately after the
  changelog is ready.
- Changelog for non-git version control systems (SVN, Mercurial).
