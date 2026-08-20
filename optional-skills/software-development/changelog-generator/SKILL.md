---
name: changelog-generator
description: "Build CHANGELOG.md entries from git commit history."
version: 0.1.0
author: Burak Koç (@HeLLGURD), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: software-development
    tags: [Changelog, Git, Release, Documentation, Versioning]
    related_skills: [code-wiki, github-pr-workflow]
---

# Changelog Generator Skill

Turn a range of git commits into a [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
section, grouped by type (breaking changes, added, fixed, changed, and so
on), and merge it into `CHANGELOG.md` without dropping existing history.
It categorizes commits and formats the release notes; it does not decide the
next version number, publish a release, or work outside git.

Categorization and the history-preserving merge live in
`scripts/changelog.py`; run it through the `terminal` tool. No external
services or extra dependencies beyond `git`.

## When to Use

- User says "generate changelog", "update CHANGELOG.md", or "write release notes".
- User points at a tag or commit range and asks what changed.
- Before cutting a release when the changelog is stale or missing.

Do NOT use this for:

- Single-commit summaries — describe the commit directly.
- Choosing the version number — that is semantic-versioning judgment, not this skill.
- Publishing to GitHub/npm/PyPI — a separate step after the changelog is ready.
- Non-git repositories (SVN, Mercurial) — there is no commit history to read.

## Prerequisites

- `git` on PATH and the working directory inside a git repository.
- No env vars, no network, no MCP servers.

## How to Run

1. Resolve the commit range with `git` (through `terminal`).
2. Pipe the log into `scripts/changelog.py` to categorize and render.
3. Point the script at `CHANGELOG.md` so it prepends the new section and
   preserves every prior entry; or read the full file with `read_file` and
   insert the section yourself with `patch`.

## Quick Reference

| Step | Action | Tool |
|---|---|---|
| 1 | Confirm repo root; list tags newest-first | `terminal` (`git`) |
| 2 | Resolve the version range | `terminal` (`git`) |
| 3 | Categorize + render the section | `terminal` (`scripts/changelog.py`) |
| 4 | Merge into `CHANGELOG.md`, preserving history | `scripts/changelog.py` or `read_file` + `patch` |
| 5 | Report the output path and per-category counts | — |

## Procedure

### 1. Resolve repo and range

```bash
git rev-parse --show-toplevel
git tag --sort=-version:refname
```

Read the top of the tag list for the most recent tags. Determine the range:
tag-to-tag (`v1.1.0..v1.2.0`), tag-to-HEAD (`v1.1.0..HEAD`), or, when the
repo has no tags, the last 50 commits (`-n 50`). Count total commits with
`git rev-list --count HEAD` rather than piping through a line counter.

### 2. Collect commits

```bash
git log v1.1.0..v1.2.0 --pretty=format:"%H|%s|%an|%ae|%ad" --date=short > /tmp/commits.txt
```

Each line is `hash|subject|author|email|date`. For a suspected breaking
change, read the body with `git log --format=%B <hash>` to check for a
`BREAKING CHANGE:` footer.

### 3. Categorize and render

```bash
python scripts/changelog.py --version v1.2.0 --log-file /tmp/commits.txt
```

Prefix and keyword rules are documented in `references/categories.md`. The
script strips prefixes, appends detected `(#PR)` numbers, lists breaking
changes first, and omits chore/test noise unless you pass `--all`.

### 4. Merge without losing history

Pass the existing file so the script reads it in full and splices the new
release above the previous one:

```bash
python scripts/changelog.py --version v1.2.0 --log-file /tmp/commits.txt --changelog CHANGELOG.md
```

If no `CHANGELOG.md` exists, the script creates one with the standard Keep a
Changelog header. To merge by hand instead, `read_file` the entire file and
apply `patch` in insert mode — never `write_file` a changelog you only
partially read, because it replaces the whole file and discards history.

### 5. Report

Summarize the output path and per-category counts (the script emits these
with `--json`).

## Pitfalls

- **Truncating history.** `write_file` overwrites the entire file. Reading a
  few lines and then writing drops every older release. Use the script's
  `--changelog` merge or `read_file` (full) + `patch`.
- **No conventional-commit prefixes.** The keyword fallback still works;
  suggest adopting Conventional Commits for cleaner future runs.
- **Squash-merge repos.** Prefer merge-commit subjects (`git log --merges`);
  they carry PR titles that read better than individual commits.
- **Very large ranges (500+ commits).** Warn the user and offer to scope to
  a subdirectory (`git log <range> -- path/`) or the last 100 commits.

## Verification

- `python scripts/changelog.py --version vX.Y.Z --log-file <file>` prints a
  well-formed section with breaking changes first.
- After a `--changelog` merge, `read_file` the result and confirm previous
  release headings (for example `## [1.0.0]`) are still present above the
  header and the new section sits at the top.
- Run the skill test: `scripts/run_tests.sh tests/skills/test_changelog_generator_skill.py -q`.
