---
name: changelog-gen
description: Generate changelogs from git history.
version: 1.0.0
author: Kewe63
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [changelog, git, conventional-commits, release-notes]
    related_skills: [writing-plans, humanizer]
prerequisites:
  commands: [python3, git]
---

# Changelog Generator Skill

Generate Markdown or JSON changelogs from a repo's git history. Parses
conventional commits, auto-categorizes features/fixes/breaking changes, and
groups them by version tag. Runs entirely locally — no API key, no network.

## When to Use

- You need a release changelog grouped by commit type and version tag.
- You want a JSON summary of commit counts by type for tooling/CI.
- You are preparing release notes and want a structured starting point.

Prefer `humanizer` when the goal is rewriting prose; this skill only summarizes
existing commit history — it does not invent or polish wording.

## Prerequisites

- Python 3.10+ (stdlib only — `argparse`, `json`, `re`, `subprocess`).
- A local git repository with conventional-commit messages.
- No network, no API key.

## How to Run

Drive the bundled helper script through the native `terminal` tool — that is
the supported interaction surface here, not ad-hoc Python pasting. Substitute
the resolved `${HERMES_SKILL_DIR}` at runtime (or run from inside the skill
directory).

```bash
# Generate a changelog for the current repo (last 100 commits)
python3 "${HERMES_SKILL_DIR}/scripts/changelog_gen.py"

# From a specific path, all commits, written to a file
python3 "${HERMES_SKILL_DIR}/scripts/changelog_gen.py" --path /path/to/repo --all --output CHANGELOG.md

# Emit only the JSON stats object on stdout
python3 "${HERMES_SKILL_DIR}/scripts/changelog_gen.py" --json
```

## Quick Reference

| Flag | Meaning | Notes |
|------|---------|-------|
| `--path <dir>` | Target repository (default: `.`) | Any local git repo |
| `--all` | Include all commits (default: last 100) | Also includes pre-oldest-tag bucket |
| `--output <file>` | Write Markdown to file | Otherwise prints to stdout |
| `--json` | Print JSON stats exclusively | No Markdown on stdout |

The 8 type buckets: Breaking Changes, Features, Bug Fixes, Performance,
Refactoring, Documentation, Styling, Tests, Chores, CI/CD, Build, Other.

## Procedure

1. Run `changelog_gen.py` against the repo to get a Markdown changelog sorted by
   tag range (Unreleased → newest tag → … → oldest).
2. Within each tag range, commits are bucketed by conventional-commit type.
3. Use `--output` to save the Markdown, or `--json` when a downstream tool needs
   the structured `by_type` counts instead of prose.
4. For release notes, hand-edit the generated Markdown — the script only
   summarizes, it does not rewrite wording.

## Pitfalls

- The parser keys off the conventional-commit `type(scope): description` shape.
  Commits without a recognized type land in `Other`; commits without a `type:`
  colon are still listed under `Other`, not dropped.
- `--json` is exclusive: when set, **no Markdown is printed** — stdout is a
  single JSON object. Do not pipe it into a Markdown renderer.
- Tag grouping relies on actual git tags. A repo with no tags collapses to a
  single flat changelog (no "Unreleased" section).
- Multi-line commit bodies are preserved verbatim inside each entry's record;
  the NUL/RS stream splitter keeps bodies intact rather than field-splitting
  them.

## Verification

Run the bundled tests (no network — pure functions + temp git repos):

```bash
scripts/run_tests.sh tests/skills/test_changelog_gen_skill.py -q
```

Spot-check live output before quoting results:

```bash
python3 "${HERMES_SKILL_DIR}/scripts/changelog_gen.py" --path . --json
```
