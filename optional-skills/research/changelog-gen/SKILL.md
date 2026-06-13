---
name: changelog-gen
description: Generate changelogs from git commit history. Parses conventional commits, auto-categorizes features/fixes/breaking changes, groups by version tag, and outputs clean Markdown.
platforms: [linux, macos, windows]
---

# Changelog Generator

Generate changelogs from git commit history using conventional commits format.

## Helper script

This skill includes `scripts/changelog_gen.py` — a complete CLI tool.

```bash
# Generate changelog from current repo
python3 SKILL_DIR/scripts/changelog_gen.py

# From a specific path
python3 SKILL_DIR/scripts/changelog_gen.py --path /path/to/repo

# Include all commits since beginning
python3 SKILL_DIR/scripts/changelog_gen.py --all

# Output to file
python3 SKILL_DIR/scripts/changelog_gen.py --output CHANGELOG.md
```

Output is clean Markdown grouped by type (Features, Bug Fixes, Breaking Changes, etc.).