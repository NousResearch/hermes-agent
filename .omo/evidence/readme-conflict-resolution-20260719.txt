README conflict resolution evidence
Date: 2026-07-19
Scope: README.md only

Rationale:
- MERGE_HEAD:README.md contains the complete current upstream README, including
  official install, getting-started, portal, documentation, migration,
  contributing, community, and license sections.
- The fork side had a single conflict block at the upstream Getting Started
  insertion point and a large fork-only README body that omitted those official
  sections.
- The resolved file therefore keeps MERGE_HEAD:README.md as the base and adds
  one concise Fork Navigation section before License. It links to fork/README.md
  and identifies the policy dry-run and Windows stack restart entry points.

Validation commands and observed output:

1. Command: rg -n -e "<<<<<<<" -e "=======" -e ">>>>>>>" README.md
   Output: no matches (exit 1).

2. Command: git diff --cached --check -- README.md
   Output: no output (exit 0).

3. Command: git ls-files -u -- README.md
   Output: no output (exit 0; README.md has no unmerged index entries).

4. Command: git diff --cached --numstat -- README.md
   Output: 171 160 README.md

5. Command: git status --short -- README.md
   Output: M  README.md (staged only).
