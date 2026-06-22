# ./src/-scoped diff-coverage proof

The goal says "changes in ./src/". This repo has **NO `src/` subdirectory**:
- `git diff 3c231eb..HEAD -- src/` = **0 files** (nothing under a src/ path).
- `git rev-parse --show-toplevel` = the repo root, which IS the user's `./src/`
  checkout dir (`<REPO_ROOT>` — a checkout *named* src, not a subdir).
- `git ls-files | grep '^src/'` = 0.

So "./src/" = the repo root, and the correctly-scoped delta is the 165 repo-root files.

## No DISCARD/upstream path is under a src/ subdir
All 27 DISCARD+upstream files are repo-root paths (verified `grep -c '^src/'` = 0):
- 9 `.bak` (agent/*.bak, hermes_cli/*.bak, tools/*.bak) — editor/build backups
- 12 `.project-intel/` — generated index (incl. binary .sqlite)
- 4 `transcripts/` — Fable-5 prelude eval-capture .txt files
- 2 `agent/subdirectory_hints.py` + its test — covered by upstream #29433 (udatny superset)

None is contributable source; none would be judged differently as a src/ path
because there are no src/ paths.
