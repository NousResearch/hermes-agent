# Scope reconciliation: the goal's "./src/" == repo root (not a subdir)

The goal says "all the changes we have in ./src/ when compared to version v0.16.0".

**Finding (evidence-backed):** `./src/` is the user's source *checkout directory*
`<REPO_ROOT>`, which **IS the git repo root** — not a subdirectory
named `src/` inside the repo.

Proof:
- `git rev-parse --show-toplevel` → `<REPO_ROOT>` (the `.git` dir lives there).
- `git ls-files | grep '^src/'` → 0 files (no tracked path starts with `src/`).
- `git diff --name-only v0.16.0..HEAD -- src/` → 0 files (no such subdir to filter on).
- `git diff --name-only v0.16.0..HEAD` (repo root) → 160 files.

**Therefore** the correctly-scoped delta set = the 160 repo-root files. Every
coverage/partition/apply result in this PR uses that exact set. The earlier
"`git diff -- src/` returns 0" was the no-subdir fact, not a missed scope.
