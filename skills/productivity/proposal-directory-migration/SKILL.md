---
name: proposal-directory-migration
description: Reorganize legacy P-proposal files from the proposals root into workspace-dev/proposals/<project>/docs/ structure per prj-proposals-manager conventions.
trigger: Proposals root (~/.hermes/proposals/) contains loose P-proposal files or P-subdirectories needing organization
---

# Proposal Directory Migration

Reorganize legacy P-proposal files from the proposals root directory into the `workspace-dev/proposals/<project>/docs/` structure per prj-proposals-manager conventions.

## Context
The `/home/hermes/.hermes/proposals/` root historically accumulated loose P-proposal files and P-subdirectories that need to be organized under their respective project directories.

## Steps

### 1. Explore the full landscape
```bash
# Proposals root: what's loose at the top level
ls -la /home/hermes/.hermes/proposals/

# Find all P-proposal files at root
find /home/hermes/.hermes/proposals/ -maxdepth 1 -name "P-*.md" -o -name "P-*" -type d

# Find all projects in workspace-dev/proposals
ls /home/hermes/.hermes/proposals/workspace-dev/proposals/

# For each project, check if docs/ exists
for dir in /home/hermes/.hermes/proposals/workspace-dev/proposals/*/; do
  echo "$dir: $(ls "$dir/docs/" 2>/dev/null | head -5)"
done

# Check for orphaned .md files directly under workspace-dev/proposals/ level
find /home/hermes/.hermes/proposals/workspace-dev/proposals/ -maxdepth 1 -name "*.md"
```

### 2. Build the mapping (manual)
Read each unmapped P-file's header to identify its project:
```bash
head -5 /home/hermes/.hermes/proposals/P-YYYYMMDD-XXX.md
```

Known mappings from past migration:
- `P-YYYYMMDD-001.md` → often `todo-list` (many items)
- `P-YYYYMMDD-002.md` → `tank-battle`, `calculator-app`, etc.
- `P-YYYYMMDD-003.md` → `calculator-app`
- `P-YYYYMMDD-004.md` → `animal-forest`
- `P-YYYYMMDD-005.md` → `room-escape-puzzle`

Note: Number suffixes (XXX) reset per day, so they don't reliably map across projects.

### 3. Execute migration (Python)
```python
import os, shutil, re

PROPOSALS = "/home/hermes/.hermes/proposals"
DEV_PROJ = f"{PROPOSALS}/workspace-dev/proposals"

mappings = {
    "P-20250416-001.md": ("todo-list", "proposal.md"),
    "P-20250416-001-tech-solution.md": ("todo-list", "technical-solution.v1.md"),
}

for root_file, (project, dest) in mappings.items():
    src = f"{PROPOSALS}/{root_file}"
    dest_dir = f"{DEV_PROJ}/{project}/docs"
    dest_path = f"{dest_dir}/{dest}"
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_path):
        continue
    shutil.copy2(src, dest_path)
    print(f"Copied {root_file} -> {project}/docs/{dest}")
```

### 4. Handle P-subdirectories
P-subdirectories at proposals root contain project content (not multiple proposals):
```bash
ls /home/hermes/.hermes/proposals/P-YYYYMMDD-XXX/
```
Typical contents:
- `docs/P-YYYYMMDD-XXX.md` → copy to project's `docs/proposal.md`
- `docs/index.md` → skip if project already has one
- `game-concept.md` / `README.md` → supplementary, copy to `docs/`
- Tech-solution files → copy to `docs/`

After extracting, delete the empty P-subdirectory.

### 5. Handle orphaned tech-solution files at workspace-dev level
```bash
find /home/hermes/.hermes/proposals/workspace-dev/proposals/ -maxdepth 1 -name "*.md"
```
Move these into the project's `docs/` directory.

### 6. Verify
```bash
# Confirm proposals root is clean
ls -la /home/hermes/.hermes/proposals/ | grep -E "^d|\.md$" | grep -v workspace | grep -v template

# Confirm workspace-dev level has no orphaned .md
find /home/hermes/.hermes/proposals/workspace-dev/proposals/ -maxdepth 1 -name "*.md"
```

### 7. Update proposal-docs-index.md and proposal-index.md
Root-level index files need regeneration after migration.

## Edge Cases

**Ambiguous project**: Check `proposal-index.md` at root for cross-references, or leave as "unmapped" and report to user.

**Project already has docs/proposal.md**: Do NOT overwrite unless the P-file is provably more complete. Supplementary files can be copied regardless.

**monopoly3d at root**: `/home/hermes/.hermes/proposals/monopoly3d/` is a project dir (not P-subdirectory). Copy to `workspace-dev/proposals/monopoly3d/` rather than move to preserve git history.

**OpenMAIC vs workspace-pm**: `P-20250418-001-MODEL-MANAGEMENT.md` belongs to `workspace-dev/proposals/OpenMAIC/docs/`, NOT workspace-pm. The `workspace-pm/` dir is for the proposals-manager project itself.

## Verification Checklist
- [ ] Proposals root has no P-subdirectories remaining
- [ ] No orphaned `.md` files under `workspace-dev/proposals/` level
- [ ] Each project with migrated content has a `docs/` directory
- [ ] Existing `proposal.md` files were not overwritten without cause
- [ ] `proposal-docs-index.md` and `proposal-index.md` updated
