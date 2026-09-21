---
name: codebase-consolidation
description: "Merge, consolidate, and restructure codebases across repos. Deep analysis first, validate imports at runtime, never guess."
version: 1.0.0
author: Kensei
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [consolidation, merge, refactor, repo-management, imports]
    related_skills: [writing-plans, github-pr-workflow, requesting-code-review]
adoption_status: permanent
---

# Codebase Consolidation

## When to Trigger

- User wants to merge two repos into one
- "Creating repos for the sake of creating repos"
- Duplicated or overlapping code across repos
- Feature creep in repo count vs useful consolidation

## Process (Do Not Skip Steps)

### Step 1: Recon — Map Both Codebases

Before touching any files, map the complete architecture of each codebase:

```bash
# For each repo: list files, line counts, key function names
for f in *.py; do echo "--- $f ($(wc -l < $f) lines) ---"; grep "^def \|^class " "$f"; done
```

Identify:
- What each file does (read the docstring + first 20 lines)
- All imports/dependencies between files
- All imports from code used externally (cron jobs, other modules)
- Function/variable names that could collide

### Step 2: Identify Overlap vs Complement

Build a matrix:

| Component | Codebase A | Codebase B | Relationship |
|-----------|-----------|-----------|-------------|
| config | config.py | src/config.py | MERGE (different purposes) |
| images | fal_client.py | image_generator.py | REPLACE (B is superset) |
| video | ffmpeg_video.py | video_generator.py | COEXIST (different functions) |
| drafts | llm_drafts.py | missing | A-ONLY |
| screenshots | missing | screenshot_repurposer.py | B-ONLY |

Three categories:
- **MERGE**: Both files serve same concept but different details (configs, registries)
- **REPLACE**: One is clearly superseding the other
- **COEXIST**: Different functions, different responsibilities, both kept

### Step 3: Plan the Merge (Write It Down)

Before copying a single file, document:
- What gets replaced
- What gets merged
- What stays as-is
- What new files are added
- Which existing files must be updated (imports, references)

### Step 4: Validate Every Merge Before Applying

**Critical**: Config merges are the most dangerous. Validate both:

1. **What existing code imports from the config?**
   ```bash
   grep -rn "from config import\|import config" --include="*.py"
   ```
   Every symbol existing code depends on MUST survive the merge.

2. **What brand keys are used as fallbacks?**
   ```bash
   grep -n 'BRANDS\[' *.py
   ```
   If existing code does `BRANDS["sahil_twitter"]` as a fallback, that key MUST exist in the merged config.

3. **What dict fields does existing code read?**
   ```bash
   grep -n 'brand_config\["'\|'cfg\["' *.py
   ```
   If code reads `cfg["bg"]`, `cfg["accent"]`, those keys must be in every brand entry.

4. **Check both ways**: existing files importing new symbols + new files importing existing symbols.

### Step 5: Build the Merged File, Validate, Then Replace

Write the merged file to `/tmp/merged_<name>.py` first:

```bash
cd /path/to/project && python3 -c "
import sys; sys.path.insert(0, '/tmp')
import merged_config as c
# Verify every exported symbol
checks = [
    ('BASE_DIR', c.BASE_DIR),
    ('BRANDS', c.BRANDS),
    ('font_path', callable(c.font_path)),
    # ... every symbol existing code depends on
]
for name, val in checks:
    status = 'OK' if val or val == '' else 'MISSING'
    print(f'{status}: {name}')
"
```

Only after ALL checks pass, copy to the real location.

### Step 6: Compile Check Everything

```bash
cd /path/to/project && for f in *.py; do
    python3 -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "OK: $f"
done
```

### Step 7: Runtime Import Check

```bash
cd /path/to/project && python3 -c "
from config import BRANDS, OUTPUT_DIR, DB_PATH, ...
from database import init_db, insert_draft, ...
from content_engine import run_stage_1, run_stage_2
# ... every import chain the codebase uses
print('ALL IMPORTS VERIFIED')
"
```

### Step 8: Commit and Clean Up

```bash
git add <merged-file> <new-files>
git commit -m "Consolidate: merge <repo-b> into <repo-a>"
git push origin main
```

Then delete the source repo.

## Pitfalls

### Config Brand Key Collisions
When V1 uses `"sahil_twitter"` as two separate keys and V2 uses `"personal"` as a combined key, DO NOT merge into one key. Existing code has hardcoded fallbacks (`BRANDS["sahil_twitter"]`). Keep the separate keys and add the new one.

### Dict Field Overwrites
V1 config dicts have `{display, handle, colour, bg, accent}` for visuals. V2 dicts have `{name, description, voice_skill, content_pillars}` for content. Merged config must have ALL fields. Existing code reads specific keys — missing any one is a runtime crash.

### "Just Copy and Replace" Is Wrong
Blindly replacing V1 config.py with V2 config.py breaks 5 module imports. Every symbol that existing code depends on must be preserved or re-mapped.

### Lightweight repo retirement (symlink migration)
When a repo's contents have been migrated to another repo but external consumers still reference the old path via relative lookups (e.g. cron jobs resolving scripts from `~/.hermes/scripts/`):

1. Verify every non-artefact file from the old repo exists in the new one:
   ```bash
   ls -1 old/scripts/ | sort > old.txt
   ls -1 new/scripts/ | sort > new.txt
   comm -23 old.txt new.txt  # must be empty or only .bak/.pyc
   ```
2. Remove the old directory entirely: `rm -rf old/scripts`
3. Replace with a symlink: `ln -sf /path/to/new/scripts /path/to/old/scripts`
4. Validate downstream consumers still resolve: `readlink -f /path/to/old/scripts/drift-check.py`
5. Never orphan live cron jobs — always verify `hermes cron list` script paths resolve after retirement.

### Don't Rush
When the user says "WE NEED TO BE CAREFUL NOT TO BREAK ANYTHING" — they're not being dramatic. Config merges are the single most likely place to break a working codebase silently. Read both files completely. Map every dependency. Validate at runtime before committing.

## Repo Retirement (Archive Mode)

Use when a remote repo has been archived/deleted and local changes are orphaned. This is NOT a merge — it's a clean-up.

### Checks Before Nuking

1. **Verify content exists in target repo(s) before deleting:**
   - List all files from the source: `find src_repo -type f -not -path '*/.git/*' -not -path '*/__pycache__/*' | sort`
   - List all files in the target: `find target_repo -type f -not -path '*/.git/*' -not -path '*/__pycache__/*' | sort`
   - Check coverage: `comm -23 src.txt target.txt` — every non-artefact file must exist in the target
   - Check custom content (scripts, .claude skills, gitnexus configs) specifically — these are easy to lose

2. **Identify salvageable work:**
   - Is any part of the repo still useful but not yet migrated? (Plugins, bridges, configs, middleware)
   - Save it: `cp -r src_repo/path/to/valuable /backup/location/`
   - Record what was saved and where the canonical replacement lives

3. **Check for baked-in references from cron jobs or other automation:**
   - `grep -r "src_repo_name" ~/.hermes/cron/` — cron prompts that reference files by path
   - `grep -r "src_repo_name" ~/.hermes/skills/` — skills that reference the old path
   - `hermes cron list | grep src_repo_name` — live cron jobs
   - Update any references BEFORE deleting. Dead cron script paths = agent failures waiting to happen.

4. **If the remote is gone (404 on GitHub, archived repo):**
   - Confirm the remote is actually unreachable: `git ls-remote origin 2>&1`
   - If yes, there's no git operations to worry about — local commits can't be pushed. They're either lost or already migrated.

### Nuke Step

```bash
rm -rf /path/to/src_repo
```

### Post-Nuke: Create Kanban Trail

When salvageable work was preserved for future reintegration:

1. Create a blocked task on the default board:
   ```bash
   TASK_ID=$(hermes kanban --board default create "Reintegrate X from retired repo" --assignee octacon --body "Backup at ~/backups/X/. Canonical code in Y/repo/." | grep -oP 't_\w+')
   python3 -c "import sqlite3; c=sqlite3.connect('/home/kensei/.hermes/kanban.db'); c.execute(\"UPDATE tasks SET status='blocked' WHERE id='$TASK_ID'\"); c.commit()"
   ```

2. Close the original triage/research tasks that flagged the orphan state.

3. Keep logs of what was nuked and where the replacement lives for reference.

### Pitfalls

**Pitfall: Don't assume KenseiAgent has everything**
The KenseiAgent repo may have the operational scripts but lack custom Claude Code skills (.claude/skills/) or GitNexus configs (.gitnexus/). Always verify these separately — they're easy to miss in dotfiles.

**Pitfall: Cron scripts may be hardlinked, not copied**
Hermes crons reference scripts from ~/.hermes/scripts/ which may be hardlinked to the repo. Deleting the repo breaks the cron even if the content exists elsewhere. Verify the script path resolves independently:
```bash
readlink -f /home/kensei/.hermes/scripts/any-script.py
```

**Pitfall: The KenseiAgent commit log is your best coverage tool**
Check `git -C ~/repos/KenseiAgent log --oneline | grep -i "migrate\|move\|copy"` before assuming content was ported. The commit message "Migrate operational scripts from hermes-scripts to agent repo" is a fast yes/no signal.

## File Structure (references/)
- See references/consolidation-checklist.md for a reusable step-by-step checklist.
- See references/repo-retirement-checklist.md for the archive-and-retire pattern used when remotes are dead and repos need clean removal (created 2026-05-16 based on nuking hermes-scripts, content-engine, mission-control).
