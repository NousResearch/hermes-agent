---
name: project-cleanup
description: "Project bloat cleanup: pycache/empty/duplicate files detection. 项目冷热分离清理。"
version: 1.0.0
author: andorexu
license: MIT
metadata:
  hermes:
    tags: [cleanup, maintenance, janitor, tidy, bloat, cold-storage]
    related_skills: []
---

# Project Cleanup — 项目冷热分离 / 一键清理

## Overview / 概述

Walks a project directory, inventories all files, and identifies bloat across six standard categories. Generates a report, then executes cleanup after user confirmation. Always backs up first.

遍历项目目录，盘点所有文件，识别六类常见垃圾。生成报告，用户确认后清理。先备份再动手。

## When to Use / 触发场景

- User says "整理一下项目" / "clean up this project" / "冷热分离" / "太臃肿了精简一下" / "tidy up" / "janitor" / "项目清理"
- After a long development sprint with accumulated test artifacts
- Before archiving or handing off a project
- Hermes Memory is full and needs compaction

**Don't use for:** single-file edits, active development cleanup (use lint/format instead), projects under 50 files.

## Workflow

### Phase 1: Backup
```
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp -r src/ backups/<timestamp>/
cp *.md backups/<timestamp>/ 2>/dev/null
```
Never skip this. If the user objects, explain the 30-second cost vs. irreversible loss.

### Phase 2: Inventory (read-only, no changes)
Run the inventory script to produce a report with exact counts and sizes:

```python
import os, datetime

def inventory(root, max_depth=4):
    """Walk project and categorize files."""
    stats = {
        'pycache': [], 'empty': [], 'duplicates': {},
        'exports': [], 'old_backups': [], 'deprecated_docs': [],
        'logs_old': [], 'total_files': 0, 'total_size': 0,
        'py_files': 0, 'py_lines': 0
    }
    
    cutoff_30d = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y%m%d')
    
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, '').count(os.sep)
        if depth > max_depth or '.git' in dirpath or 'node_modules' in dirpath:
            continue
            
        # Flag __pycache__
        if '__pycache__' in dirpath:
            for f in filenames:
                stats['pycache'].append(os.path.join(dirpath, f))
            continue
            
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(fp)
            except:
                continue
            stats['total_files'] += 1
            stats['total_size'] += size
            
            # Empty files
            if size == 0 and f not in ['.gitkeep', '__init__.py']:
                stats['empty'].append(fp)
            
            # Python cache
            if f.endswith('.pyc'):
                stats['pycache'].append(fp)
            
            # Test exports (heuristic: timestamped xlsx/docx/pdf in exports/)
            if 'exports' in dirpath and f.endswith(('.xlsx', '.docx', '.pdf')):
                stats['exports'].append(fp)
            
            # Duplicate files (by name+size)
            key = (f, size)
            if key not in stats['duplicates']:
                stats['duplicates'][key] = []
            stats['duplicates'][key].append(fp)
            
            # Old backups (>5 in backups/ dir)
            if 'backups' in dirpath and os.path.isdir(fp):
                stats['old_backups'].append(fp)
            
            # Python count
            if f.endswith('.py'):
                stats['py_files'] += 1
                try:
                    with open(fp) as fh:
                        stats['py_lines'] += len(fh.readlines())
                except:
                    pass
    
    return stats
```

### Phase 3: Report (show user, don't act yet)
Format findings into a table:

```
| # | Category | Count | Size | Action |
|---|----------|-------|------|--------|
| 1 | __pycache__ / .pyc | N | X KB | Delete |
| 2 | Empty files (0KB) | N | 0 | Delete |
| 3 | Test exports (xlsx/docx/pdf) | N | X MB | Delete |
| 4 | Duplicate files | N | X KB | Keep 1, delete rest |
| 5 | Old backups (>5 recent) | N | X MB | Archive or delete |
| 6 | Deprecated docs | N | X KB | Move to archive/ |
| 7 | Old logs (>30 days) | N | X KB | Archive |
```

**Golden rule: show the list, get confirmation, THEN execute. Never delete without showing first.**

### Phase 4: Execute (after user says "执行" / "do it" / "全部执行")

```bash
# 1. Delete __pycache__ and .pyc
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 2. Delete empty files (keep __init__.py and .gitkeep)
find . -type f -size 0 ! -name "__init__.py" ! -name ".gitkeep" -delete 2>/dev/null

# 3. Delete test exports
rm -f src/runtime/exports/*.xlsx src/runtime/exports/*.docx 2>/dev/null

# 4. Handle duplicates: keep the one in the most canonical path, remove others
# (Do this carefully — show user which one stays)

# 5. Keep last 5 backups, delete older
ls -dt backups/[0-9]*/ 2>/dev/null | tail -n +6 | xargs rm -rf

# 6. Move deprecated docs to docs/archive/
mkdir -p docs/archive
mv PROJECT_STATUS.md TASKS.md README.md docs/archive/ 2>/dev/null

# 7. Archive old logs (>30 days)
cutoff=$(date -d "30 days ago" +%Y%m%d)
mkdir -p memory/archive
for f in memory/????-??-??.md; do
    [ -f "$f" ] || continue
    d=$(basename "$f" .md | tr -d '-')
    [ "$d" -lt "$cutoff" ] && mv "$f" memory/archive/
done
```

### Phase 5: Verify
- [ ] `du -sh src/` — confirm size reduction
- [ ] `find . -name "*.pyc" | wc -l` — should be 0
- [ ] `find . -type d -name __pycache__ | wc -l` — should be 0
- [ ] `ls backups/ | wc -l` — should be ≤ 5
- [ ] Key files (settings.json, .env, main .py) still present

## Common Pitfalls

1. **Deleting without backup.** Always `mkdir backups/<ts> && cp -r src/ backups/<ts>/` first.
2. **Deleting __init__.py.** Never delete empty `__init__.py` files — they're package markers.
3. **Blindly deleting all exports.** User may have intentionally saved exports. Show the list first.
4. **Deleting .env or config files.** 0KB `.env` might be a template. Ask.
5. **Moving logo files that HTML references.** Check `grep -r "logo" *.html` before deleting image assets.
6. **Assuming one-size-fits-all.** Different projects have different junk patterns. Adapt the inventory script.

## Hermes Memory Compaction (bonus)

When Hermes Memory approaches capacity (≥90%), compact it:
1. Identify entries that overlap or are now codified in SOUL.md / AGENTS.md
2. Remove entries already covered by higher-priority rule files
3. Merge adjacent entries (e.g., two entries about task reporting → one)
4. Compress verbose entries to their essence
5. Target: ≤80% after compaction

Use `memory(action='remove', old_text='<unique substring>')` to delete, then `memory(action='add', ...)` or `replace` to write the compact version.

## Verification Checklist

- [ ] Backup created before any deletions
- [ ] Report shown to user, confirmation received
- [ ] __pycache__ and .pyc count → 0
- [ ] Empty files deleted (except __init__.py)
- [ ] Test exports cleaned
- [ ] Duplicates handled
- [ ] Old backups trimmed
- [ ] Deprecated docs archived
- [ ] src/ size reduction recorded
- [ ] HTML image references not broken


## Author / 作者

- **GitHub:** [github.com/andorexu](https://github.com/andorexu)
- **Company / 公司:** 百赛联（深圳）科技有限公司
- **Email:** andore@sina.com

