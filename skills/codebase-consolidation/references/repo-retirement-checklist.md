# Repo Retirement Checklist

Created: 2026-05-16 after nuking hermes-scripts, content-engine, mission-control.
Based on live execution by Kensei.

## The One-Off "Nuke Three Repos" Pattern

When Sahil says "nuke these repos locally" because upstream GitHub repos are archived/deleted.

## Step-by-step

### Pre-flight

- [ ] Confirm repos exist: `find ~/{repos,apps} -maxdepth 2 -name ".git" | while read d; do dirname "$d"; done`
- [ ] For each repo, confirm remote is actually gone: `git -C ~/repos/<repo> remote -v`
- [ ] Identify what board the triage tasks live on (often ops board, not default)

### Verification (scripts)

- [ ] Check KenseiAgent commit log: `git -C ~/repos/KenseiAgent log --oneline | grep -i "migrate"`
- [ ] Compare file lists: `find ~/repos/<old-repo> -type f -not -path '*/.git/*' | sort` vs `find ~/repos/KenseiAgent/<target-dir> -type f -not -path '*/.git/*' | sort`
- [ ] Use `comm -23` to find files missing from target
- [ ] Check dotfiles: `.claude/skills/`, `.gitnexus/`, custom project scaffolding (CLAUDE.md, AGENTS.md)

### Salvage

- [ ] Identify unique content worth preserving (plugins, bridges, custom middleware)
- [ ] Copy to `~/backups/<descriptive-name>/`
- [ ] Note where the canonical replacement already lives (e.g. V2 in KenseiAgent content_engine/)
- [ ] If there are uncommitted changes, save those too

### Nuking

- [ ] Run `rm -rf ~/repos/<repo>`
- [ ] Run `rm -rf ~/apps/<repo>` if it also existed there
- [ ] Verify gone: `ls ~/repos/` and `ls ~/apps/`

### Post-nuke kanban

- [ ] For salvageable work: create blocked task on default board
  ```bash
  TASK=$(hermes kanban --board default create "Title" --assignee octacon --body "Backup at ..." | grep -oP 't_\w+')
  ```
  Then set to blocked via SQLite:
  ```bash
  python3 -c "import sqlite3; c=sqlite3.connect('/home/kensei/.hermes/kanban.db'); c.execute(\"UPDATE tasks SET status='blocked' WHERE id='$TASK'\"); c.commit()"
  ```
- [ ] For board investigation needs: create task on apps board (if board DB exists)
- [ ] Close original triage tasks on ops board that flagged the issue
  ```python
  DB = '/home/kensei/.hermes/kanban/boards/ops/kanban.db'
  cur.execute("UPDATE tasks SET status='done', result='Resolved: ...', completed_at=? WHERE id=?", (now, tid))
  ```

## Real execution trace (2026-05-16)

| Repo | Found at | Nuked? | Salvage | Task created |
|------|----------|--------|---------|-------------|
| hermes-scripts | ~/repos/ | ✅ | None (all in KenseiAgent scripts/) | N/A |
| content-engine | ~/apps/ | ✅ | Assets backed up | N/A |
| mission-control | ~/apps/ | ✅ | Postiz plugin → ~/backups/ | t_b006b06e (blocked) |
| Board architecture | N/A | N/A | N/A | t_2dcff3b6 (apps board, running) |
