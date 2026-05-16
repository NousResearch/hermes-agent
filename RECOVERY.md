# Hermes Agent Fork Recovery Manual

**Repo:** `github.com/vystartasv/hermes-agent`
**Upstream:** `github.com/NousResearch/hermes-agent`
**Custom commit:** Knowledge-db tool integration + document type whitelist (6 files)
**Last updated:** 2026-05-16

## Architecture

```
upstream/main  → NousResearch (source of truth, never modified)
origin/main    → Our fork (upstream + 1 custom commit)
Local main     → ~/.hermes/hermes-agent (working copy)
```

Our custom changes (6 files):
- `tools/knowledge_tool.py` — Rust-backed knowledge store
- `tools/knowledge_fallback.py` — Pure Python fallback
- `tools/memory_tool.py` — Knowledge-db integration
- `gateway/platforms/base.py` — Added .py, .har, .sh to whitelist
- `tests/tools/test_memory_tool.py` — Updated tests
- `tests/tools/test_memory_tool_import_fallback.py` — Updated tests

## Disaster Scenarios

### 1. Rebase conflict after upstream pull

**Symptom:** `git rebase upstream` fails with conflict markers. Gateway may be in inconsistent state.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent
git rebase --abort                    # Undo the failed rebase
git stash pop                         # Restore any stashed changes
```

**Diagnosis:**
```bash
# Which files conflict?
git diff upstream/main --name-only   # List all files we modify that upstream changed
```

**Resolution:**
```bash
# Option A: Manual merge (if confident)
git rebase upstream
# Resolve conflicts in each file, then:
git add <resolved-files>
git rebase --continue
pip install -e .
hermes gateway restart

# Option B: Re-apply our changes from scratch (safer)
git checkout upstream/main           # Start from clean upstream
# Copy our 6 custom files from backup or re-create them
# Commit, push
```

### 2. Gateway won't restart after update

**Symptom:** `hermes gateway restart` fails. `systemctl status hermes-gateway` shows errors.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent
pip install -e .                     # Reinstall
hermes doctor --fix                  # Fix dependencies
hermes gateway restart               # Try again

# If still failing, check logs:
tail -50 ~/.hermes/logs/gateway.log
tail -50 ~/.hermes/logs/errors.log

# Common fixes:
# - Missing dependency: pip install <missing-package>
# - Config migration: hermes config migrate
# - Venv broken: rebuild (see Scenario 6)
```

### 3. Custom changes lost (force push or bad rebase)

**Symptom:** `knowledge` tool returns "tool not found". Knowledge-db writes failing silently.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent

# Check if our commit exists
git log --oneline | grep "custom: knowledge-db"
# If missing, our commit was lost

# Recover from origin
git fetch origin
git log origin/main --oneline | grep "custom: knowledge-db"
# If found on origin: git reset --hard origin/main
# If not on origin: rebuild from scratch (Scenario 7)
```

### 4. Upstream breaks our tools

**Symptom:** After update, `knowledge_tool.py` crashes with import errors. Internal APIs changed.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent

# Check what changed in files we depend on
git diff upstream/main -- tools/registry.py model_tools.py toolsets.py

# Quick fix: patch our tool files to match new APIs
# Or rollback to previous working state:
git reset --hard HEAD~1              # Undo last rebase commit
pip install -e .
hermes gateway restart

# Then fix the compatibility issue before re-rebasing
```

### 5. Complete fork corruption

**Symptom:** Local repo in unrecoverable state. Force pushes broke history.

**Recovery:**
```bash
# Clone fresh
cd /tmp
git clone git@github.com:vystartasv/hermes-agent.git hermes-agent-recovery
cd hermes-agent-recovery

# Add upstream
git remote add upstream git@github.com:NousResearch/hermes-agent.git
git fetch upstream

# Rebuild our branch
git checkout -b main origin/main
git rebase upstream/main             # Replay our commit on latest upstream

# Verify our files exist
ls tools/knowledge_tool.py tools/knowledge_fallback.py

# Replace broken local copy
rm -rf ~/.hermes/hermes-agent
mv /tmp/hermes-agent-recovery ~/.hermes/hermes-agent
cd ~/.hermes/hermes-agent
pip install -e .
hermes gateway restart
```

### 6. Venv corruption

**Symptom:** `ModuleNotFoundError`, `ImportError`, Python can't find hermes modules.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent

# Rebuild venv from scratch
rm -rf venv .venv
python3.13 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt 2>/dev/null || true
hermes doctor --fix

# Verify
python3 -c "from tools.knowledge_tool import knowledge_write; print('OK')"
hermes gateway restart
```

### 7. Rebuild custom commit from scratch

**Symptom:** Our commit is gone from both local and origin. Need to recreate.

**Recovery:**
```bash
cd ~/.hermes/hermes-agent

# Rebase onto latest upstream
git fetch upstream
git reset --hard upstream/main

# Recreate our files (copy from backup or memory):
# - tools/knowledge_tool.py    → Rust-backed knowledge store
# - tools/knowledge_fallback.py → Pure Python fallback  
# - tools/memory_tool.py       → modified for knowledge-db
# - gateway/platforms/base.py  → added doc types
# - tests/tools/test_memory_tool.py
# - tests/tools/test_memory_tool_import_fallback.py

# Commit and push
git add tools/knowledge_tool.py tools/knowledge_fallback.py tools/memory_tool.py \
        gateway/platforms/base.py tests/tools/test_memory_tool.py tests/tools/test_memory_tool_import_fallback.py
git commit -m "custom: knowledge-db tool integration + document type whitelist"
git push origin main --force
pip install -e .
hermes gateway restart
```

## Verification Checklist

After ANY recovery, verify:
```bash
# 1. Hermes version
hermes --version

# 2. Knowledge tool works
~/.hermes/bin/knowledge search facts "project:wwa" | head -3

# 3. Gateway running
hermes gateway status

# 4. Cron jobs healthy
hermes cron list | head -10

# 5. Git state clean
cd ~/.hermes/hermes-agent && git status

# 6. Custom files present
ls -la tools/knowledge_tool.py tools/knowledge_fallback.py

# 7. Import works
python3.13 -c "from tools.knowledge_tool import knowledge_write; print('OK')"
```

## Prevention

- **Never** commit directly to upstream (we don't have push access anyway)
- **Never** force push to origin/main without verifying our commit is in the history
- **Always** run verification checklist after any git operation that touches our branch
- The `daily-hermes-update` cron (0cd913415a2b) stashes before rebase — if it fails, it aborts and reports
- Keep this RECOVERY.md in the repo root — agents and humans can both read it

## Backup Locations

- **Fork:** `github.com/vystartasv/hermes-agent` (our commit is safe here)
- **Knowledge-db:** `~/.hermes/knowledge/` backed up every 3h to `vystartasv/hermes-knowledge`
- **This manual:** In repo root + knowledge-db fact `domain:infra project:knowledge-db type:recovery`
