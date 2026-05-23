# Memory Steward

## Purpose

Reduce resource pressure from stale memories, outdated preferences, and accumulated artifacts - while preserving all project-critical knowledge. Works across both Town (memories, skills, content library) and Hermes (sessions, caches, logs) environments.

Default mode: READ-ONLY AUDIT.

---

## What is never touched (forbidden list)

Regardless of what the user asks, NEVER delete, move, modify, or clear any of the following:

### Town Environment
- User profile (user.md)
- Active routine configurations
- Content Library items without explicit user confirmation
- Memories tagged to active routines (unless user confirms they are stale)

### Hermes Environment
- SYSTEM_STATE.md
- MEMORY.md
- specs/changes/
- docs/MODEL_DOCUMENTATION.md
- Any committed source file (check with git status before touching anything)
- production_data/ and any production snapshots
- .credentials.json
- auth-profiles.json
- Any active cron entry or job definition
- Active Hermes job roster
- Any audit/spec artifact from the current operating cycle
- Any file whose loss cannot be recovered from git

---

## Safe cleanup candidates (after explicit approval only)

### Town Environment
- Global memories that duplicate routine-specific memories
- Memories for routines that no longer exist
- Stale content library items in uncategorized/
- Outdated people research documents in Memories/people/

### Hermes Environment
- OpenClaw/Hermes sessions older than a user-specified threshold
- Failed or abandoned task records
- Temporary scratch files (tmp/, scratch/, *.tmp, *.bak)
- Old run logs beyond retention window
- Duplicate generated artifacts
- Cache directories that can be regenerated (__pycache__, .cache/, etc.)
- Obsolete untracked tool outputs

---

## Audit Steps

### Town Audit (run all, read-only)

1. **Global memories**: List all via `get_memories()`. Flag duplicates, contradictions, or stale entries.
2. **Routine-specific memories**: For each active routine, call `get_memories(routine_slug=...)`. Flag memories that reference deprecated behavior or routines.
3. **Skills inventory**: List via `town_ls skills://`. Flag skills that may be unused or outdated.
4. **Content Library**: Browse via `town_ls content://collections`. Flag large uncategorized collections or stale items.
5. **People documents**: Check Memories/people/ for outdated or duplicate profiles.

### Hermes Audit (run all, read-only)

6. Hermes job roster and recent run health
7. OpenClaw agent/session status
8. Disk usage on likely bloat locations
9. Large log files (>1M) and old log files (>7 days)
10. Stale sessions by age
11. Temporary and scratch files in project
12. Cache directories
13. Hermes job history files

---

## Required Audit Deliverable

### 1. Current Memory/Resource Status
Summary of active memories (global and per-routine), skills, and content library state.

### 2. Largest Resource Consumers
What's taking up the most space or context budget.

### 3. Stale Items by Age
Memories, skills, or content that haven't been referenced or updated in >30 days.

### 4. Safe Cleanup Candidates
Table: Item | Type | Age | Why Safe

### 5. Unsafe Candidates
Table: Item | Type | Why Unsafe (what would be lost)

### 6. Exact Proposed Actions
One action per target. No broad sweeps.

### 7. Estimated Risk Per Candidate
NONE / LOW / MEDIUM / HIGH with brief rationale.

### 8. Rollback Plan
- What to back up or note before each action
- How to recover if something goes wrong
- For Town memories: note the content before deleting so it can be re-added

---

## Decision Options

- AUDIT_ONLY - no action taken, report only
- CLEAN_STALE_MEMORIES - remove confirmed-stale memories (Town)
- CLEAN_SAFE_CACHES - delete only regenerable cache dirs (Hermes)
- CLEAN_STALE_SESSIONS - archive/remove specific stale sessions (Hermes)
- FULL_APPROVAL_REQUIRED - approve each item individually

STOP after delivering audit. Do not execute any cleanup until user explicitly says proceed.

---

## Execution Rules (when approved)

- For Town memories: note the full content before calling `delete_memory`
- For Hermes files: back up before deleting (cp -r or tar archive)
- Prefer mv to ~/archive/ over rm for Hermes artifacts
- Never rm -rf without first listing exact targets
- Execute one category at a time, confirm after each
- If anything is ambiguous, skip it and flag for manual review
