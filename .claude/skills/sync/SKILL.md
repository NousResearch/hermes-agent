---
name: sync
version: "1.0.0"
description: Syncs WDS skills from the current project (_bmad/wds/) to ~/.claude/commands/ so they work in any project. Called automatically on every agent activation.
agents: [saga, freya, mimir]
---

# WDS Sync

Keeps `~/.claude/commands/` current with the WDS installation in the active project.
Agents call this silently on every activation. Users can also call it directly.

---

## Entry points

**On agent activation — silent mode**
- First time (no global commands exist): ask the user once, then sync.
- Subsequent activations: compare files, sync silently if changed, skip if identical.
- Never block activation. If sync fails for any reason, continue.

**Direct user request** ("sync", "update WDS", "sync skills", "check for updates") — verbose, report each step.

---

## Steps

### 1 — Locate project installation

Find `_bmad/wds/` relative to the current project root.

If `_bmad/wds/` does not exist:
- Silent mode: stop silently.
- Verbose mode: `⚠️ WDS not installed in this project. Run: npx bmad-method install`

Store the source root: `WDS_SRC = {project-root}/_bmad/wds`

---

### 2 — Detect home directory

```bash
# Mac / Linux
HOME_DIR="$HOME"

# Windows (PowerShell)
$HOME_DIR = $env:USERPROFILE
```

Destination directories:
- Commands: `{HOME_DIR}/.claude/commands/`
- Tools:    `{HOME_DIR}/.claude/wds/tools/memory/`
- Data:     `{HOME_DIR}/.claude/wds/data/`

---

### 3 — Check sync state

Check if `{HOME_DIR}/.claude/commands/wrap.md` exists.

**First time (file missing):** go to step 4 — ask user.

**Already synced:** compare `{HOME_DIR}/.claude/commands/wrap.md` to `{WDS_SRC}/skills/wrap.md`.

```bash
# Mac / Linux
diff -q "{HOME_DIR}/.claude/commands/wrap.md" "{WDS_SRC}/skills/wrap.md"

# Windows
Compare-Object (Get-Content "{HOME_DIR}\.claude\commands\wrap.md") (Get-Content "{WDS_SRC}\skills\wrap.md")
```

If identical: finish silently.
If different: go to step 5 — sync silently.

---

### 4 — First time: ask user

Print exactly:

```
WDS is installed in this project but skills are not yet available globally.
Sync now? Adds /saga /freya /mimir /start /wrap /handoff to all Claude Code sessions.
[Y/n]
```

If **n**: stop. Do not ask again this session.
If **y** (or Enter): go to step 5.

---

### 5 — Sync files

Create destination directories if they don't exist:

```bash
# Mac / Linux
mkdir -p "{HOME_DIR}/.claude/commands"
mkdir -p "{HOME_DIR}/.claude/wds/tools/memory"
mkdir -p "{HOME_DIR}/.claude/wds/data"

# Windows
New-Item -ItemType Directory -Force "{HOME_DIR}\.claude\commands"
New-Item -ItemType Directory -Force "{HOME_DIR}\.claude\wds\tools\memory"
New-Item -ItemType Directory -Force "{HOME_DIR}\.claude\wds\data"
```

Copy agent commands:

| Source | Destination |
|--------|-------------|
| `{WDS_SRC}/skills/saga.activation.md`  | `{HOME_DIR}/.claude/commands/saga.md`    |
| `{WDS_SRC}/skills/freya.activation.md` | `{HOME_DIR}/.claude/commands/freya.md`   |
| `{WDS_SRC}/skills/mimir.activation.md` | `{HOME_DIR}/.claude/commands/mimir.md`   |
| `{WDS_SRC}/skills/start.md`            | `{HOME_DIR}/.claude/commands/start.md`   |
| `{WDS_SRC}/skills/wrap.md`             | `{HOME_DIR}/.claude/commands/wrap.md`    |
| `{WDS_SRC}/skills/handoff.md`          | `{HOME_DIR}/.claude/commands/handoff.md` |

Copy tools and data:

| Source | Destination |
|--------|-------------|
| `{WDS_SRC}/tools/memory/SKILL.md`    | `{HOME_DIR}/.claude/wds/tools/memory/SKILL.md`    |
| `{WDS_SRC}/data/wds-glossary.md`     | `{HOME_DIR}/.claude/wds/data/wds-glossary.md`     |
| `{WDS_SRC}/data/agent-contracts.md`  | `{HOME_DIR}/.claude/wds/data/agent-contracts.md`  |
| `{WDS_SRC}/data/shared-activation.md`| `{HOME_DIR}/.claude/wds/data/shared-activation.md`|

Skip any source file that does not exist — do not error.

---

### 6 — Report

**First-time / direct call:**
```
✓ WDS skills synced to ~/.claude/commands/
  /saga  /freya  /mimir  /start  /wrap  /handoff
```

**Silent update (changes detected):**
```
✓ WDS skills updated.
```

**No changes (direct call only):**
```
WDS skills are up to date.
```

**Silent mode with no changes:** no output.
