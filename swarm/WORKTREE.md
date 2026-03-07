# Worktree Manager

Git worktree orchestration for branch-per-agent isolation in the swarm system.

## Overview

Each swarm agent gets its own git worktree with a dedicated branch (`swarm/<agent_id>`), allowing parallel, isolated file modifications without conflicts.

## Usage

```python
from swarm.worktree import WorktreeManager
from pathlib import Path

mgr = WorktreeManager(Path("/path/to/repo"))

# Create isolated worktree for an agent
wt_path = mgr.create("agent-001", base_branch="main")

# Check for changes, commit, get diff
if mgr.has_changes("agent-001"):
    diff = mgr.get_diff("agent-001")
    sha = mgr.commit("agent-001", "feat: implement widget")

# Merge back to main
success = mgr.merge_to("agent-001", target_branch="main", strategy="squash")

# Cleanup
mgr.cleanup("agent-001")
mgr.cleanup_all()
```

## API

| Method | Description |
|---|---|
| `create(agent_id, base_branch="main")` | Create worktree + branch `swarm/{agent_id}`, returns worktree path |
| `get_path(agent_id)` | Get worktree path or `None` |
| `list_active()` | List all active swarm worktrees (`agent_id`, `path`, `branch`) |
| `has_changes(agent_id)` | Check for uncommitted changes |
| `commit(agent_id, message)` | Stage all + commit, returns commit hash |
| `get_diff(agent_id)` | Combined staged + unstaged diff |
| `merge_to(agent_id, target, strategy)` | Merge agent branch into target (`merge`/`squash`/`rebase`). Returns `False` on conflict. |
| `cleanup(agent_id)` | Remove worktree and delete branch |
| `cleanup_all()` | Remove all swarm worktrees and prune |

## Merge Strategies

- **merge** (default): `--no-ff` merge commit preserving full history
- **squash**: Single commit on target with all changes combined
- **rebase**: Replay agent commits onto target

## Thread Safety

All git-mutating operations are serialized via a threading lock. Read-only operations (`has_changes`, `get_diff`) do not acquire the lock.

## Directory Layout

```
repo/
  .swarm-worktrees/
    agent-001/    # worktree on branch swarm/agent-001
    agent-002/    # worktree on branch swarm/agent-002
```
