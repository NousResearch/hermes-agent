# Checkpoint & Rollback — Implementation Plan

## Goal

Automatic filesystem snapshots before destructive file operations, with user-facing rollback. The agent never sees or interacts with this — it's transparent infrastructure.

## Design Principles

1. **Not a tool** — the LLM never knows about it. Zero prompt tokens, zero tool schema overhead.
2. **Once per turn** — checkpoint at most once per conversation turn (user message → agent response cycle), triggered lazily on the first file-mutating operation. Not on every write.
3. **Opt-in via config** — disabled by default, enabled with `checkpoints: true` in config.yaml.
4. **Works on any directory** — uses a shadow git repo completely separate from the user's project git. Works on git repos, non-git directories, anything.
5. **User-facing rollback** — `/rollback` slash command (CLI + gateway) to list and restore checkpoints. Also `hermes rollback` CLI subcommand.

## Architecture

```
~/.hermes/checkpoints/
  {sha256(abs_dir)[:16]}/       # Shadow git repo per working directory
    HEAD, refs/, objects/...    # Standard git internals
    HERMES_WORKDIR              # Original dir path (for display)
    info/exclude                # Default excludes (node_modules, .env, etc.)
```

### Core: CheckpointManager (new file: tools/checkpoint_manager.py)

Adapted from PR #559's CheckpointStore. Key changes from the PR:

- **Not a tool** — no schema, no registry entry, no handler
- **Turn-scoped deduplication** — tracks `_checkpointed_dirs: Set[str]` per turn
- **Configurable** — reads `checkpoints` config key
- **Pruning** — keeps last N snapshots per directory (default 50), prunes on take

```python
class CheckpointManager:
    def __init__(self, enabled: bool = False, max_snapshots: int = 50):
        self.enabled = enabled
        self.max_snapshots = max_snapshots
        self._checkpointed_dirs: Set[str] = set()  # reset each turn

    def new_turn(self):
        """Call at start of each conversation turn to reset dedup."""
        self._checkpointed_dirs.clear()

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> None:
        """Take a checkpoint if enabled and not already done this turn."""
        if not self.enabled:
            return
        abs_dir = str(Path(working_dir).resolve())
        if abs_dir in self._checkpointed_dirs:
            return
        self._checkpointed_dirs.add(abs_dir)
        try:
            self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)

    def list_checkpoints(self, working_dir: str) -> List[dict]:
        """List available checkpoints for a directory."""
        ...

    def restore(self, working_dir: str, commit_hash: str) -> dict:
        """Restore files to a checkpoint state."""
        ...

    def _take(self, working_dir: str, reason: str):
        """Shadow git: add -A + commit. Prune if over max_snapshots."""
        ...

    def _prune(self, shadow_repo: Path):
        """Keep only last max_snapshots commits."""
        ...
```

### Integration Point: run_agent.py

The AIAgent already owns the conversation loop. Add CheckpointManager as an instance attribute:

```python
class AIAgent:
    def __init__(self, ...):
        ...
        # Checkpoint manager — reads config to determine if enabled
        self._checkpoint_mgr = CheckpointManager(
            enabled=config.get("checkpoints", False),
            max_snapshots=config.get("checkpoint_max_snapshots", 50),
        )
```

**Turn boundary** — in `run_conversation()`, call `new_turn()` at the start of each agent iteration (before processing tool calls):

```python
# Inside the main loop, before _execute_tool_calls():
self._checkpoint_mgr.new_turn()
```

**Trigger point** — in `_execute_tool_calls()`, before dispatching file-mutating tools:

```python
# Before the handle_function_call dispatch:
if function_name in ("write_file", "patch"):
    # Determine working dir from the file path in the args
    file_path = function_args.get("path", "") or function_args.get("old_string", "")
    if file_path:
        work_dir = str(Path(file_path).parent.resolve())
        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
```

This means:
- First `write_file` in a turn → checkpoint (fast, one `git add -A && git commit`)
- Subsequent writes in the same turn → no-op (already checkpointed)
- Next turn (new user message) → fresh checkpoint eligibility

### Config

Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`:

```python
"checkpoints": False,          # Enable filesystem checkpoints before destructive ops
"checkpoint_max_snapshots": 50, # Max snapshots to keep per directory
```

User enables with:
```yaml
# ~/.hermes/config.yaml
checkpoints: true
```

### User-Facing Rollback

**CLI slash command** — add `/rollback` to `process_command()` in `cli.py`:

```
/rollback         — List recent checkpoints for the current directory
/rollback <hash>  — Restore files to that checkpoint
```

Shows a numbered list:
```
📸 Checkpoints for /home/user/project:
  1. abc1234  2026-03-09 21:15  before write_file (3 files changed)
  2. def5678  2026-03-09 20:42  before patch (1 file changed)
  3. ghi9012  2026-03-09 20:30  before write_file (2 files changed)

Use /rollback <number> to restore, e.g. /rollback 1
```

**Gateway slash command** — add `/rollback` to gateway/run.py with the same behavior.

**CLI subcommand** — `hermes rollback` (optional, lower priority).

### What Gets Excluded (not checkpointed)

Same as the PR's defaults — written to the shadow repo's `info/exclude`:

```
node_modules/
dist/
build/
.env
.env.*
__pycache__/
*.pyc
.DS_Store
*.log
.cache/
.venv/
.git/
```

Also respects the project's `.gitignore` if present (shadow repo can read it via `core.excludesFile`).

### Safety

- `ensure_checkpoint()` wraps everything in try/except — a checkpoint failure never blocks the actual file operation
- Shadow repo is completely isolated — GIT_DIR + GIT_WORK_TREE env vars, never touches user's .git
- If git isn't installed, checkpoints silently disable
- Large directories: add a file count check — skip checkpoint if >50K files to avoid slowdowns

## Files to Create/Modify

| File | Change |
|------|--------|
| `tools/checkpoint_manager.py` | **NEW** — CheckpointManager class (adapted from PR #559) |
| `run_agent.py` | Add CheckpointManager init + trigger in `_execute_tool_calls()` |
| `hermes_cli/config.py` | Add `checkpoints` + `checkpoint_max_snapshots` to DEFAULT_CONFIG |
| `cli.py` | Add `/rollback` slash command handler |
| `gateway/run.py` | Add `/rollback` slash command handler |
| `tests/tools/test_checkpoint_manager.py` | **NEW** — tests (adapted from PR #559's tests) |

## What We Take From PR #559

- `_shadow_repo_path()` — deterministic path hashing ✅
- `_git_env()` — GIT_DIR/GIT_WORK_TREE isolation ✅
- `_run_git()` — subprocess wrapper with timeout ✅
- `_init_shadow_repo()` — shadow repo initialization ✅
- `DEFAULT_EXCLUDES` list ✅
- Test structure and patterns ✅

## What We Change From PR #559

- **Remove tool schema/registry** — not a tool
- **Remove injection into file_operations.py and patch_parser.py** — trigger from run_agent.py instead
- **Add turn-scoped deduplication** — one checkpoint per turn, not per operation
- **Add pruning** — keep last N snapshots
- **Add config flag** — opt-in, not mandatory
- **Add /rollback command** — user-facing restore UI
- **Add file count guard** — skip huge directories

## Implementation Order

1. `tools/checkpoint_manager.py` — core class with take/list/restore/prune
2. `tests/tools/test_checkpoint_manager.py` — tests
3. `hermes_cli/config.py` — config keys
4. `run_agent.py` — integration (init + trigger)
5. `cli.py` — `/rollback` slash command
6. `gateway/run.py` — `/rollback` slash command
7. Full test suite run + manual smoke test
