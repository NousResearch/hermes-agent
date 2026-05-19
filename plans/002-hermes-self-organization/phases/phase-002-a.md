# Phase 002-A: Directory Layout Migration

## Goal
Create the canonical `~/.hermes/users/{id}/` directory tree and add path-resolver functions to `hermes_constants.py` so all future code uses named accessors — no hardcoded paths. Strictly additive: existing paths remain untouched.

## Context
Today `~/.hermes/` is flat and user-unaware. Skills, memories, plans, logs, and session state all live at the same depth with no namespace separation. The goal of this phase is to create the *shape* of the new world without breaking anything in the old one. Phase B and D depend on the path constants this phase adds. Memory relocation (`memories/` → `users/{id}/memory/`) happens here via a migration script; the read path in `memory_tool.py` is updated to check the new location first.

## Dependencies
None — this phase has no upstream dependencies. Can start immediately.

## Scope

### Files to Create
- `~/.hermes/users/blake/` — user namespace root (directory tree, via migration script)
- `~/.hermes/users/blake/memory/` — future home of MEMORY.md + USER.md
- `~/.hermes/users/blake/skills/` — user-authored skills (not yet migrated; structure only)
- `~/.hermes/users/blake/plans/active/` — active plans for this user
- `~/.hermes/users/blake/plans/archive/` — completed plans (YYYY-MM subdirs created at archive time)
- `~/.hermes/users/blake/artifacts/sessions/` — promoted session outputs
- `~/.hermes/users/blake/artifacts/projects/` — ongoing project work outputs
- `~/.hermes/users/blake/credentials/` — credential refs only (no values)
- `~/.hermes/users/blake/credentials/README.md` — explains ref-only contract
- `~/.hermes/system/skills/` — hub-installed skills (not yet populated; structure only)
- `~/.hermes/system/mcp-servers.json` — empty `{}` stub (populated in Phase C)
- `~/.hermes/runtime/sessions/` — ephemeral session sandboxes (Phase B writes here)
- `~/.hermes/teams/` — future team-scoped namespace (structure only)
- `scripts/migrate_002a.sh` (in the hermes-agent repo) — creates all dirs, copies `memories/` → `users/blake/memory/`, reports what it did

### Files to Modify
- `hermes_constants.py` — add 8 new path-resolver functions (see Implementation Notes)
- `tools/memory_tool.py` — update `get_memory_dir()` to check new location first, fall back to legacy

### Explicitly Out of Scope
- Moving any skills files — skills migration is a separate effort (post-002)
- Moving any plan files — plan files are in `~/Documents/hermes-agent/plans/`, not `~/.hermes/`
- Actually populating `system/skills/` or `system/mcp-servers.json`
- Changing any auth/credential files — Phase B handles runtime injection

## Implementation Notes

### New functions to add to `hermes_constants.py`

Add these after the existing `get_skills_dir()` function (line ~286). Follow the exact same pattern as `get_hermes_home()` — check env var first, return `Path`:

```python
def get_users_root() -> Path:
    """Return the users namespace root, overridable via HERMES_USERS_ROOT."""
    override = os.environ.get("HERMES_USERS_ROOT", "").strip()
    if override:
        return Path(override)
    return get_hermes_home() / "users"

def get_user_home(user_id: str) -> Path:
    """Return the home directory for a specific user under the users root."""
    return get_users_root() / user_id

def get_memory_root(user_id: str) -> Path:
    """Return the memory directory for a user (MEMORY.md, USER.md live here)."""
    return get_user_home(user_id) / "memory"

def get_plans_root(user_id: str) -> Path:
    """Return the plans directory for a user."""
    return get_user_home(user_id) / "plans"

def get_skills_root(user_id: str) -> Path:
    """Return the user-authored skills directory."""
    return get_user_home(user_id) / "skills"

def get_artifacts_root(user_id: str) -> Path:
    """Return the artifacts directory for a user."""
    return get_user_home(user_id) / "artifacts"

def get_credentials_root(user_id: str) -> Path:
    """Return the credentials ref directory for a user (refs only, never values)."""
    return get_user_home(user_id) / "credentials"

def get_runtime_root() -> Path:
    """Return the ephemeral runtime root, overridable via HERMES_RUNTIME_ROOT."""
    override = os.environ.get("HERMES_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override)
    return get_hermes_home() / "runtime"
```

### Update `get_memory_dir()` in `tools/memory_tool.py`

The current implementation (line ~56) is:
```python
def get_memory_dir() -> Path:
    return get_hermes_home() / "memories"
```

Replace with a migration-aware version that checks the new canonical path first:
```python
def get_memory_dir() -> Path:
    """Return the memory directory.

    Checks the new canonical location (users/{id}/memory/) first.
    Falls back to the legacy ~/.hermes/memories/ for backward compatibility.
    Migration: run scripts/migrate_002a.sh to move files to canonical location.
    """
    # Determine user_id from env (set by session runtime in Phase B) or config
    user_id = os.environ.get("HERMES_USER_ID", "").strip()
    if user_id:
        from hermes_constants import get_memory_root
        canonical = get_memory_root(user_id)
        if canonical.exists():
            return canonical
    # Legacy fallback — always works before migration
    return get_hermes_home() / "memories"
```

### Migration script logic (`scripts/migrate_002a.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
USER_ID="${HERMES_USER_ID:-blake}"
USER_HOME="$HERMES_HOME/users/$USER_ID"

# Create full directory tree
mkdir -p \
  "$USER_HOME/memory" \
  "$USER_HOME/skills" \
  "$USER_HOME/plans/active" \
  "$USER_HOME/plans/archive" \
  "$USER_HOME/artifacts/sessions" \
  "$USER_HOME/artifacts/projects" \
  "$USER_HOME/credentials" \
  "$HERMES_HOME/system/skills" \
  "$HERMES_HOME/runtime/sessions" \
  "$HERMES_HOME/teams"

# Copy memory files (not move — keep legacy fallback working)
if [ -d "$HERMES_HOME/memories" ]; then
  cp -n "$HERMES_HOME/memories/MEMORY.md" "$USER_HOME/memory/MEMORY.md" 2>/dev/null || true
  cp -n "$HERMES_HOME/memories/USER.md"   "$USER_HOME/memory/USER.md"   2>/dev/null || true
  echo "Copied memories → $USER_HOME/memory/ (legacy path preserved)"
fi

# Write system/mcp-servers.json stub
echo '{}' > "$HERMES_HOME/system/mcp-servers.json"
echo "Phase 002-A migration complete. Tree rooted at $USER_HOME"
```

### `credentials/README.md` content

```markdown
# Credentials Directory — Refs Only

This directory contains ONLY credential references (URIs pointing to
Keychain or Secrets Manager entries) — never actual secret values.

## Format

Each file is a JSON ref:
  { "type": "keychain", "service": "hermes/blake/gmail", "account": "blake" }
  { "type": "secrets-manager", "arn": "arn:aws:secretsmanager:us-east-1:..." }

## Contract

- CredentialResolver (Phase 002-B) reads these refs and injects resolved
  values as ephemeral env vars into the session sandbox
- Resolved values exist only in memory for the session lifetime
- These files are safe to commit (they contain no secrets)
```

## Acceptance Criteria
- [ ] `~/.hermes/users/blake/` tree exists with all listed subdirs after running `scripts/migrate_002a.sh`
- [ ] `get_user_home("blake")` returns `Path("~/.hermes/users/blake")` (expanded)
- [ ] `get_memory_root("blake")` returns `Path("~/.hermes/users/blake/memory")`
- [ ] Setting `HERMES_USERS_ROOT=/tmp/test-users` causes `get_user_home("blake")` to return `/tmp/test-users/blake`
- [ ] `~/.hermes/users/blake/memory/MEMORY.md` exists with same content as `~/.hermes/memories/MEMORY.md`
- [ ] Legacy `~/.hermes/memories/MEMORY.md` still exists (no destructive migration)
- [ ] With `HERMES_USER_ID=blake`, `get_memory_dir()` returns the new canonical path
- [ ] Without `HERMES_USER_ID` set, `get_memory_dir()` returns legacy `~/.hermes/memories/`
- [ ] `pytest tests/ -v` — zero regressions (all existing tests pass)
- [ ] `hermes` CLI starts and completes one full turn without errors

## Verification Steps

```bash
# 1. Run migration script
cd ~/Documents/hermes-agent
bash scripts/migrate_002a.sh

# 2. Verify directory tree
find ~/.hermes/users/blake -type d | sort
# Expected: users/blake/, memory/, skills/, plans/, plans/active/, plans/archive/, 
#           artifacts/, artifacts/sessions/, artifacts/projects/, credentials/

# 3. Verify system dirs
ls ~/.hermes/system/
# Expected: mcp-servers.json  skills/

# 4. Verify memory copy
diff ~/.hermes/memories/MEMORY.md ~/.hermes/users/blake/memory/MEMORY.md
# Expected: no diff (identical)

# 5. Test path constants
cd ~/Documents/hermes-agent
python3 -c "
from hermes_constants import get_user_home, get_memory_root, get_users_root
import os
os.environ['HERMES_USERS_ROOT'] = '/tmp/test-users'
assert str(get_user_home('blake')) == '/tmp/test-users/blake'
del os.environ['HERMES_USERS_ROOT']
print('get_user_home OK')
print('get_memory_root:', get_memory_root('blake'))
"

# 6. Test memory fallback
python3 -c "
import os
# Without user id — legacy path
from tools.memory_tool import get_memory_dir
path = get_memory_dir()
print('legacy path:', path)
assert 'memories' in str(path), f'Expected legacy path, got {path}'

# With user id and canonical dir present — new path
os.environ['HERMES_USER_ID'] = 'blake'
from importlib import reload
import tools.memory_tool as m
reload(m)
path = m.get_memory_dir()
print('canonical path:', path)
assert 'users/blake/memory' in str(path), f'Expected canonical path, got {path}'
"

# 7. Run tests
pytest tests/ -v --tb=short 2>&1 | tail -20
# Expected: no FAILED lines

# 8. Smoke test CLI
hermes --help
echo "hello" | hermes --one-shot
```

## Status
Not started

## Bug Log
| # | Description | Status |
|---|-------------|--------|
