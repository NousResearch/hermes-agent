# Phase 002-D: Cloud Env-Var Routing

## Goal
Make every path-returning function in `hermes_constants.py` check an env var override before returning the local default. Validate these vars at startup. Document the full env var contract. Result: pointing any subtree at cloud storage requires zero code changes — just set an env var.

## Context
`hermes_constants.py` already implements this pattern for `HERMES_HOME` (line 30). `get_memory_dir()` in `memory_tool.py` checks `HERMES_USER_ID` as of Phase A. This phase extends the same pattern to all new path functions added in Phase A, adds a startup validator, and documents the contract. This is deliberately a low-risk, high-leverage change — it unlocks cloud decomposition without any architectural rework.

Phase D can run in parallel with Phases B and C once Phase A is complete, since it only touches `hermes_constants.py` (no session or MCP code).

## Dependencies
- **Phase 002-A must be complete** — adds the path functions that this phase enhances

## Scope

### Files to Create
- `scripts/validate_env.py` — startup env var validator (warns on missing paths, errors on malformed URIs)

### Files to Modify
- `hermes_constants.py` — update 8 new path functions (from Phase A) to check their respective env vars before returning defaults
- `run_agent.py` — call `validate_env.validate_hermes_env()` early in `AIAgent.__init__` to surface misconfigurations at startup
- `docs/env-vars.md` (create if not exists) — canonical reference for all `HERMES_*` env vars

### Explicitly Out of Scope
- Actually implementing S3 or remote backends — env vars just change the *path prefix*; the code that reads/writes those paths still uses local filesystem ops. A future cloud phase would swap in a storage backend abstraction.
- Session/runtime env vars (`HERMES_SESSION_ID`, `HERMES_USER_ID`) — these are set by the agent process, not by operators. Not validated here.
- Changing `HERMES_HOME` behavior — it's already implemented and tested; we don't touch it.

## Implementation Notes

### Updated path functions in `hermes_constants.py`

Each function from Phase A gets its env var check. The pattern is identical for all of them:

```python
def get_users_root() -> Path:
    """Return the users namespace root.
    
    Override: HERMES_USERS_ROOT (e.g., /mnt/shared/hermes-users or s3://... — note:
    s3:// URIs require a FUSE mount or storage backend, not native Path ops)
    """
    override = os.environ.get("HERMES_USERS_ROOT", "").strip()
    return Path(override) if override else get_hermes_home() / "users"

def get_runtime_root() -> Path:
    """Return the ephemeral runtime root.
    
    Override: HERMES_RUNTIME_ROOT (should be local/fast storage — /tmp or tmpfs)
    """
    override = os.environ.get("HERMES_RUNTIME_ROOT", "").strip()
    return Path(override) if override else get_hermes_home() / "runtime"

# get_user_home, get_memory_root, get_plans_root, get_skills_root,
# get_artifacts_root, get_credentials_root — all delegate to get_users_root()
# so they automatically inherit the HERMES_USERS_ROOT override. No per-function
# env var needed for user-scoped paths.

def get_mcp_servers_config() -> Path:
    """Return path to system/mcp-servers.json.
    
    Override: HERMES_MCP_SERVERS_CONFIG (absolute path to a JSON file)
    """
    override = os.environ.get("HERMES_MCP_SERVERS_CONFIG", "").strip()
    return Path(override) if override else get_hermes_home() / "system" / "mcp-servers.json"
```

### Full env var contract

| Variable | Affects | Default | Notes |
|----------|---------|---------|-------|
| `HERMES_HOME` | Entire `~/.hermes/` root | `~/.hermes/` | Pre-existing; not changed |
| `HERMES_USERS_ROOT` | `users/` subtree | `$HERMES_HOME/users/` | All user-scoped paths inherit |
| `HERMES_RUNTIME_ROOT` | `runtime/` subtree | `$HERMES_HOME/runtime/` | Should be local, fast storage |
| `HERMES_MCP_SERVERS_CONFIG` | `system/mcp-servers.json` | `$HERMES_HOME/system/mcp-servers.json` | Path to JSON file |
| `HERMES_USER_ID` | Which user's subdirectory is active | (empty — no user-scoped paths used) | Set per-session by `AIAgent.__init__` |
| `HERMES_SESSION_ID` | Session identity | (auto-generated) | Set by `AIAgent.__init__`; not an operator var |

### `scripts/validate_env.py`

```python
"""validate_env.py — validate HERMES_* env vars at startup.

Called from AIAgent.__init__. Warns on missing dirs, errors on malformed URIs.
Never blocks startup for missing dirs (operator may be creating them lazily).
Only blocks on clearly wrong configs (e.g., HERMES_USERS_ROOT=s3://... when
no FUSE mount is present and no backend abstraction exists).
"""
from __future__ import annotations

import logging
import os
import sys
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# Env vars that should be local filesystem paths (not URIs)
_LOCAL_PATH_VARS = [
    "HERMES_USERS_ROOT",
    "HERMES_RUNTIME_ROOT",
    "HERMES_MCP_SERVERS_CONFIG",
]

# URI schemes that are NOT yet supported (no backend abstraction)
_UNSUPPORTED_SCHEMES = {"s3", "gs", "az", "dynamodb", "postgres", "postgresql"}


def validate_hermes_env(*, strict: bool = False) -> None:
    """Validate all HERMES_* env vars.
    
    Args:
        strict: If True, raise ValueError on any warning-level issue.
                If False (default), log warnings and continue.
    """
    errors = []
    warnings = []

    for var in _LOCAL_PATH_VARS:
        val = os.environ.get(var, "").strip()
        if not val:
            continue  # unset = use default, always fine
        
        # Check for URI scheme (unsupported without backend abstraction)
        parsed = urllib.parse.urlparse(val)
        if parsed.scheme in _UNSUPPORTED_SCHEMES:
            errors.append(
                f"{var}={val!r}: URI scheme '{parsed.scheme}://' is not yet supported. "
                f"Use a local path or FUSE mount."
            )
            continue
        
        # Check path exists (warn only — lazy creation is valid)
        p = Path(val)
        if not p.exists():
            warnings.append(f"{var}={val!r}: path does not exist (will be created on first use)")
    
    for w in warnings:
        logger.warning("hermes env: %s", w)
    
    if errors:
        for e in errors:
            logger.error("hermes env: %s", e)
        if strict:
            raise ValueError(f"Invalid HERMES_* env vars: {errors}")
        else:
            # Non-strict: log and continue — don't block startup
            logger.warning("hermes env: %d error(s) above — proceeding anyway (non-strict mode)", len(errors))
```

### `docs/env-vars.md`

Create this file to serve as the canonical operator reference. Content should match the table above, plus examples for common deployment scenarios:

```markdown
# HERMES Environment Variables

## Operator Variables (set before starting Hermes)

| Variable | Description | Default |
|----------|-------------|---------|
| `HERMES_HOME` | Root of Hermes state directory | `~/.hermes/` |
| `HERMES_USERS_ROOT` | Parent of all user namespaces | `$HERMES_HOME/users/` |
| `HERMES_RUNTIME_ROOT` | Ephemeral session sandboxes (use fast local storage) | `$HERMES_HOME/runtime/` |
| `HERMES_MCP_SERVERS_CONFIG` | Path to system MCP server definitions JSON | `$HERMES_HOME/system/mcp-servers.json` |

## Runtime Variables (set by Hermes itself — do not set manually)

| Variable | Set by | Purpose |
|----------|--------|---------|
| `HERMES_HOME` | CLI startup | Profile resolution |
| `HERMES_SESSION_ID` | `AIAgent.__init__` | Session identity |
| `HERMES_USER_ID` | `AIAgent.__init__` | User namespace routing |

## Deployment Examples

### Local single-user (default)
No vars needed. All state in `~/.hermes/`.

### Shared NFS mount (team)
```bash
export HERMES_USERS_ROOT=/mnt/team-hermes/users
export HERMES_RUNTIME_ROOT=/tmp/hermes-runtime   # keep runtime local/fast
```

### Docker container
```bash
export HERMES_HOME=/opt/hermes-state             # mapped volume
export HERMES_RUNTIME_ROOT=/tmp/hermes-runtime   # tmpfs in container
```
```

## Acceptance Criteria
- [ ] Setting `HERMES_USERS_ROOT=/tmp/test-users` causes `get_user_home("blake")` to return `/tmp/test-users/blake`
- [ ] Setting `HERMES_RUNTIME_ROOT=/tmp/rt` causes `get_runtime_root()` to return `/tmp/rt`
- [ ] Setting `HERMES_MCP_SERVERS_CONFIG=/tmp/my-mcp.json` causes `get_mcp_servers_config()` to return `/tmp/my-mcp.json`
- [ ] Setting `HERMES_USERS_ROOT=s3://my-bucket/hermes` causes `validate_hermes_env()` to log an error (non-strict: does not raise)
- [ ] Setting `HERMES_USERS_ROOT=s3://my-bucket` with `strict=True` causes `validate_hermes_env()` to raise `ValueError`
- [ ] Setting `HERMES_USERS_ROOT=/nonexistent/path` causes a `logger.warning` (not an error)
- [ ] `docs/env-vars.md` exists and documents all 4 operator vars
- [ ] `pytest tests/test_constants.py -v` — all pass (new tests to write as part of this phase)
- [ ] `pytest tests/ -v` — zero regressions

## Verification Steps

```bash
# 1. Test env var overrides
cd ~/Documents/hermes-agent
python3 - <<'EOF'
import os

# Test HERMES_USERS_ROOT override
os.environ["HERMES_USERS_ROOT"] = "/tmp/test-users"
# Force reimport to pick up env var (constants are functions, not module-level vars)
from hermes_constants import get_user_home, get_memory_root, get_runtime_root, get_mcp_servers_config
assert str(get_user_home("blake")) == "/tmp/test-users/blake", f"Got {get_user_home('blake')}"
assert "test-users" in str(get_memory_root("blake")), f"Got {get_memory_root('blake')}"
print("HERMES_USERS_ROOT override OK")

# Test HERMES_RUNTIME_ROOT override
os.environ["HERMES_RUNTIME_ROOT"] = "/tmp/test-runtime"
assert str(get_runtime_root()) == "/tmp/test-runtime", f"Got {get_runtime_root()}"
print("HERMES_RUNTIME_ROOT override OK")

# Clean up
del os.environ["HERMES_USERS_ROOT"]
del os.environ["HERMES_RUNTIME_ROOT"]
print("All env var override tests PASSED")
EOF

# 2. Test validator
python3 - <<'EOF'
import os, logging
logging.basicConfig(level=logging.WARNING)

from scripts.validate_env import validate_hermes_env

# Should warn (missing path), not raise
os.environ["HERMES_USERS_ROOT"] = "/tmp/does-not-exist-xyz"
validate_hermes_env()   # should log a warning, not raise
print("missing path: warned OK")

# Should error (unsupported URI) in non-strict mode
os.environ["HERMES_USERS_ROOT"] = "s3://my-bucket/hermes"
validate_hermes_env()   # should log error, not raise
print("s3 URI non-strict: logged error OK")

# Should raise in strict mode
try:
    validate_hermes_env(strict=True)
    print("FAIL: should have raised")
except ValueError:
    print("s3 URI strict: raised ValueError OK")

del os.environ["HERMES_USERS_ROOT"]
print("All validator tests PASSED")
EOF

# 3. Run new unit tests
pytest tests/test_constants.py -v

# 4. Full regression
pytest tests/ -v --tb=short 2>&1 | tail -20

# 5. Verify docs file exists
cat ~/Documents/hermes-agent/docs/env-vars.md | head -20
```

## Status
Complete — 2026-05-19

## Bug Log
| # | Description | Status |
|---|-------------|--------|
