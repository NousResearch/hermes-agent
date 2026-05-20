# HERMES Environment Variables

Canonical reference for all `HERMES_*` environment variables.
Last updated: Plan 002-D (2026-05-19).

---

## Operator Variables

Set these **before starting Hermes** to customize the storage layout.
Hermes path functions read them at call time (not at import time) so
changing them between sessions works correctly.

| Variable | Description | Default |
|----------|-------------|---------|
| `HERMES_HOME` | Root of the Hermes state directory | `~/.hermes/` |
| `HERMES_USERS_ROOT` | Parent of all user namespace directories | `$HERMES_HOME/users/` |
| `HERMES_RUNTIME_ROOT` | Ephemeral session sandbox root (prefer local fast storage) | `$HERMES_HOME/runtime/` |
| `HERMES_MCP_SERVERS_CONFIG` | Absolute path to the system MCP server definitions JSON | `$HERMES_HOME/system/mcp-servers.json` |

### Notes

- **`HERMES_USERS_ROOT`** affects all user-scoped paths automatically:
  `users/{id}/memory/`, `users/{id}/skills/`, `users/{id}/plans/`,
  `users/{id}/artifacts/`, `users/{id}/credentials/`.
  You only need to set this one variable to redirect the entire user namespace.

- **`HERMES_RUNTIME_ROOT`** should always be on local, low-latency storage.
  Do not point it at a network mount — session sandboxes are created and
  destroyed per session and latency matters.

- **`HERMES_MCP_SERVERS_CONFIG`** must be an absolute path to a JSON file.
  URI schemes are not supported (no S3 backend yet).

---

## Runtime Variables (set by Hermes itself)

These are set by Hermes at runtime. **Do not set them manually.**

| Variable | Set by | Purpose |
|----------|--------|---------|
| `HERMES_HOME` | CLI startup / `hermes_constants.get_hermes_home()` | Profile resolution |
| `HERMES_SESSION_ID` | `AIAgent.__init__` | Session identity for tools |
| `HERMES_USER_ID` | `AIAgent.__init__` (or operator for testing) | Routes paths to the correct user namespace |

---

## Validation

Run the validator at startup to catch misconfiguration early:

```bash
python3 scripts/validate_env.py
```

Or from code:

```python
from scripts.validate_env import validate_hermes_env
validate_hermes_env()          # non-strict: log errors, continue
validate_hermes_env(strict=True)  # strict: raise ValueError on errors
```

The validator:
- **Warns** if a path doesn't exist yet (lazy creation is normal)
- **Errors** if a URI scheme is used that has no backend abstraction (e.g. `s3://`)

---

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

### Multi-profile local setup

Use `HERMES_HOME` to point at a profile directory:

```bash
export HERMES_HOME=~/.hermes/profiles/work
```

All other paths derive from `HERMES_HOME` by default.

---

## Path Function Reference

| Function | Returns | Env var override |
|----------|---------|-----------------|
| `get_hermes_home()` | `~/.hermes/` | `HERMES_HOME` |
| `get_users_root()` | `$HERMES_HOME/users/` | `HERMES_USERS_ROOT` |
| `get_user_home(uid)` | `get_users_root()/{uid}/` | (inherits `HERMES_USERS_ROOT`) |
| `get_memory_root(uid)` | `get_user_home(uid)/memory/` | (inherits) |
| `get_plans_root(uid)` | `get_user_home(uid)/plans/` | (inherits) |
| `get_skills_root(uid)` | `get_user_home(uid)/skills/` | (inherits) |
| `get_artifacts_root(uid)` | `get_user_home(uid)/artifacts/` | (inherits) |
| `get_credentials_root(uid)` | `get_user_home(uid)/credentials/` | (inherits) |
| `get_runtime_root()` | `$HERMES_HOME/runtime/` | `HERMES_RUNTIME_ROOT` |
| `get_mcp_servers_config()` | `$HERMES_HOME/system/mcp-servers.json` | `HERMES_MCP_SERVERS_CONFIG` |
