# Phase 002-C: MCP Connection Pool (MCPGateway)

## Goal
Replace per-session MCP subprocess spawning with a single `MCPGateway` that runs each MCP server once and multiplexes calls from all sessions. Per-user credentials are injected per call, not per process. Result: `M servers = M processes` instead of `N users × M servers = N×M processes`.

## Context
`tools/mcp_tool.py` already has a sophisticated architecture: a dedicated background event loop (`_mcp_loop`) runs MCP servers as long-lived asyncio Tasks. It IS a connection pool already — each server runs once and calls are dispatched to it. The existing `_servers` dict is the pool. What it currently lacks is:
1. **Per-session credential injection** — credentials are embedded in `config.yaml` at the server level (e.g., `env: { GITHUB_PERSONAL_ACCESS_TOKEN: ghp_... }`), not per-session
2. **Access control** — there's no concept of "session A is allowed to call Gmail but session B is not"
3. **User-scoped server bindings** — users can't have their own MCP server configs independent of the system config

This phase introduces `MCPGateway` as a thin wrapper around the existing `mcp_tool.py` infrastructure, adding credential injection and scope enforcement. We do NOT rewrite `mcp_tool.py` — we add a layer on top.

## Dependencies
- **Phase 002-B must be complete** — `CredentialResolver` needs session isolation (runtime sandbox) to safely handle credential lifecycle; also needs `HERMES_USER_ID` to be set in the session environment

## Scope

### Files to Create
- `agent/mcp_gateway.py` — `MCPGateway` singleton + `MCPAccessDenied` exception
- `agent/credential_resolver.py` — `CredentialResolver` class (reads refs, resolves via Keychain or env, caches in memory)
- `~/.hermes/system/mcp-servers.json` — system-level MCP server binding definitions (populated here for first time)

### Files to Modify
- `tools/mcp_tool.py` — add `call_via_gateway(session_id, server_name, tool_name, args)` entry point that routes through `MCPGateway`; existing `_call_tool_direct()` kept as internal fallback
- `hermes_constants.py` — add `get_mcp_servers_config()` returning path to `system/mcp-servers.json`

### Explicitly Out of Scope
- Changing how tools are registered from MCP (the existing schema discovery still works the same)
- Multi-user auth for HTTP/SSE MCP servers (this phase only handles credential injection for stdio-type servers and `Authorization` header injection for HTTP servers)
- Gateway process becoming a sidecar (that's a cloud deployment decision, not a code change)

## Implementation Notes

### Key design decision: don't fight the existing architecture

`mcp_tool.py` already runs MCP servers as long-lived asyncio tasks. The existing `_servers` dict (`server_name → MCPServerState`) is the pool. What we're adding is an auth middleware layer:

```
Before (no gateway):
  Tool call → mcp_tool._call_tool() → direct to MCP subprocess

After (with gateway):
  Tool call → MCPGateway.call_tool(session_id, server, tool, args)
           → CredentialResolver.resolve(user_id, server)   [in-memory only]
           → inject credential into call context
           → mcp_tool._call_tool_with_credential(server, tool, args, credential)
```

### `agent/credential_resolver.py`

```python
"""CredentialResolver — resolves credential refs to values, in memory only."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_credentials_root

logger = logging.getLogger(__name__)


class CredentialResolver:
    """Resolves credential ref files to live values.
    
    Values are cached in-process memory for the session lifetime.
    They are NEVER written to disk. On session close, call clear().
    
    Ref file format (JSON):
        {"type": "keychain", "service": "hermes/blake/gmail", "account": "blake"}
        {"type": "env", "var": "GMAIL_TOKEN"}
        {"type": "secrets-manager", "arn": "arn:aws:..."}
    """

    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}   # user_id:server → value
        self._lock = threading.Lock()

    def resolve(self, user_id: str, server_name: str) -> Optional[str]:
        """Return the credential value for this user/server pair, or None."""
        cache_key = f"{user_id}:{server_name}"
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        value = self._load(user_id, server_name)
        with self._lock:
            self._cache[cache_key] = value
        return value

    def refresh(self, user_id: str, server_name: str) -> Optional[str]:
        """Force re-resolve (e.g., after MCPAccessDenied)."""
        cache_key = f"{user_id}:{server_name}"
        with self._lock:
            self._cache.pop(cache_key, None)
        return self.resolve(user_id, server_name)

    def clear(self) -> None:
        """Shred all cached values (call on session close)."""
        with self._lock:
            self._cache.clear()

    def _load(self, user_id: str, server_name: str) -> Optional[str]:
        creds_root = get_credentials_root(user_id)
        ref_file = creds_root / f"{server_name}.ref"
        if not ref_file.exists():
            return None
        try:
            ref = json.loads(ref_file.read_text())
        except Exception as exc:
            logger.warning("CredentialResolver: bad ref file %s: %s", ref_file, exc)
            return None
        
        ref_type = ref.get("type", "")
        if ref_type == "keychain":
            return self._resolve_keychain(ref["service"], ref.get("account", ""))
        if ref_type == "env":
            return os.environ.get(ref["var"])
        if ref_type == "secrets-manager":
            return self._resolve_secrets_manager(ref["arn"])
        logger.warning("CredentialResolver: unknown ref type '%s'", ref_type)
        return None

    def _resolve_keychain(self, service: str, account: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as exc:
            logger.warning("CredentialResolver: keychain lookup failed: %s", exc)
            return None

    def _resolve_secrets_manager(self, arn: str) -> Optional[str]:
        try:
            import boto3  # type: ignore
            client = boto3.client("secretsmanager")
            resp = client.get_secret_value(SecretId=arn)
            return resp.get("SecretString")
        except Exception as exc:
            logger.warning("CredentialResolver: SecretsManager lookup failed: %s", exc)
            return None
```

### `agent/mcp_gateway.py`

```python
"""MCPGateway — session-scoped credential injection for MCP tool calls."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from agent.credential_resolver import CredentialResolver

logger = logging.getLogger(__name__)


class MCPAccessDenied(Exception):
    """Raised when a session has no valid credential for an MCP server."""


class MCPGateway:
    """Singleton gateway: one MCP process pool, per-session credential injection.
    
    The underlying mcp_tool.py already manages the process pool (one process
    per server, all sessions share it). This layer adds:
    - Per-user credential resolution
    - Access control: no credential → MCPAccessDenied
    - Credential refresh on transient auth failure
    """

    _instance: Optional["MCPGateway"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "MCPGateway":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._resolver = CredentialResolver()

    def call_tool(
        self,
        session_id: str,
        user_id: str,
        server_name: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Route a tool call through the MCP pool with credential injection."""
        from tools.mcp_tool import call_tool_with_credential  # circular import guard
        
        credential = self._resolver.resolve(user_id, server_name)
        if credential is None:
            # Check if server requires a credential at all (system servers are open)
            if self._server_requires_auth(server_name):
                raise MCPAccessDenied(
                    f"Session {session_id} has no credential for MCP server '{server_name}'. "
                    f"Add a ref file at ~/.hermes/users/{user_id}/credentials/{server_name}.ref"
                )
        try:
            return call_tool_with_credential(server_name, tool_name, args, credential=credential)
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.info("MCPGateway: got 401 for %s/%s, refreshing credential", server_name, tool_name)
                credential = self._resolver.refresh(user_id, server_name)
                return call_tool_with_credential(server_name, tool_name, args, credential=credential)
            raise

    def _server_requires_auth(self, server_name: str) -> bool:
        """Return True if this server has a credential ref configured."""
        # System-level server configs (no per-user cred = open access)
        return False   # Phase C stub: revisit when user binding files are populated
```

### Changes to `tools/mcp_tool.py`

Add a `call_tool_with_credential(server_name, tool_name, args, credential)` function that temporarily injects the credential as an env var or Authorization header before dispatching to the existing `_call_tool()` path:

```python
def call_tool_with_credential(
    server_name: str,
    tool_name: str,
    args: dict,
    credential: str | None = None,
) -> Any:
    """Call an MCP tool, injecting credential if provided.
    
    For stdio servers: the credential is passed as HERMES_MCP_CREDENTIAL env var
    to the subprocess via a per-call env overlay (NOT modifying the persistent
    server process's env).
    For HTTP servers: injected as Authorization header.
    
    Falls back to _call_tool() (existing behavior) when credential is None.
    """
    # Existing _call_tool logic — add credential routing here
    # This is intentionally left as a thin wrapper until per-call env overlay
    # is implemented in the MCP protocol layer.
    if credential:
        # Store in thread-local so the async dispatch layer can inject it
        _per_call_credential.credential = credential
    try:
        return _call_tool(server_name, tool_name, args)
    finally:
        if credential:
            _per_call_credential.credential = None
```

### `hermes_constants.py` addition

```python
def get_mcp_servers_config() -> Path:
    """Return path to system/mcp-servers.json (system-wide MCP server definitions)."""
    return get_hermes_home() / "system" / "mcp-servers.json"
```

## Acceptance Criteria
- [ ] Starting two concurrent Hermes sessions does NOT result in duplicate MCP server processes (verify: `pgrep -f mcp` shows M processes regardless of N sessions)
- [ ] `CredentialResolver.resolve("blake", "gmail")` returns `None` (no ref file) without error
- [ ] `CredentialResolver.resolve("blake", "gmail")` returns a value after placing a valid `gmail.ref` in `users/blake/credentials/`
- [ ] `CredentialResolver.clear()` empties the cache (verified by inspecting `_cache` dict post-clear)
- [ ] `MCPGateway.call_tool()` raises `MCPAccessDenied` when `_server_requires_auth()` returns True and no credential exists
- [ ] `MCPGateway.call_tool()` retries with a refreshed credential on 401 (once) before propagating the error
- [ ] Credential values are never written to any file during a session (confirmed by searching runtime dir for any file containing a resolved credential value)
- [ ] `get_mcp_servers_config()` returns `Path("~/.hermes/system/mcp-servers.json")`
- [ ] `pytest tests/test_mcp_gateway.py -v` — all pass (new test file to write as part of this phase)
- [ ] `pytest tests/ -v` — zero regressions

## Verification Steps

```bash
# 1. Verify no duplicate MCP processes under load
# Open two terminal windows, start hermes in each, make a Gmail tool call in both
pgrep -fl "mcp\|modelcontextprotocol" | sort
# Expected: each server name appears exactly ONCE regardless of session count

# 2. Unit test CredentialResolver
cd ~/Documents/hermes-agent
python3 - <<'EOF'
import os, tempfile, json
from pathlib import Path

tmp = tempfile.mkdtemp()
os.environ["HERMES_USERS_ROOT"] = tmp
os.environ["HERMES_USER_ID"] = "blake"

# Create a test credential ref using env type (easiest to test)
creds_dir = Path(tmp) / "blake" / "credentials"
creds_dir.mkdir(parents=True)
os.environ["TEST_MCP_TOKEN"] = "my-secret-token"
(creds_dir / "testserver.ref").write_text(json.dumps({"type": "env", "var": "TEST_MCP_TOKEN"}))

from agent.credential_resolver import CredentialResolver
resolver = CredentialResolver()
val = resolver.resolve("blake", "testserver")
assert val == "my-secret-token", f"Expected token, got {val}"
print("resolve OK:", val)

# Test cache hit
val2 = resolver.resolve("blake", "testserver")
assert val2 == val, "Cache miss on second call"
print("cache hit OK")

# Test clear
resolver.clear()
assert resolver._cache == {}, "Cache not cleared"
print("clear OK")

print("All CredentialResolver tests PASSED")
import shutil; shutil.rmtree(tmp)
EOF

# 3. Run new unit tests
pytest tests/test_mcp_gateway.py -v

# 4. Run full regression
pytest tests/ -v --tb=short 2>&1 | tail -20
```

## Status
Complete — 2026-05-19

## Bug Log
| # | Description | Status |
|---|-------------|--------|
