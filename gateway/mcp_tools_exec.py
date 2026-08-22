"""9 execution tools port 1:1 từ mcp-bridge (JS) — chạy NATIVE trong gateway.

Nguồn: mcp-bridge/tools/{terminal,filesystem,http,docker}.js +
utils/validator.js + middleware/ratelimit.js (semantics giữ nguyên).

3 lớp safety tái lập trong Python (bridge container đã gỡ — R1):
  1. ``validate_command`` — binary phải nằm whitelist (gateway.mcp.whitelist_commands).
  2. ``validate_path`` — path phải nằm trong allowed roots
     (gateway.mcp.allowed_paths, REMAP /shared/workspace container → host path).
  3. Rate-limit sliding-window 1h per-identity + timeout subprocess
     (gateway.mcp.rate_limit_per_hour / tool_timeout_ms).

Identity lấy từ headers MCP request (X-Hermes-User / X-Hermes-User-Id) qua
``ctx.request_context.request`` — Phase 3 sẽ resolve role qua Mongo (mcp_acl).
Audit mỗi execution gọi ``_audit_writer`` (Phase 3 gắn writer Mongo; trước đó
chỉ log).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import Context
except Exception:  # pragma: no cover
    Context = Any  # type: ignore[assignment,misc]

# --------------------------------------------------------------------------
# Config (gateway.mcp trong config.yaml — rule Hermes: phi-secret vào config).
# hermes config set lưu LIST dạng JSON-string (pitfall D3) → parse cả 2 dạng.
# --------------------------------------------------------------------------

_CONFIG_CACHE: dict = {}
_CONFIG_TS = 0.0

DEFAULT_ALLOWED_PATHS = [
    str(Path.home() / "dev" / "hermes-openwebui-stack" / "shared" / "workspace")
]
DEFAULT_WHITELIST = [
    "ls", "cat", "grep", "find", "df", "du", "ps", "top", "free",
    "netstat", "ss", "python3", "node", "npm", "pip", "git", "curl",
    "wget", "ping", "docker", "docker-compose",
]
DEFAULT_RATE_LIMIT = 100
DEFAULT_TIMEOUT_MS = 30000
_MAX_BUFFER = 10 * 1024 * 1024  # 10MB — như maxBuffer JS
_HTTP_TIMEOUT_S = 15
_HTTP_CAP = 10000


def _parse_list(value: Any) -> list:
    """List hoặc JSON-string (cách hermes config set lưu) → list."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                parsed = json.loads(value.replace("'", '"'))
            except Exception:
                return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(parsed, list):
            return list(parsed)
        return [parsed]
    return []


def _mcp_config() -> dict:
    """Block gateway.mcp với defaults; cache 60s."""
    global _CONFIG_CACHE, _CONFIG_TS
    import time

    now = time.monotonic()
    if _CONFIG_CACHE and now - _CONFIG_TS < 60:
        return _CONFIG_CACHE
    block: dict = {}
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        block = (cfg.get("gateway") or {}).get("mcp") or {}
    except Exception:
        logger.warning("mcp_tools_exec: không đọc được config gateway.mcp", exc_info=True)
    _CONFIG_CACHE = block
    _CONFIG_TS = now
    return block


def _allowed_paths() -> list[str]:
    paths = _parse_list(_mcp_config().get("allowed_paths"))
    return paths or DEFAULT_ALLOWED_PATHS


def _whitelist() -> list[str]:
    cmds = _parse_list(_mcp_config().get("whitelist_commands"))
    return cmds or DEFAULT_WHITELIST


def _rate_limit_per_hour() -> int:
    try:
        return int(_mcp_config().get("rate_limit_per_hour", DEFAULT_RATE_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT


def _tool_timeout_ms() -> int:
    try:
        return int(_mcp_config().get("tool_timeout_ms", DEFAULT_TIMEOUT_MS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_MS


# --------------------------------------------------------------------------
# Lớp safety 1: validateCommand (port validator.js)
# --------------------------------------------------------------------------

def validate_command(command: str) -> dict:
    """Binary đầu tiên (bỏ env-prefix/path) phải nằm whitelist."""
    if not isinstance(command, str) or not command.strip():
        return {"ok": False, "error": "command is required"}
    first = command.strip().split()[0]
    bin_name = os.path.basename(first)
    allowed = [c.lower() for c in _whitelist()]
    if bin_name.lower() not in allowed:
        return {"ok": False, "error": f"command not allowed: {bin_name}"}
    return {"ok": True, "bin": bin_name}


# --------------------------------------------------------------------------
# Lớp safety 2: validatePath (port validator.js)
# --------------------------------------------------------------------------

def _is_inside(root: str, target: str) -> bool:
    try:
        Path(target).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_path(target: str) -> dict:
    """Path phải nằm trong allowed roots (chặn traversal)."""
    if not isinstance(target, str) or not target:
        return {"ok": False, "error": "path is required"}
    abs_path = str(Path(target).resolve())
    for root in _allowed_paths():
        r = str(Path(root).resolve())
        if _is_inside(r, abs_path):
            return {"ok": True, "path": abs_path, "root": r}
    return {"ok": False, "error": f"path is outside allowed roots: {abs_path}"}


# --------------------------------------------------------------------------
# Identity + rate-limit (port ratelimit.js) + audit hook
# --------------------------------------------------------------------------

_RATE_WINDOWS: dict[str, deque] = {}
_RATE_LIMIT_HOUR_S = 3600.0

# Phase 3 gắn writer Mongo vào đây; mặc định chỉ log.
_audit_writer: Optional[Callable[[dict], Any]] = None


def set_audit_writer(writer: Optional[Callable[[dict], Any]]) -> None:
    """Gắn hàm ghi audit (Mongo) từ mcp_tools_admin (Phase 3)."""
    global _audit_writer
    _audit_writer = writer


def _get_identity(ctx: Any) -> str:
    """Identity = X-Hermes-User (fallback X-Hermes-User-Id, fallback unknown).

    Dùng cho rate-limit + audit trước Phase 3; Phase 3 resolve role qua Mongo.
    """
    username = None
    user_id = None
    try:
        req = getattr(getattr(ctx, "_request_context", None), "request", None)
        headers = getattr(req, "headers", None)
        if headers is not None:
            username = headers.get("x-hermes-user") or None
            user_id = headers.get("x-hermes-user-id") or None
    except Exception:
        pass
    return username or user_id or "unknown"


def _audit_event(event: dict) -> None:
    """Ghi audit — Mongo writer (P3) hoặc fallback log."""
    try:
        if _audit_writer is not None:
            _audit_writer(event)
            return
    except Exception:
        logger.warning("audit writer lỗi", exc_info=True)
    logger.info("mcp audit (no-mongo): %s", json.dumps(event, default=str)[:500])


def _rate_limit_check(identity: str) -> Optional[dict]:
    """Sliding window 1h per-identity. Vượt → dict lỗi (như 429 bridge)."""
    limit = _rate_limit_per_hour()
    now = asyncio.get_event_loop().time()
    w = _RATE_WINDOWS.get(identity)
    if w is None:
        w = _RATE_WINDOWS[identity] = deque()
    while w and now - w[0] > _RATE_LIMIT_HOUR_S:
        w.popleft()
    if len(w) >= limit:
        reset_in = int(_RATE_LIMIT_HOUR_S - (now - w[0]))
        return {
            "error": "Rate limit exceeded",
            "retryAfter": max(reset_in, 1),
            "limit": limit,
        }
    w.append(now)
    return None


def _rate_guard(ctx: Any, tool: str) -> Optional[dict]:
    identity = _get_identity(ctx)
    err = _rate_limit_check(identity)
    if err:
        _audit_event(
            {
                "type": "tool",
                "action": "ratelimited",
                "tool": tool,
                "identity": identity,
                "retryAfter": err.get("retryAfter"),
            }
        )
    return err


async def _run_subprocess(
    argv: list[str],
    timeout_ms: int,
    *,
    shell_command: Optional[str] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Chạy subprocess với timeout + cap output 10MB (port execFile)."""
    try:
        if shell_command is not None:
            proc = await asyncio.create_subprocess_shell(
                shell_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_ms / 1000.0
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": f"command timed out after {timeout_ms}ms"}
        stdout = (stdout_b or b"").decode("utf-8", "replace")[:_MAX_BUFFER]
        stderr = (stderr_b or b"").decode("utf-8", "replace")[:_MAX_BUFFER]
        if proc.returncode != 0:
            return {
                "error": f"exit code {proc.returncode}",
                "stdout": stdout,
                "stderr": stderr,
                "code": proc.returncode,
            }
        return {"stdout": stdout, "stderr": stderr, "code": 0}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------
# 9 tools — schema/tên giữ NGUYÊN từ tools/index.js
# --------------------------------------------------------------------------

def register_execution_tools(mcp: Any) -> None:
    """Đăng ký 9 execution tools vào FastMCP server."""
    timeout_ms = _tool_timeout_ms()
    ws_root = _allowed_paths()[0]

    @mcp.tool()
    async def execute_terminal(
        command: str, timeout: Optional[float] = None, ctx: Context = None
    ) -> dict:
        """Run a whitelisted shell command (3 lớp safety: whitelist binary,
        timeout, rate-limit)."""
        err = _rate_guard(ctx, "execute_terminal")
        if err:
            return err
        check = validate_command(command)
        if not check["ok"]:
            return {"error": check["error"]}
        _audit_event({"type": "tool", "action": "execute", "tool": "execute_terminal", "identity": _get_identity(ctx), "bin": check["bin"]})
        t_ms = int((timeout or timeout_ms / 1000.0) * 1000)
        t_ms = max(100, min(t_ms, 120000))
        return await _run_subprocess(
            [], t_ms, shell_command=command, cwd=ws_root
        )

    @mcp.tool()
    async def read_file(path: str, ctx: Context = None) -> dict:
        """Read a file inside the workspace."""
        err = _rate_guard(ctx, "read_file")
        if err:
            return err
        v = validate_path(path)
        if not v["ok"]:
            return {"error": v["error"]}
        _audit_event({"type": "tool", "action": "execute", "tool": "read_file", "identity": _get_identity(ctx)})
        try:
            data = await asyncio.to_thread(Path(v["path"]).read_text, encoding="utf-8")
            return {"content": data[:_MAX_BUFFER]}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def write_file(path: str, content: str, ctx: Context = None) -> dict:
        """Write a file inside the workspace."""
        err = _rate_guard(ctx, "write_file")
        if err:
            return err
        v = validate_path(path)
        if not v["ok"]:
            return {"error": v["error"]}
        _audit_event({"type": "tool", "action": "execute", "tool": "write_file", "identity": _get_identity(ctx)})
        try:
            p = Path(v["path"])
            await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(p.write_text, content, encoding="utf-8")
            return {"ok": True, "path": v["path"]}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def list_files(path: Optional[str] = None, ctx: Context = None) -> dict:
        """List a directory inside the workspace."""
        err = _rate_guard(ctx, "list_files")
        if err:
            return err
        target = path or _allowed_paths()[0]
        v = validate_path(target)
        if not v["ok"]:
            return {"error": v["error"]}
        _audit_event({"type": "tool", "action": "execute", "tool": "list_files", "identity": _get_identity(ctx)})
        try:
            entries = sorted(
                Path(v["path"]).iterdir(), key=lambda p: p.name.lower()
            )
            return {
                "path": v["path"],
                "entries": [
                    {"name": e.name, "type": "dir" if e.is_dir() else "file"}
                    for e in entries
                ],
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def search_files(
        path: Optional[str] = None, pattern: str = "", ctx: Context = None
    ) -> dict:
        """Search filenames inside the workspace (substring match)."""
        err = _rate_guard(ctx, "search_files")
        if err:
            return err
        if not pattern:
            return {"error": "pattern is required"}
        target = path or _allowed_paths()[0]
        v = validate_path(target)
        if not v["ok"]:
            return {"error": v["error"]}
        _audit_event({"type": "tool", "action": "execute", "tool": "search_files", "identity": _get_identity(ctx)})
        results = []

        def _walk(root: str) -> None:
            for dirpath, dirnames, filenames in os.walk(root):
                for name in filenames:
                    if pattern in name:
                        results.append(str(Path(dirpath) / name))

        try:
            await asyncio.to_thread(_walk, v["path"])
            return {"matches": results[:500]}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def http_request(
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        body: Any = None,
        timeout: Optional[float] = None,
        ctx: Context = None,
    ) -> dict:
        """Perform an HTTP(S) request (timeout 15s, response cap 10k)."""
        err = _rate_guard(ctx, "http_request")
        if err:
            return err
        import re

        if not url or not re.match(r"^https?://", url, re.I):
            return {"error": "url must be http(s)"}
        _audit_event({"type": "tool", "action": "execute", "tool": "http_request", "identity": _get_identity(ctx)})
        try:
            import aiohttp

            merged = {"Content-Type": "application/json"}
            merged.update(headers or {})
            t = float(timeout or _HTTP_TIMEOUT_S)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=t)
            ) as session:
                async with session.request(
                    method,
                    url,
                    headers=merged,
                    json=body if body is not None else None,
                ) as resp:
                    text = (await resp.text())[:_HTTP_CAP]
            try:
                data = json.loads(text)
            except Exception:
                data = text
            return {"status": resp.status, "ok": resp.ok, "data": data}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def docker_ps(ctx: Context = None) -> dict:
        """List running Docker containers (read-only)."""
        err = _rate_guard(ctx, "docker_ps")
        if err:
            return err
        _audit_event({"type": "tool", "action": "execute", "tool": "docker_ps", "identity": _get_identity(ctx)})
        r = await _run_subprocess(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            15000,
        )
        if "error" in r:
            return r
        return {
            "containers": [
                line for line in r["stdout"].strip().split("\n") if line
            ]
        }

    @mcp.tool()
    async def docker_logs(name: str, tail: Optional[int] = None, ctx: Context = None) -> dict:
        """Get container logs (read-only)."""
        err = _rate_guard(ctx, "docker_logs")
        if err:
            return err
        if not name:
            return {"error": "name is required"}
        _audit_event({"type": "tool", "action": "execute", "tool": "docker_logs", "identity": _get_identity(ctx)})
        r = await _run_subprocess(
            ["docker", "logs", "--tail", str(int(tail or 100)), name], 15000
        )
        if "error" in r:
            return r
        return {"logs": r["stdout"]}

    @mcp.tool()
    async def docker_exec(container: str, command: str, ctx: Context = None) -> dict:
        """Execute a command inside a container (whitelisted docker binary)."""
        err = _rate_guard(ctx, "docker_exec")
        if err:
            return err
        if not container or not command:
            return {"error": "container and command are required"}
        _audit_event({"type": "tool", "action": "execute", "tool": "docker_exec", "identity": _get_identity(ctx), "container": container})
        r = await _run_subprocess(
            ["docker", "exec", container] + command.split(), 15000
        )
        if "error" in r:
            return r
        return {"stdout": r["stdout"], "stderr": r["stderr"], "code": 0}
