"""lean-ctx context bootstrap provider.

This provider talks to lean-ctx through its MCP stdio interface so Hermes can
reuse the same ctx_* workflow that worker agents use. Lean-ctx owns ephemeral
context discovery here; configured Hermes memory and compression providers keep
their own responsibilities.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 12_000
_DEFAULT_DELEGATION_MAX_CHARS = 6_000
_DEFAULT_TIMEOUT_SECONDS = 8.0
_SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
    "TMPDIR",
}
_CODE_TASK_RE = re.compile(
    r"\b("
    r"code|repo|file|class|function|symbol|callers?|implementation|implement|"
    r"plugin|test|tests|pytest|fix|debug|review|refactor|build|compile|"
    r"trace|inspect|setup|config|configuration|AGENTS\.md|README|CLAUDE\.md"
    r")\b",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(
    r"(?:`([A-Za-z_][A-Za-z0-9_]{2,80})(?:\(\))?`|\b(?:class|def|function|method)\s+([A-Za-z_][A-Za-z0-9_]{2,80}))"
)


@dataclass(frozen=True)
class LeanCtxConfig:
    command: str = "lean-ctx"
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    packet_timeout_seconds: float = 25.0
    max_chars: int = _DEFAULT_MAX_CHARS
    delegation_max_chars: int = _DEFAULT_DELEGATION_MAX_CHARS
    max_task_chars: int = 4_000
    max_sessions: int = 1_024
    first_turn_only: bool = True
    code_task_only: bool = False
    include_overview: bool = True
    include_preload: bool = True
    include_handoff: bool = True
    include_symbols: bool = True
    include_callers: bool = True
    max_symbols: int = 3


class LeanCtxBootstrapProvider:
    name = "lean_ctx"

    def __init__(
        self,
        config: LeanCtxConfig,
        *,
        default_workspace_root: Path | None = None,
        call_tool: Callable[[str, dict[str, Any], Path, float], str] | None = None,
    ):
        self.config = config
        self.default_workspace_root = default_workspace_root
        self._call_tool = call_tool
        self._bootstrapped_sessions: dict[str, None] = {}

    def is_available(self) -> bool:
        if not shutil.which(self.config.command):
            return False
        try:
            import mcp  # noqa: F401
        except Exception:
            return False
        return True

    def context_for_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        is_first_turn: bool,
        workspace_root: Path,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        if self.config.first_turn_only and not is_first_turn:
            return ""
        if session_id in self._bootstrapped_sessions:
            return ""
        task = _clip_task(user_message, self.config.max_task_chars)
        if self.config.code_task_only and not _looks_like_code_task(task):
            return ""

        root = _resolve_root(workspace_root, self.default_workspace_root)
        context = self._build_packet(
            task=task,
            root=root,
            max_chars=self.config.max_chars,
            include_symbols=True,
            header="LEAN-CTX BOOTSTRAP CONTEXT",
        )
        if context:
            self._bootstrapped_sessions[session_id] = None
            if len(self._bootstrapped_sessions) > self.config.max_sessions:
                self._bootstrapped_sessions.pop(next(iter(self._bootstrapped_sessions)), None)
        return context

    def context_for_delegation(
        self,
        *,
        goal: str,
        context: str,
        workspace_root: Path,
    ) -> str:
        task = _clip_task(
            "\n\n".join(part for part in (goal, context) if part),
            self.config.max_task_chars,
        )
        if self.config.code_task_only and not _looks_like_code_task(task):
            return ""

        root = _resolve_root(workspace_root, self.default_workspace_root)
        return self._build_packet(
            task=task,
            root=root,
            max_chars=self.config.delegation_max_chars,
            include_symbols=True,
            header="LEAN-CTX DELEGATION CONTEXT",
        )

    def _build_packet(
        self,
        *,
        task: str,
        root: Path,
        max_chars: int,
        include_symbols: bool,
        header: str,
    ) -> str:
        calls: list[tuple[str, str, dict[str, Any]]] = []
        if self.config.include_overview:
            calls.append(("ctx_overview", "ctx_overview", {"path": str(root), "task": task}))
        if self.config.include_preload:
            calls.append(("ctx_preload", "ctx_preload", {"path": str(root), "task": task}))
        if self.config.include_handoff:
            calls.append(("ctx_handoff", "ctx_handoff", {"action": "list"}))

        if include_symbols and self.config.include_symbols:
            for symbol in _extract_symbols(task, self.config.max_symbols):
                calls.append((f"ctx_symbol:{symbol}", "ctx_symbol", {"name": symbol}))
                if self.config.include_callers:
                    calls.append((f"ctx_callers:{symbol}", "ctx_callers", {"symbol": symbol}))

        sections = self._safe_tools(calls, root)

        parts = [
            f"<{header.lower().replace(' ', '_')}>",
            header,
            "Use this as ephemeral lean-ctx context and verify files and symbols before acting.",
            f"workspace_root: {root}",
            f"task: {_clip(task, 700)}",
        ]
        budget_per_section = max(500, max_chars // max(1, len([s for s in sections if s[1]]) + 1))
        for name, content in sections:
            if not content:
                continue
            parts.append(f"\n## {name}\n{_clip(content, budget_per_section)}")
        parts.append(f"</{header.lower().replace(' ', '_')}>")
        return _clip("\n".join(parts), max_chars)

    def _safe_tools(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        root: Path,
    ) -> list[tuple[str, str]]:
        if not calls:
            return []
        try:
            if self._call_tool is not None:
                return [
                    (label, self._call_tool(tool_name, args, root, self.config.timeout_seconds))
                    for label, tool_name, args in calls
                ]
            return self._call_tools_via_mcp(calls, root, self.config.timeout_seconds)
        except Exception as exc:
            logger.debug("lean-ctx bootstrap failed: %s", type(exc).__name__)
            return []

    def _call_tools_via_mcp(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        root: Path,
        timeout_seconds: float,
    ) -> list[tuple[str, str]]:
        return _run_coro_sync(
            asyncio.wait_for(
                self._call_tools_via_mcp_async(calls, root, timeout_seconds),
                timeout=self.config.packet_timeout_seconds,
            ),
            timeout_seconds=self.config.packet_timeout_seconds + 1.0,
        )

    async def _call_tools_via_mcp_async(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        root: Path,
        timeout_seconds: float,
    ) -> list[tuple[str, str]]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=_build_env(self.config.env),
            cwd=str(root),
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout_seconds)
                results: list[tuple[str, str]] = []
                for label, tool_name, args in calls:
                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments=args),
                            timeout=timeout_seconds,
                        )
                    except Exception as exc:
                        logger.debug("lean-ctx bootstrap %s failed: %s", tool_name, type(exc).__name__)
                        results.append((label, ""))
                        continue
                    if getattr(result, "isError", False):
                        results.append((label, ""))
                    else:
                        results.append((label, _result_to_text(result)))
                return results


def create_provider(
    *,
    cfg: dict[str, Any],
    workspace_root: Path | None = None,
) -> LeanCtxBootstrapProvider:
    return LeanCtxBootstrapProvider(
        _load_config(cfg),
        default_workspace_root=workspace_root,
    )


def _load_config(cfg: dict[str, Any]) -> LeanCtxConfig:
    raw = _raw_lean_ctx_config(cfg)
    command = str(raw.get("command") or "lean-ctx")
    args = tuple(str(arg) for arg in (raw.get("args") or []))
    env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
    return LeanCtxConfig(
        command=command,
        args=args,
        env=env or None,
        timeout_seconds=float(raw.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
        packet_timeout_seconds=float(raw.get("packet_timeout_seconds", 25.0)),
        max_chars=int(raw.get("max_chars", _DEFAULT_MAX_CHARS)),
        delegation_max_chars=int(
            raw.get("delegation_max_chars", _DEFAULT_DELEGATION_MAX_CHARS)
        ),
        max_task_chars=int(raw.get("max_task_chars", 4_000)),
        max_sessions=int(raw.get("max_sessions", 1_024)),
        first_turn_only=bool(raw.get("first_turn_only", True)),
        code_task_only=bool(raw.get("code_task_only", False)),
        include_overview=bool(raw.get("include_overview", True)),
        include_preload=bool(raw.get("include_preload", True)),
        include_handoff=bool(raw.get("include_handoff", True)),
        include_symbols=bool(raw.get("include_symbols", True)),
        include_callers=bool(raw.get("include_callers", True)),
        max_symbols=int(raw.get("max_symbols", 3)),
    )


def _raw_lean_ctx_config(cfg: dict[str, Any]) -> dict[str, Any]:
    direct = cfg.get("lean_ctx")
    if isinstance(direct, dict):
        return direct
    bootstrap = cfg.get("context_bootstrap")
    if isinstance(bootstrap, dict) and isinstance(bootstrap.get("lean_ctx"), dict):
        return bootstrap["lean_ctx"]
    return {}


def _build_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
    if extra:
        env.update(
            {key: value for key, value in extra.items() if key.startswith("LEAN_CTX_")}
        )
    return env


def _result_to_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _run_coro_sync(coro, *, timeout_seconds: float | None = None):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise TimeoutError("lean-ctx MCP call timed out")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _looks_like_code_task(message: str) -> bool:
    if not message:
        return False
    if _CODE_TASK_RE.search(message):
        return True
    return bool(re.search(r"(^|\s)(\.?/|~/)?[\w.-]+/[\w./-]+", message))


def _extract_symbols(message: str, limit: int) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for match in _SYMBOL_RE.finditer(message or ""):
        symbol = next((group for group in match.groups() if group), "")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def _resolve_root(workspace_root: Path, fallback: Path | None) -> Path:
    root = workspace_root or fallback or Path.cwd()
    try:
        return root.expanduser().resolve()
    except Exception:
        return Path.cwd()


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 80)].rstrip() + "\n...[lean-ctx bootstrap truncated]"


def _clip_task(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 80)].rstrip() + "\n...[lean-ctx task truncated]"
