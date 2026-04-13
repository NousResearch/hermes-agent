from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


class ObsidianMemoryProvider(MemoryProvider):
    name = "obsidian"

    def __init__(self) -> None:
        self._active = False
        self._vault_path: Optional[Path] = None
        self._session_id = ""
        self._platform = ""
        self._turn_count = 0
        self._tool_calls_total = 0
        self._tool_calls_since_checkpoint = 0
        self._checkpoint_interval = 4
        self._max_chars_per_note = 2500
        self._prefetch_on_next_turn = True
        self._session_started_logged = False
        self._last_turn_message = ""
        self._agent_home: Optional[Path] = None
        self._shared_home: Optional[Path] = None
        self._daily_home: Optional[Path] = None
        self._private_home: Optional[Path] = None

    def is_available(self) -> bool:
        return self._resolve_vault_path() is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._platform = str(kwargs.get("platform") or "")
        cfg = _load_config()
        mem_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
        self._checkpoint_interval = max(3, min(5, int(mem_cfg.get("obsidian_checkpoint_tool_calls", 4) or 4)))
        self._max_chars_per_note = max(800, int(mem_cfg.get("obsidian_read_char_limit", 2500) or 2500))
        self._vault_path = self._resolve_vault_path()
        self._active = self._vault_path is not None
        if not self._active:
            return
        self._shared_home = self._vault_path / "Agent-Shared"
        self._private_home = self._vault_path / "Agent-Hermes"
        self._daily_home = self._private_home / "daily"
        self._agent_home = self._private_home
        self._ensure_layout()

    def system_prompt_block(self) -> str:
        if not self._active or not self._vault_path:
            return ""
        today_rel = self._relative(self._today_note())
        return "\n".join([
            "# Obsidian vault memory layer",
            f"Layer 3 lives in the shared Obsidian vault at {self._vault_path}.",
            "Read these notes at session start and again after context compression or when you need more detail:",
            f"- {self._relative(self._shared('user-profile.md'))}",
            f"- {self._relative(self._shared('project-state.md'))}",
            f"- {self._relative(self._private('working-context.md'))}",
            f"- {today_rel}",
            f"- {self._relative(self._shared('decisions-log.md'))} when decisions/history matter",
            "Write to the vault on task start, every 3-5 tool calls, after corrections, on task completion, after compaction, and at session end.",
            "Never write inside Agent-Aria/. Use Agent-Shared/ for shared state and Agent-Hermes/ for Hermes-private state.",
        ])

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active:
            return ""
        if not self._prefetch_on_next_turn and not self._query_needs_refresh(query):
            return ""
        self._prefetch_on_next_turn = False
        parts = []
        for title, path in [
            ("Shared user profile", self._shared("user-profile.md")),
            ("Shared project state", self._shared("project-state.md")),
            ("Hermes working context", self._private("working-context.md")),
            ("Today's daily log", self._today_note()),
            ("Shared decisions", self._shared("decisions-log.md")),
        ]:
            text = self._safe_read(path)
            if text:
                parts.append(f"## {title} ({self._relative(path)})\n{text}")
        if not parts:
            return ""
        return "# Obsidian vault context\n" + "\n\n".join(parts)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if not self._active:
            return
        self._turn_count = max(turn_number, 0)
        self._last_turn_message = (message or "").strip()
        if not self._session_started_logged:
            summary = self._shorten(self._last_turn_message or "Session started", 220)
            self._append_daily("## Hermes session start", [
                f"- session: `{self._session_id or 'unknown'}`",
                f"- platform: {self._platform or 'unknown'}",
                f"- first task: {summary}",
            ])
            self._append_working_context("session-start", [
                f"Session {self._session_id or 'unknown'} started on {self._platform or 'unknown'}.",
                f"Initial request: {summary}",
            ])
            self._session_started_logged = True

    def on_tool_call_complete(self, tool_name: str, args: Dict[str, Any], result: str, **kwargs) -> None:
        if not self._active:
            return
        self._tool_calls_total += 1
        self._tool_calls_since_checkpoint += 1
        if self._tool_calls_since_checkpoint < self._checkpoint_interval:
            return
        self._tool_calls_since_checkpoint = 0
        preview = self._tool_preview(tool_name, args, result)
        self._append_daily("## Hermes checkpoint", [
            f"- turn: {self._turn_count}",
            f"- tools total: {self._tool_calls_total}",
            f"- latest tool: {preview}",
        ])
        self._append_working_context("checkpoint", [
            f"Turn {self._turn_count}, tool {self._tool_calls_total}: {preview}",
            f"Current request: {self._shorten(self._last_turn_message, 220)}",
        ])

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._active:
            return ""
        self._prefetch_on_next_turn = True
        recent = self._conversation_summary(messages)
        self._append_daily("## Hermes compaction", [
            "- Context compression fired.",
            f"- Snapshot: {recent}",
        ])
        self._append_working_context("compaction", [
            "Context compression fired; re-read vault context next turn.",
            recent,
        ])
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active:
            return
        recent = self._conversation_summary(messages)
        self._append_daily("## Hermes session end", [
            f"- session: `{self._session_id or 'unknown'}`",
            f"- turns observed: {self._turn_count}",
            f"- tool calls observed: {self._tool_calls_total}",
            f"- summary: {recent}",
        ])
        self._append_working_context("session-end", [
            f"Session {self._session_id or 'unknown'} ended.",
            recent,
        ])

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._active or action not in {"add", "replace"}:
            return
        cleaned = self._shorten((content or "").strip(), 240)
        if not cleaned:
            return
        if target == "user":
            self._append_shared_profile([f"- Explicit Hermes user memory ({action}): {cleaned}"])
        else:
            self._append_working_context("built-in-memory", [f"Hermes built-in memory {action}: {cleaned}"])

    def shutdown(self) -> None:
        return

    def _resolve_vault_path(self) -> Optional[Path]:
        env_path = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
        if env_path:
            p = Path(env_path).expanduser()
            if p.exists():
                return p
        fallback = Path.home() / "Documents" / "Obsidian Vault"
        if fallback.exists():
            return fallback
        config_path = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                vaults = data.get("vaults") or {}
                for item in vaults.values():
                    path = str((item or {}).get("path") or "").strip()
                    if path:
                        p = Path(path).expanduser()
                        if p.exists():
                            return p
            except Exception:
                logger.debug("Failed to parse Obsidian config", exc_info=True)
        return None

    def _ensure_layout(self) -> None:
        for folder in [self._shared_home, self._private_home, self._daily_home]:
            if folder:
                folder.mkdir(parents=True, exist_ok=True)
        self._ensure_file(self._shared("user-profile.md"), "# Shared user profile\n\n")
        self._ensure_file(self._shared("project-state.md"), "# Shared project state\n\n")
        self._ensure_file(self._shared("decisions-log.md"), "# Shared decisions log\n\n")
        self._ensure_file(self._private("working-context.md"), "# Hermes working context\n\n## Activity log\n")
        self._ensure_file(self._private("mistakes.md"), "# Hermes mistakes\n\n")
        self._ensure_file(self._today_note(), f"# Hermes daily log - {self._today_str()}\n\n")

    def _ensure_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content)

    def _shared(self, name: str) -> Path:
        return (self._shared_home or Path(".")) / name

    def _private(self, name: str) -> Path:
        return (self._private_home or Path(".")) / name

    def _today_note(self) -> Path:
        return (self._daily_home or Path(".")) / f"{self._today_str()}.md"

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _relative(self, path: Path) -> str:
        try:
            if self._vault_path:
                return str(path.relative_to(self._vault_path))
        except Exception:
            pass
        return str(path)

    def _safe_read(self, path: Path) -> str:
        try:
            if not path.exists():
                return ""
            text = path.read_text().strip()
            if len(text) > self._max_chars_per_note:
                return text[: self._max_chars_per_note].rstrip() + "\n...[truncated]"
            return text
        except Exception:
            logger.debug("Failed reading vault note %s", path, exc_info=True)
            return ""

    def _append_daily(self, heading: str, lines: List[str]) -> None:
        self._append_block(self._today_note(), heading, lines)

    def _append_working_context(self, label: str, lines: List[str]) -> None:
        heading = f"## {datetime.now().strftime('%H:%M')} - {label}"
        self._append_block(self._private("working-context.md"), heading, lines)

    def _append_shared_profile(self, lines: List[str]) -> None:
        self._append_block(self._shared("user-profile.md"), "## Hermes additions", lines)

    def _append_block(self, path: Path, heading: str, lines: List[str]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            block = ["", heading, f"- logged: {timestamp}"]
            block.extend(line if line.startswith("- ") else f"- {line}" for line in lines if line)
            with path.open("a") as f:
                f.write("\n".join(block) + "\n")
        except Exception:
            logger.debug("Failed writing vault note %s", path, exc_info=True)

    def _shorten(self, text: str, limit: int) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3].rstrip() + "..."

    def _tool_preview(self, tool_name: str, args: Dict[str, Any], result: str) -> str:
        arg_keys = ", ".join(sorted((args or {}).keys())[:6])
        summary = self._shorten(result or "", 140)
        return f"{tool_name}({arg_keys}) -> {summary}"

    def _conversation_summary(self, messages: List[Dict[str, Any]]) -> str:
        pairs = []
        for msg in messages[-12:]:
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = self._shorten(str(msg.get("content") or ""), 120)
            if content:
                pairs.append(f"{role}: {content}")
        if not pairs:
            return "No recent user/assistant messages captured."
        return " | ".join(pairs[-4:])

    def _query_needs_refresh(self, query: str) -> bool:
        q = (query or "").lower()
        return any(token in q for token in [
            "obsidian", "vault", "memory", "profile", "project", "decision",
            "context", "last time", "working on", "working context", "daily log",
        ])


def register(ctx) -> None:
    ctx.register_memory_provider(ObsidianMemoryProvider())
