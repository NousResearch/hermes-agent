"""Obsidian memory plugin — disciplined external memory layer for Hermes.

This provider complements the built-in MEMORY.md / USER.md stores with a small,
structured Obsidian workspace intended for high-signal continuity:
- user-profile.md    — mirrored from USER.md
- active-projects.md — manually curated / low-frequency project status note
- decisions-log.md   — mirrored from MEMORY.md
- current-focus.md   — latest snapshot / handoff note

Design goals:
- Hermes owns the vault; Claude consumes injected summaries rather than
  free-writing arbitrary notes.
- Writes are disciplined: mirror built-in curated memory, plus a bounded
  session-end focus snapshot.
- Reads are selective: compact session-start bootstrap and lightweight
  query-based snippet recall.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore, get_memory_dir

logger = logging.getLogger(__name__)

_DEFAULT_VAULT = Path.home() / "Documents" / "Obsidian Vault"
_DEFAULT_WORKSPACE = "Hermes"
_MAX_NOTE_SNIPPET = 700
_MAX_PREFETCH_NOTES = 3
_MAX_RECENT_USER_MESSAGES = 3
_MAX_RECENT_ASSISTANT_MESSAGES = 2

_QUERY_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _truncate(text: str, limit: int = _MAX_NOTE_SNIPPET) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class ObsidianPaths:
    vault_root: Path
    workspace_root: Path
    user_profile: Path
    active_projects: Path
    decisions_log: Path
    current_focus: Path


class ObsidianMemoryProvider(MemoryProvider):
    def __init__(self):
        self._hermes_home = get_hermes_home()
        self._paths: Optional[ObsidianPaths] = None
        self._session_id = ""
        self._platform = "cli"
        self._current_workspace = ""
        self._initialized = False
        self._session_write_count = 0
        self._focus_dirty = False
        self._last_prefetch = ""
        self._first_prefetch_done = False
        self._recent_user_messages: list[str] = []
        self._recent_assistant_messages: list[str] = []

    @property
    def name(self) -> str:
        return "obsidian"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "vault_path",
                "description": "Obsidian vault path",
                "default": os.environ.get("OBSIDIAN_VAULT_PATH", str(_DEFAULT_VAULT)),
            },
            {
                "key": "workspace",
                "description": "Folder inside the vault for Hermes-managed notes",
                "default": _DEFAULT_WORKSPACE,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli") or "cli"
        self._hermes_home = Path(kwargs.get("hermes_home") or get_hermes_home())
        self._current_workspace = os.environ.get("TERMINAL_CWD") or os.getcwd()

        cfg = self._load_provider_config()
        vault_path = Path(
            os.environ.get("OBSIDIAN_VAULT_PATH")
            or cfg.get("vault_path")
            or _DEFAULT_VAULT
        ).expanduser()
        workspace = str(cfg.get("workspace") or _DEFAULT_WORKSPACE).strip() or _DEFAULT_WORKSPACE

        workspace_root = vault_path / workspace
        self._paths = ObsidianPaths(
            vault_root=vault_path,
            workspace_root=workspace_root,
            user_profile=workspace_root / "user-profile.md",
            active_projects=workspace_root / "active-projects.md",
            decisions_log=workspace_root / "decisions-log.md",
            current_focus=workspace_root / "current-focus.md",
        )
        self._ensure_workspace()
        self._sync_structured_notes_from_builtin_memory()
        self._initialized = True

    def system_prompt_block(self) -> str:
        if not self._paths:
            return ""
        return (
            "External memory provider active: Obsidian structured workspace. "
            "Hermes owns note updates. Use built-in memory for durable user/system facts; "
            "Claude should consume the recalled summaries instead of inventing independent note writes. "
            f"Workspace: {self._paths.workspace_root}"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._paths or not self._initialized:
            return ""

        notes = self._load_note_map()
        if not notes:
            return ""

        selected: list[tuple[str, str]] = []
        if not self._first_prefetch_done:
            self._first_prefetch_done = True
            selected = [
                ("current-focus", notes.get("current-focus", "")),
                ("active-projects", notes.get("active-projects", "")),
                ("user-profile", notes.get("user-profile", "")),
            ]
        else:
            tokens = {tok.lower() for tok in _QUERY_TOKEN_RE.findall(query or "")}
            scored: list[tuple[int, str, str]] = []
            for name, content in notes.items():
                if not content.strip():
                    continue
                text = content.lower()
                score = 1 if name == "current-focus" else 0
                for token in tokens:
                    score += text.count(token)
                if score > 0:
                    scored.append((score, name, content))
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = [(name, content) for _, name, content in scored[:_MAX_PREFETCH_NOTES]]

        selected = [(name, content) for name, content in selected if content.strip()]
        if not selected:
            return ""

        parts = ["## Obsidian external memory"]
        for name, content in selected:
            parts.append(f"### {name}\n{_truncate(content)}")
        self._last_prefetch = "\n\n".join(parts)
        return self._last_prefetch

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._paths or not self._initialized:
            return
        user_text = self._normalize_message_text(user_content)
        assistant_text = self._normalize_message_text(assistant_content)
        if user_text:
            self._recent_user_messages.append(user_text)
            self._recent_user_messages = self._recent_user_messages[-_MAX_RECENT_USER_MESSAGES:]
        if assistant_text:
            self._recent_assistant_messages.append(assistant_text)
            self._recent_assistant_messages = self._recent_assistant_messages[-_MAX_RECENT_ASSISTANT_MESSAGES:]
        # Mark focus as dirty when the interaction seems meaningful.
        if user_text or assistant_text:
            self._focus_dirty = True

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._paths or action not in {"add", "replace"}:
            return
        self._session_write_count += 1
        self._focus_dirty = True
        self._sync_structured_notes_from_builtin_memory()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._paths:
            return
        if self._focus_dirty:
            self._write_current_focus(messages)
            self._focus_dirty = False

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError("Obsidian provider does not expose tools")

    def _load_provider_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception:
            return {}
        memory_cfg = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        provider_cfg = memory_cfg.get("obsidian", {}) if isinstance(memory_cfg, dict) else {}
        return provider_cfg if isinstance(provider_cfg, dict) else {}

    def _ensure_workspace(self) -> None:
        assert self._paths is not None
        self._paths.workspace_root.mkdir(parents=True, exist_ok=True)
        if not self._paths.user_profile.exists():
            self._write_note(
                self._paths.user_profile,
                self._render_note(
                    title="User Profile",
                    purpose="Durable user preferences and personal operating style mirrored from USER.md.",
                    body="## Durable facts\n\n- No captured user-profile facts yet.\n",
                ),
            )
        if not self._paths.active_projects.exists():
            self._write_note(
                self._paths.active_projects,
                self._render_note(
                    title="Active Projects",
                    purpose="High-signal project statuses and cross-project context. Keep this concise and manually curated.",
                    body=(
                        "## Projects\n\n"
                        "- Add or refine active project snapshots here when architecture or roadmap state materially changes.\n"
                    ),
                ),
            )
        if not self._paths.decisions_log.exists():
            self._write_note(
                self._paths.decisions_log,
                self._render_note(
                    title="Decisions Log",
                    purpose="Durable system, project, and environment context mirrored from MEMORY.md.",
                    body="## Durable system and project notes\n\n- No mirrored system/project notes yet.\n",
                ),
            )
        if not self._paths.current_focus.exists():
            self._write_note(
                self._paths.current_focus,
                self._render_note(
                    title="Current Focus",
                    purpose="Latest bounded snapshot of what Hermes was working on. Overwritten, not appended.",
                    body=(
                        "## Snapshot\n\n"
                        f"- Last updated: {_utc_now_iso()}\n"
                        "- No session snapshot captured yet.\n"
                    ),
                ),
            )

    def _render_note(self, *, title: str, purpose: str, body: str) -> str:
        return (
            "---\n"
            f"title: {title}\n"
            f"updated: {_utc_now_iso()}\n"
            f"managed_by: hermes-obsidian-provider\n"
            "---\n\n"
            f"# {title}\n\n"
            f"> {purpose}\n\n"
            f"{body.rstrip()}\n"
        )

    def _write_note(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _load_note_map(self) -> Dict[str, str]:
        assert self._paths is not None
        note_paths = {
            "user-profile": self._paths.user_profile,
            "active-projects": self._paths.active_projects,
            "decisions-log": self._paths.decisions_log,
            "current-focus": self._paths.current_focus,
        }
        results: Dict[str, str] = {}
        for name, path in note_paths.items():
            try:
                results[name] = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                results[name] = ""
        return results

    def _normalize_message_text(self, content: Any) -> str:
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    pieces.append(str(item.get("text", "")))
            content = " ".join(pieces)
        if not isinstance(content, str):
            return ""
        return _truncate(content, 220)

    def _sync_structured_notes_from_builtin_memory(self) -> None:
        assert self._paths is not None
        store = MemoryStore()
        store.load_from_disk()

        user_entries = store.user_entries
        memory_entries = store.memory_entries

        user_body = "## Durable facts\n\n"
        if user_entries:
            user_body += "\n".join(f"- {entry}" for entry in user_entries) + "\n"
        else:
            user_body += "- No captured user-profile facts yet.\n"
        self._write_note(
            self._paths.user_profile,
            self._render_note(
                title="User Profile",
                purpose="Durable user preferences and personal operating style mirrored from USER.md.",
                body=user_body,
            ),
        )

        memory_body = "## Durable system and project notes\n\n"
        if memory_entries:
            memory_body += "\n".join(f"- {entry}" for entry in memory_entries) + "\n"
        else:
            memory_body += "- No mirrored system/project notes yet.\n"
        self._write_note(
            self._paths.decisions_log,
            self._render_note(
                title="Decisions Log",
                purpose="Durable system, project, and environment context mirrored from MEMORY.md.",
                body=memory_body,
            ),
        )

    def _write_current_focus(self, messages: List[Dict[str, Any]]) -> None:
        assert self._paths is not None
        recent_user = list(self._recent_user_messages)
        recent_assistant = list(self._recent_assistant_messages)

        if not recent_user or not recent_assistant:
            for message in messages[-12:]:
                role = message.get("role")
                text = self._normalize_message_text(message.get("content"))
                if not text:
                    continue
                if role == "user":
                    recent_user.append(text)
                elif role == "assistant":
                    recent_assistant.append(text)

        user_lines = recent_user[-_MAX_RECENT_USER_MESSAGES:] or ["No recent user messages captured."]
        assistant_lines = recent_assistant[-_MAX_RECENT_ASSISTANT_MESSAGES:] or ["No recent assistant output captured."]
        body = (
            "## Snapshot\n\n"
            f"- Last updated: {_utc_now_iso()}\n"
            f"- Session ID: {self._session_id or 'unknown'}\n"
            f"- Platform: {self._platform}\n"
            f"- Workspace: {self._current_workspace}\n"
            f"- Built-in memory writes this session: {self._session_write_count}\n\n"
            "## Recent user focus\n\n"
            + "\n".join(f"- {line}" for line in user_lines)
            + "\n\n## Latest assistant direction\n\n"
            + "\n".join(f"- {line}" for line in assistant_lines)
            + "\n"
        )
        self._write_note(
            self._paths.current_focus,
            self._render_note(
                title="Current Focus",
                purpose="Latest bounded snapshot of what Hermes was working on. Overwritten, not appended.",
                body=body,
            ),
        )


def register(ctx):
    ctx.register_memory_provider(ObsidianMemoryProvider())
