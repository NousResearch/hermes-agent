from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

_CONTEXT_HEADER = (
    "[LIFEOS CONTEXT — REFERENCE ONLY] This is structured operating context from "
    "Tony's LifeOS vault. Use it as current background context. Prefer the latest "
    "user request if anything conflicts."
)

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_STATUS_RANK = {"active": 0, "waiting": 1, "someday": 2, "done": 9}


@dataclass
class VaultNote:
    path: Path
    title: str
    body: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def candidates(self) -> List[str]:
        names = [self.stem, self.title, *self.aliases]
        out: List[str] = []
        seen = set()
        for name in names:
            norm = _normalize_phrase(name)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out


class LifeOSContextEngine(ContextCompressor):
    def __init__(self) -> None:
        super().__init__(
            model="lifeos-bootstrap",
            quiet_mode=True,
            config_context_length=200_000,
        )
        self.hermes_home = os.path.expanduser("~/.hermes")
        self.session_id = ""
        self.vault_path = Path(
            os.getenv("OBSIDIAN_VAULT_PATH", "~/Documents/Obsidian Vault")
        ).expanduser()
        self.state_path = Path(self.hermes_home) / "context" / "lifeos_state.json"
        self.max_block_chars = 5000
        self.max_project_chars = 1200
        self.max_people_chars = 900
        self.max_claude_chars = 1200
        self.max_daily_chars = 900
        self.max_task_count = 8
        self.refresh_minutes = {
            "base": 20,
            "daily": 30,
            "tasks": 20,
            "projects": 20,
            "people": 30,
        }
        self.state: Dict[str, Any] = {
            "session_id": "",
            "current_focus": "",
            "active_projects": [],
            "active_people": [],
            "active_tasks": [],
            "tentative_updates": [],
            "promotion_candidates": [],
            "last_refresh": {},
            "pinned_blocks": {"base": "", "projects": {}, "people": {}},
        }
        self._project_notes: List[VaultNote] = []
        self._people_notes: List[VaultNote] = []

    @property
    def name(self) -> str:
        return "lifeos"

    def is_available(self) -> bool:
        return self.vault_path.exists()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or ""
        self.hermes_home = str(kwargs.get("hermes_home") or self.hermes_home)
        self.state_path = Path(self.hermes_home) / "context" / "lifeos_state.json"
        self._load_runtime_config()
        self._load_state()
        self._ensure_state_shape()
        self.state["session_id"] = self.session_id
        self._rebuild_indexes()
        self._refresh_base_context(force=True)
        self._save_state()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.state["session_id"] = session_id or self.state.get("session_id", "")
        self._save_state()

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.state["current_focus"] = ""
        self.state["active_projects"] = []
        self.state["active_people"] = []
        self.state["active_tasks"] = []
        self.state["tentative_updates"] = []
        self.state["promotion_candidates"] = []
        self._save_state()

    def prefetch(self, query: str, **kwargs) -> str:
        if not self.is_available():
            return ""

        self._load_runtime_config()
        self._rebuild_indexes()

        normalized_query = _normalize_phrase(query)
        wants_day = any(
            token in normalized_query
            for token in ("today", "task", "tasks", "priority", "priorities", "agenda", "focus", "next")
        )
        if wants_day or self._is_stale("base"):
            self._refresh_base_context(force=wants_day)

        matched_projects = self._match_notes(normalized_query, self._project_notes)
        matched_people = self._match_notes(normalized_query, self._people_notes)

        if not matched_projects and self.state.get("active_projects"):
            matched_projects = self._match_prior_state(self.state.get("active_projects", []), self._project_notes)
        if not matched_people and self.state.get("active_people"):
            matched_people = self._match_prior_state(self.state.get("active_people", []), self._people_notes)

        blocks: List[str] = []
        base_block = self.state.get("pinned_blocks", {}).get("base", "") or ""
        if wants_day or (not matched_projects and not matched_people):
            if base_block:
                blocks.append(base_block)
        else:
            snapshot = self._render_operational_snapshot(max_tasks=4)
            if snapshot:
                blocks.append(snapshot)

        if matched_projects:
            blocks.extend(self._render_project_bundle(note) for note in matched_projects[:2])
        if matched_people:
            blocks.extend(self._render_person_bundle(note) for note in matched_people[:2])

        blocks = [block.strip() for block in blocks if block and block.strip()]
        if not blocks:
            return ""

        self.state["active_projects"] = [note.title for note in matched_projects[:2]]
        self.state["active_people"] = [note.title for note in matched_people[:2]]
        if query and query.strip():
            self.state["current_focus"] = query.strip()[:240]
        self._save_state()

        joined = _trim_text("\n\n".join(blocks), self.max_block_chars)
        return f"{_CONTEXT_HEADER}\n\n{joined}" if joined else ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "lifeos_context_status",
                "description": "Inspect the current LifeOS context-engine state and active cached context.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "lifeos_refresh_context",
                "description": "Refresh cached LifeOS context from the vault. Optionally target a project or person.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["base", "project", "person", "all"],
                            "description": "What to refresh. Default: all.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Optional project or person name to refresh and preview.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_set_focus",
                "description": "Set the current focus for the LifeOS context engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {"type": "string", "description": "Short description of the current focus."}
                    },
                    "required": ["focus"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_capture_update",
                "description": "Capture a tentative session-local operational update without writing it into LifeOS notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "Update type, e.g. focus_shift, project_update, decision, idea."},
                        "target": {"type": "string", "description": "Optional related project, person, or topic."},
                        "content": {"type": "string", "description": "The tentative update text to store in the session overlay."},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"], "description": "Confidence in the update. Default: medium."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_promote_to_project",
                "description": "Queue a promotion candidate for a LifeOS project note without writing it yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Project name or identifier."},
                        "content": {"type": "string", "description": "Operational update to potentially write into the project note."},
                        "why": {"type": "string", "description": "Why this should be promoted."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_promote_to_task",
                "description": "Queue a promotion candidate for a new or updated task without writing it yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Optional related project or task grouping."},
                        "content": {"type": "string", "description": "Task text or update to promote later."},
                        "why": {"type": "string", "description": "Why this should become a task candidate."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_promote_to_honcho",
                "description": "Queue a promotion candidate for Honcho memory without writing it yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Potential durable user-memory conclusion."},
                        "why": {"type": "string", "description": "Why this belongs in durable memory."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_promote_to_daily",
                "description": "Write selected context directly into today's daily note under a session-promoted section.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text to append to today's daily note."}
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_promote_to_claude",
                "description": "Write selected context directly into CLAUDE.md under a session-promoted section.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text to append to CLAUDE.md."}
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_apply_promotion_candidate",
                "description": "Apply a queued promotion candidate to its LifeOS destination when supported.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Zero-based index in promotion_candidates."}
                    },
                    "required": ["index"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lifeos_review_promotion_candidate",
                "description": "Mark a queued promotion candidate as approved, rejected, or pending before apply.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Zero-based index in promotion_candidates."},
                        "action": {"type": "string", "enum": ["approve", "reject", "pending"], "description": "Review action to apply."}
                    },
                    "required": ["index", "action"],
                    "additionalProperties": False,
                },
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if name == "lifeos_context_status":
                return json.dumps(self._status_payload(), ensure_ascii=False)
            if name == "lifeos_set_focus":
                focus = (args or {}).get("focus", "")
                self.state["current_focus"] = str(focus).strip()[:240]
                self._save_state()
                return json.dumps({"ok": True, "focus": self.state["current_focus"]}, ensure_ascii=False)
            if name == "lifeos_refresh_context":
                scope = ((args or {}).get("scope") or "all").strip().lower()
                target = ((args or {}).get("target") or "").strip()
                preview = self._refresh_via_tool(scope=scope, target=target)
                return json.dumps(
                    {"ok": True, "scope": scope, "target": target, "preview": preview, "status": self._status_payload()},
                    ensure_ascii=False,
                )
            if name == "lifeos_capture_update":
                entry = self._record_tentative_update(args or {})
                return json.dumps({"ok": True, "tentative_update": entry}, ensure_ascii=False)
            if name == "lifeos_promote_to_project":
                entry = self._queue_promotion_candidate("lifeos_project", args or {})
                return json.dumps({"ok": True, "promotion_candidate": entry}, ensure_ascii=False)
            if name == "lifeos_promote_to_task":
                entry = self._queue_promotion_candidate("lifeos_task", args or {})
                return json.dumps({"ok": True, "promotion_candidate": entry}, ensure_ascii=False)
            if name == "lifeos_promote_to_honcho":
                entry = self._queue_promotion_candidate("honcho_user", args or {})
                return json.dumps({"ok": True, "promotion_candidate": entry}, ensure_ascii=False)
            if name == "lifeos_promote_to_daily":
                details = self._write_to_daily(str((args or {}).get("content") or "").strip())
                return json.dumps({"ok": True, **details}, ensure_ascii=False)
            if name == "lifeos_promote_to_claude":
                details = self._write_to_claude(str((args or {}).get("content") or "").strip())
                return json.dumps({"ok": True, **details}, ensure_ascii=False)
            if name == "lifeos_apply_promotion_candidate":
                details = self._apply_promotion_candidate(args or {})
                return json.dumps({"ok": True, "applied": details}, ensure_ascii=False)
            if name == "lifeos_review_promotion_candidate":
                details = self._review_promotion_candidate(args or {})
                return json.dumps({"ok": True, "candidate": details}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("LifeOS context tool failed: %s", exc)
            return json.dumps({"error": f"LifeOS context tool failed: {exc}"}, ensure_ascii=False)
        return super().handle_tool_call(name, args, **kwargs)

    def _status_payload(self) -> Dict[str, Any]:
        pinned = self.state.get("pinned_blocks", {}) if isinstance(self.state, dict) else {}
        return {
            "engine": self.name,
            "vault_path": str(self.vault_path),
            "vault_available": self.is_available(),
            "session_id": self.session_id,
            "current_focus": self.state.get("current_focus", ""),
            "active_projects": self.state.get("active_projects", []),
            "active_people": self.state.get("active_people", []),
            "active_tasks": self.state.get("active_tasks", []),
            "tentative_updates": self.state.get("tentative_updates", []),
            "promotion_candidates": self.state.get("promotion_candidates", []),
            "last_refresh": self.state.get("last_refresh", {}),
            "base_preview": _trim_text(pinned.get("base", ""), 800),
        }

    def _refresh_via_tool(self, scope: str, target: str) -> str:
        self._rebuild_indexes()
        preview_blocks: List[str] = []
        if scope in {"base", "all"}:
            self._refresh_base_context(force=True)
            preview_blocks.append(self.state.get("pinned_blocks", {}).get("base", ""))
        if scope in {"project", "all"} and target:
            matches = self._match_notes(_normalize_phrase(target), self._project_notes)
            if matches:
                block = self._render_project_bundle(matches[0])
                self.state.setdefault("pinned_blocks", {}).setdefault("projects", {})[matches[0].title] = block
                preview_blocks.append(block)
        if scope in {"person", "all"} and target:
            matches = self._match_notes(_normalize_phrase(target), self._people_notes)
            if matches:
                block = self._render_person_bundle(matches[0])
                self.state.setdefault("pinned_blocks", {}).setdefault("people", {})[matches[0].title] = block
                preview_blocks.append(block)
        self._save_state()
        return _trim_text("\n\n".join(block for block in preview_blocks if block), self.max_block_chars)

    def _load_runtime_config(self) -> None:
        cfg_path = Path(self.hermes_home) / "config.yaml"
        config: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                config = yaml.safe_load(cfg_path.read_text()) or {}
            except Exception as exc:
                logger.debug("LifeOS context could not read %s: %s", cfg_path, exc)
        life_cfg = config.get("lifeos_context", {}) if isinstance(config, dict) else {}
        vault_value = (
            life_cfg.get("vault_path")
            or os.getenv("LIFEOS_CONTEXT_VAULT_PATH")
            or os.getenv("OBSIDIAN_VAULT_PATH")
            or str(self.vault_path)
        )
        self.vault_path = Path(vault_value).expanduser()
        self.max_block_chars = _safe_int(life_cfg.get("max_block_chars"), self.max_block_chars)
        self.max_task_count = _safe_int(life_cfg.get("max_task_count"), self.max_task_count)
        refresh_cfg = life_cfg.get("refresh", {}) if isinstance(life_cfg, dict) else {}
        if isinstance(refresh_cfg, dict):
            for key, current in list(self.refresh_minutes.items()):
                self.refresh_minutes[key] = _safe_int(refresh_cfg.get(f"{key}_minutes"), current)

    def _load_state(self) -> None:
        try:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
        except Exception as exc:
            logger.debug("LifeOS context state load failed: %s", exc)
        self._ensure_state_shape()

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
            tmp_path.replace(self.state_path)
        except Exception as exc:
            logger.debug("LifeOS context state save failed: %s", exc)

    def _ensure_state_shape(self) -> None:
        self.state.setdefault("session_id", "")
        self.state.setdefault("current_focus", "")
        self.state.setdefault("active_projects", [])
        self.state.setdefault("active_people", [])
        self.state.setdefault("active_tasks", [])
        self.state.setdefault("tentative_updates", [])
        self.state.setdefault("promotion_candidates", [])
        self.state.setdefault("last_refresh", {})
        pinned = self.state.setdefault("pinned_blocks", {})
        if not isinstance(pinned, dict):
            pinned = {}
            self.state["pinned_blocks"] = pinned
        pinned.setdefault("base", "")
        pinned.setdefault("projects", {})
        pinned.setdefault("people", {})

    def _record_tentative_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "type": str(args.get("type") or "note").strip() or "note",
            "target": str(args.get("target") or "").strip(),
            "content": str(args.get("content") or "").strip(),
            "source": "conversation",
            "confidence": str(args.get("confidence") or "medium").strip().lower() or "medium",
            "created_at": _utc_now_iso(),
        }
        self.state.setdefault("tentative_updates", []).append(entry)
        self.state["tentative_updates"] = self.state["tentative_updates"][-20:]
        self._save_state()
        return entry

    def _queue_promotion_candidate(self, candidate_type: str, args: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "type": candidate_type,
            "target": str(args.get("target") or "").strip(),
            "content": str(args.get("content") or "").strip(),
            "why": str(args.get("why") or "").strip(),
            "status": "pending",
            "created_at": _utc_now_iso(),
        }
        self.state.setdefault("promotion_candidates", []).append(entry)
        self.state["promotion_candidates"] = self.state["promotion_candidates"][-20:]
        self._save_state()
        return entry

    def _write_to_daily(self, content: str) -> Dict[str, Any]:
        if not content:
            raise ValueError("content is required")
        path = self._get_daily_note_path(create_if_missing=True)
        self._append_section_block(path, "Session Promoted Context", content)
        return {"path": str(path), "content": content}

    def _write_to_claude(self, content: str) -> Dict[str, Any]:
        if not content:
            raise ValueError("content is required")
        path = self._get_claude_path(create_if_missing=True)
        self._append_section_block(path, "Session Promoted Context", content)
        return {"path": str(path), "content": content}

    def _apply_promotion_candidate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self.state.setdefault("promotion_candidates", [])
        try:
            index = int(args.get("index"))
        except Exception as exc:
            raise ValueError("valid index is required") from exc
        if index < 0 or index >= len(candidates):
            raise IndexError("promotion candidate index out of range")
        candidate = candidates[index]
        candidate_type = candidate.get("type", "")
        content = str(candidate.get("content") or "").strip()
        target = str(candidate.get("target") or "").strip()
        if not content:
            raise ValueError("promotion candidate has no content")
        if candidate.get("status") == "rejected":
            raise ValueError("rejected promotion candidates cannot be applied")

        if candidate_type == "lifeos_project":
            project_note = self._find_project_note(target)
            if not project_note:
                raise ValueError(f"project note not found for target: {target}")
            self._append_section_block(project_note.path, "Session Promoted Context", content)
            candidate["status"] = "applied"
            candidate["applied_at"] = _utc_now_iso()
            candidate["destination"] = str(project_note.path)
        elif candidate_type == "lifeos_task":
            task_path = self._create_task_candidate_note(content, target)
            candidate["status"] = "applied"
            candidate["applied_at"] = _utc_now_iso()
            candidate["destination"] = str(task_path)
        elif candidate_type == "honcho_user":
            self._apply_honcho_conclusion(content)
            candidate["status"] = "applied"
            candidate["applied_at"] = _utc_now_iso()
            candidate["destination"] = "honcho"
        else:
            raise ValueError(f"promotion candidate type not supported for direct apply: {candidate_type}")

        self._save_state()
        return candidate

    def _review_promotion_candidate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self.state.setdefault("promotion_candidates", [])
        try:
            index = int(args.get("index"))
        except Exception as exc:
            raise ValueError("valid index is required") from exc
        if index < 0 or index >= len(candidates):
            raise IndexError("promotion candidate index out of range")
        action = str(args.get("action") or "").strip().lower()
        mapping = {"approve": "approved", "reject": "rejected", "pending": "pending"}
        if action not in mapping:
            raise ValueError("action must be one of: approve, reject, pending")
        candidates[index]["status"] = mapping[action]
        candidates[index]["reviewed_at"] = _utc_now_iso()
        self._save_state()
        return candidates[index]

    def _rebuild_indexes(self) -> None:
        self._project_notes = self._load_note_index(self.vault_path / "memory" / "projects")
        self._people_notes = self._load_note_index(self.vault_path / "memory" / "people")

    def _load_note_index(self, directory: Path) -> List[VaultNote]:
        if not directory.exists():
            return []
        notes: List[VaultNote] = []
        for path in sorted(directory.glob("*.md")):
            note = self._read_vault_note(path)
            if note:
                notes.append(note)
        return notes

    def _read_vault_note(self, path: Path) -> Optional[VaultNote]:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)
        aliases = frontmatter.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        title = (
            frontmatter.get("title")
            or _extract_heading(body)
            or _clean_wikilink(frontmatter.get("project") or "")
            or path.stem.replace("-", " ").title()
        )
        return VaultNote(path=path, title=str(title).strip(), body=body.strip(), frontmatter=frontmatter, aliases=[str(a) for a in aliases])

    def _refresh_base_context(self, force: bool = False) -> None:
        if not force and not self._is_stale("base"):
            return
        base_blocks = [self._render_operational_snapshot(max_tasks=self.max_task_count)]
        claude_block = self._render_claude_block()
        if claude_block:
            base_blocks.append(claude_block)
        daily_block = self._render_daily_block()
        if daily_block:
            base_blocks.append(daily_block)
        combined = _trim_text("\n\n".join(block for block in base_blocks if block), self.max_block_chars)
        self.state.setdefault("pinned_blocks", {})["base"] = combined
        self.state.setdefault("last_refresh", {})["base"] = _utc_now_iso()

    def _render_operational_snapshot(self, max_tasks: int) -> str:
        tasks = self._load_tasks()
        self.state["active_tasks"] = [task["title"] for task in tasks[:max_tasks]]
        lines = ["## Operational Snapshot"]
        if self.state.get("current_focus"):
            lines.append(f"- Current focus: {self.state['current_focus']}")
        if tasks:
            lines.append("- Active tasks:")
            for task in tasks[:max_tasks]:
                suffix_parts = []
                if task.get("priority"):
                    suffix_parts.append(task["priority"])
                if task.get("due"):
                    suffix_parts.append(f"due {task['due']}")
                if task.get("project"):
                    suffix_parts.append(task["project"])
                suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
                lines.append(f"  - {task['title']}{suffix}")
        top_projects = [task.get("project") for task in tasks if task.get("project")]
        top_projects = [project for project in top_projects if project]
        if top_projects:
            deduped = []
            seen = set()
            for project in top_projects:
                key = _normalize_phrase(project)
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(project)
            if deduped:
                lines.append("- Active projects: " + ", ".join(deduped[:4]))
        return "\n".join(lines)

    def _render_claude_block(self) -> str:
        candidates = [self.vault_path / "CLAUDE.md", self.vault_path / "Daily Control Center" / "CLAUDE.md"]
        for path in candidates:
            if path.exists():
                text = _trim_text(_strip_markdown_noise(path.read_text(encoding="utf-8")), self.max_claude_chars)
                if text:
                    return f"## Command Center Memory\n{text}"
        return ""

    def _render_daily_block(self) -> str:
        daily_dir = self.vault_path / "daily"
        if not daily_dir.exists():
            return ""
        today_name = datetime.now().strftime("%Y-%m-%d") + ".md"
        today_path = daily_dir / today_name
        target = today_path if today_path.exists() else None
        if target is None:
            candidates = sorted(daily_dir.glob("*.md"), reverse=True)
            target = candidates[0] if candidates else None
        if not target:
            return ""
        body = self._read_vault_note(target)
        if not body:
            return ""
        summary = _trim_text(_strip_markdown_noise(body.body), self.max_daily_chars)
        if not summary:
            return ""
        label = "today" if target.name == today_name else f"latest note ({target.stem})"
        return f"## Daily Note — {label}\n{summary}"

    def _render_project_bundle(self, note: VaultNote) -> str:
        summary = _trim_text(_strip_markdown_noise(note.body), self.max_project_chars)
        related_tasks = [task for task in self._load_tasks() if self._same_topic(task.get("project", ""), note.title, note.stem)]
        lines = [f"## Project — {note.title}"]
        if summary:
            lines.append(summary)
        if related_tasks:
            lines.append("Related tasks:")
            for task in related_tasks[:4]:
                due = f" — due {task['due']}" if task.get("due") else ""
                lines.append(f"- {task['title']}{due}")
        return "\n".join(lines)

    def _render_person_bundle(self, note: VaultNote) -> str:
        summary = _trim_text(_strip_markdown_noise(note.body), self.max_people_chars)
        related_tasks = []
        for task in self._load_tasks():
            who = f"{task.get('assigned_to', '')} {task.get('waiting_on', '')}".strip()
            if who and self._same_topic(who, note.title, note.stem):
                related_tasks.append(task)
        lines = [f"## Person — {note.title}"]
        if summary:
            lines.append(summary)
        if related_tasks:
            lines.append("Related tasks:")
            for task in related_tasks[:4]:
                due = f" — due {task['due']}" if task.get("due") else ""
                lines.append(f"- {task['title']}{due}")
        return "\n".join(lines)

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks_dir = self.vault_path / "tasks"
        if not tasks_dir.exists():
            return []
        tasks: List[Dict[str, Any]] = []
        for path in tasks_dir.glob("*.md"):
            note = self._read_vault_note(path)
            if not note:
                continue
            tags = note.frontmatter.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if tags and "task" not in [str(tag).lower() for tag in tags]:
                continue
            status = str(note.frontmatter.get("status") or "active").lower()
            if status == "done":
                continue
            project = _clean_wikilink(note.frontmatter.get("project") or "")
            assigned_to = _clean_wikilink(note.frontmatter.get("assigned_to") or note.frontmatter.get("assigned-to") or "")
            waiting_on = _clean_wikilink(note.frontmatter.get("waiting_on") or note.frontmatter.get("waiting-on") or "")
            due_value = _format_due(note.frontmatter.get("due"))
            tasks.append(
                {
                    "title": note.title,
                    "path": str(path),
                    "priority": str(note.frontmatter.get("priority") or "medium").lower(),
                    "status": status,
                    "project": project,
                    "assigned_to": assigned_to,
                    "waiting_on": waiting_on,
                    "due": due_value,
                }
            )
        tasks.sort(key=self._task_sort_key)
        return tasks

    def _task_sort_key(self, task: Dict[str, Any]) -> tuple:
        due_key = task.get("due") or "9999-12-31"
        return (
            _STATUS_RANK.get(task.get("status", "active"), 50),
            _PRIORITY_RANK.get(task.get("priority", "medium"), 50),
            due_key,
            task.get("title", ""),
        )

    def _match_notes(self, normalized_query: str, notes: Iterable[VaultNote]) -> List[VaultNote]:
        if not normalized_query:
            return []
        scored = []
        for note in notes:
            best = 0
            for candidate in note.candidates:
                if candidate and candidate in normalized_query:
                    best = max(best, len(candidate))
            if best:
                scored.append((best, note.title.lower(), note))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [note for _, _, note in scored]

    def _match_prior_state(self, names: Iterable[str], notes: Iterable[VaultNote]) -> List[VaultNote]:
        wanted = {_normalize_phrase(name) for name in names if name}
        matches = []
        for note in notes:
            if wanted.intersection(note.candidates):
                matches.append(note)
        return matches

    def _find_project_note(self, target: str) -> Optional[VaultNote]:
        normalized = _normalize_phrase(target)
        if not normalized:
            return None
        matches = self._match_notes(normalized, self._project_notes)
        return matches[0] if matches else None

    def _get_daily_note_path(self, create_if_missing: bool = False) -> Path:
        daily_dir = self.vault_path / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        today_name = datetime.now().strftime("%Y-%m-%d") + ".md"
        path = daily_dir / today_name
        if create_if_missing and not path.exists():
            path.write_text(f"# {today_name[:-3]}\n", encoding="utf-8")
        return path

    def _get_claude_path(self, create_if_missing: bool = False) -> Path:
        path = self.vault_path / "CLAUDE.md"
        if create_if_missing and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Memory\n", encoding="utf-8")
        return path

    def _append_section_block(self, path: Path, heading: str, content: str) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        block = f"## {heading}\n- {content.strip()}\n"
        if not existing.strip():
            path.write_text(block, encoding="utf-8")
            return
        section_header = f"## {heading}"
        if section_header in existing:
            updated = existing.rstrip() + f"\n- {content.strip()}\n"
        else:
            spacer = "\n\n" if not existing.endswith("\n\n") else ""
            updated = existing.rstrip() + f"{spacer}\n{block}"
        path.write_text(updated, encoding="utf-8")

    def _create_task_candidate_note(self, content: str, target: str) -> Path:
        tasks_dir = self.vault_path / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        title = content.strip().splitlines()[0].strip()
        slug = _slugify(title) or f"task-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        path = tasks_dir / f"{slug}.md"
        counter = 2
        while path.exists():
            path = tasks_dir / f"{slug}-{counter}.md"
            counter += 1
        project_line = f"project: '[[{target}]]'\n" if target else ""
        raw = (
            "---\n"
            f"title: {title}\n"
            "tags:\n  - task\n"
            "status: active\n"
            "priority: medium\n"
            f"{project_line}"
            f"created: '{datetime.now().strftime('%Y-%m-%d')}'\n"
            "---\n"
            f"# {title}\n\n"
            f"{content.strip()}\n"
        )
        path.write_text(raw, encoding="utf-8")
        return path

    def _apply_honcho_conclusion(self, content: str) -> None:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        from plugins.memory.honcho.session import HonchoSessionManager

        cfg = HonchoClientConfig.from_global_config()
        client = get_honcho_client(cfg)
        manager = HonchoSessionManager(honcho=client, config=cfg, context_tokens=cfg.context_tokens)
        session_key = cfg.resolve_session_name(cwd=str(self.vault_path), session_id=self.session_id) or self.session_id or "lifeos"
        manager.get_or_create(session_key)
        ok = manager.create_conclusion(session_key, content)
        if not ok:
            raise ValueError("failed to write Honcho conclusion")

    def _is_stale(self, scope: str) -> bool:
        last_refresh = self.state.get("last_refresh", {}).get(scope)
        if not last_refresh:
            return True
        try:
            ts = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
        except Exception:
            return True
        return datetime.now(timezone.utc) - ts >= timedelta(minutes=self.refresh_minutes.get(scope, 20))

    def _same_topic(self, value: str, *candidates: str) -> bool:
        norm_value = _normalize_phrase(value)
        for candidate in candidates:
            norm_candidate = _normalize_phrase(candidate)
            if norm_candidate and norm_value and (norm_candidate in norm_value or norm_value in norm_candidate):
                return True
        return False


def register(ctx) -> None:
    ctx.register_context_engine(LifeOSContextEngine())


def _parse_frontmatter(raw: str) -> tuple[Dict[str, Any], str]:
    if raw.startswith("---\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            frontmatter_text = raw[4:end]
            body = raw[end + 4 :].lstrip("\n")
            try:
                return yaml.safe_load(frontmatter_text) or {}, body
            except Exception:
                return {}, body
    return {}, raw


def _extract_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def _strip_markdown_noise(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and cleaned_lines:
            continue
        cleaned_lines.append(_clean_wikilink(stripped))
    return "\n".join(cleaned_lines)


def _clean_wikilink(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    matches = re.findall(r"\[\[([^\]]+)\]\]", text)
    if matches:
        cleaned = []
        for match in matches:
            target, _, display = match.partition("|")
            cleaned.append((display or target).strip())
        return ", ".join(part for part in cleaned if part)
    return text.strip("'\" ")


def _normalize_phrase(value: Any) -> str:
    text = _clean_wikilink(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cutoff = max_chars - 1
    trimmed = text[:cutoff].rsplit("\n", 1)[0].rsplit(" ", 1)[0].strip()
    return (trimmed or text[:cutoff].strip()) + "…"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_due(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        return text.split("T", 1)[0]
    return text


def _slugify(text: str) -> str:
    text = _normalize_phrase(text)
    return text.replace(" ", "-")
