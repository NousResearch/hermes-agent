"""File-backed Project/Batch Context Cards for Hermes sessions.

The context card is deliberately Markdown-first: it is human-readable, easy to
sync into an Obsidian/repo handoff, and does not require a SQLite schema change.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from hermes_constants import get_hermes_home
from utils import atomic_replace

ContextStatus = Literal["active", "paused", "done"]


@dataclass
class ContextCard:
    id: str
    project_id: str
    batch_id: str | None
    status: ContextStatus
    source_surface: str | None = None
    session_id: str | None = None
    handoff_path: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    paused_at: str | None = None
    done_at: str | None = None
    resume_prompt: str | None = None
    body: str = ""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(value: str | None, *, fallback: str = "context") -> str:
    text = unicodedata.normalize("NFKC", value or "").strip().lower()
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"\.\.+", "-", text)
    text = re.sub(r"[^\w가-힣.-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-+", "-", text).strip("-._")
    if not text:
        text = fallback
    return text[:96]


def context_namespace(value: str | None, *, fallback: str = "default") -> str:
    """Return a stable, safe namespace for per-surface context-card state."""
    raw = (value or fallback).strip() or fallback
    slug = _slug(raw, fallback=fallback)[:48]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def _escape(value: str | None) -> str:
    if value is None or value == "":
        return "null"
    value = str(value).replace("\n", "\\n")
    if re.search(r"[:#\[\]{}]|^null$|^true$|^false$", value, re.IGNORECASE):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _unescape(value: str) -> str | None:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1].replace('\\"', '"')
    return value.replace("\\n", "\n")


def _split_frontmatter(text: str) -> tuple[dict[str, str | None], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + 5 :]
    data: dict[str, str | None] = {}
    for line in raw.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _unescape(value)
    return data, body


def _frontmatter(data: dict[str, str | None]) -> str:
    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {_escape(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _replace_or_append_section(body: str, heading: str, content: str) -> str:
    marker = f"## {heading}"
    section = f"{marker}\n\n{content.strip()}\n"
    if marker not in body:
        return body.rstrip() + "\n\n" + section
    start = body.index(marker)
    next_match = re.search(r"\n##\s+", body[start + len(marker) :])
    if next_match:
        end = start + len(marker) + next_match.start() + 1
        return body[:start] + section.rstrip() + "\n\n" + body[end:]
    return body[:start] + section


class ContextCardStore:
    """Store Context Cards as Markdown files under a small directory tree."""

    def __init__(self, root: str | Path | None = None):
        env_root = os.getenv("HERMES_CONTEXT_CARDS_DIR")
        if root is None and env_root:
            root = env_root
        if root is None:
            root = get_hermes_home() / "context-cards"
        self.root = Path(root).expanduser()
        self.cards_dir = self.root / "cards"
        self.index_path = self.root / "_index.md"

    def _ensure_dirs(self) -> None:
        self.cards_dir.mkdir(parents=True, exist_ok=True)

    def _card_path(self, card_id: str) -> Path:
        safe_id = _slug(card_id)
        path = (self.cards_dir / f"{safe_id}.md").resolve()
        cards_root = self.cards_dir.resolve()
        if not path.is_relative_to(cards_root):
            raise ValueError("Context card path escaped cards directory")
        return path

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, path)

    def _read_index(self) -> dict[str, str | None]:
        if not self.index_path.exists():
            return {"active_card_id": None}
        data, _body = _split_frontmatter(self.index_path.read_text(encoding="utf-8"))
        return {"active_card_id": data.get("active_card_id")}

    def _write_index(self, active_card_id: str | None) -> None:
        self._ensure_dirs()
        lines = ["# Hermes Context Cards", ""]
        for card in self.list_cards():
            marker = "active" if card.id == active_card_id else card.status
            lines.append(f"- [[cards/{card.id}|{card.project_id}]] — {marker}")
        text = _frontmatter({"active_card_id": active_card_id, "updated_at": _now()}) + "\n".join(lines) + "\n"
        self._write_text(self.index_path, text)

    def _serialize(self, card: ContextCard) -> str:
        data = {
            "id": card.id,
            "project_id": card.project_id,
            "batch_id": card.batch_id,
            "status": card.status,
            "source_surface": card.source_surface,
            "session_id": card.session_id,
            "handoff_path": card.handoff_path,
            "created_at": card.created_at,
            "updated_at": card.updated_at,
            "paused_at": card.paused_at,
            "done_at": card.done_at,
            "resume_prompt": card.resume_prompt,
        }
        return _frontmatter(data) + card.body.rstrip() + "\n"

    def _deserialize(self, text: str) -> ContextCard:
        data, body = _split_frontmatter(text)
        return ContextCard(
            id=str(data.get("id") or "context"),
            project_id=str(data.get("project_id") or "unknown"),
            batch_id=data.get("batch_id"),
            status=(data.get("status") or "paused"),  # type: ignore[arg-type]
            source_surface=data.get("source_surface"),
            session_id=data.get("session_id"),
            handoff_path=data.get("handoff_path"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            paused_at=data.get("paused_at"),
            done_at=data.get("done_at"),
            resume_prompt=data.get("resume_prompt"),
            body=body,
        )

    def save_card(self, card: ContextCard) -> ContextCard:
        card.updated_at = _now()
        self._write_text(self._card_path(card.id), self._serialize(card))
        return card

    def get_card(self, card_id: str) -> ContextCard | None:
        path = self._card_path(card_id)
        if not path.exists():
            return None
        return self._deserialize(path.read_text(encoding="utf-8"))

    def list_cards(self) -> list[ContextCard]:
        if not self.cards_dir.exists():
            return []
        cards: list[ContextCard] = []
        for path in sorted(self.cards_dir.glob("*.md")):
            try:
                cards.append(self._deserialize(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return cards

    def active_card(self) -> ContextCard | None:
        active_id = self._read_index().get("active_card_id")
        return self.get_card(active_id) if active_id else None

    def _unique_id(self, preferred: str) -> str:
        base = _slug(preferred)
        candidate = base
        suffix = 2
        while self._card_path(candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def create_card(
        self,
        project_id: str,
        *,
        batch_id: str | None = None,
        source_surface: str | None = None,
        session_id: str | None = None,
        handoff_path: str | None = None,
        switch: bool = True,
    ) -> ContextCard:
        self._ensure_dirs()
        now = _now()
        card_id = self._unique_id(batch_id or project_id)
        body = (
            f"# {project_id}\n\n"
            "## Goal\n\n"
            "- Context Card로 현재 Project/Batch를 고정한다.\n\n"
            "## Current State\n\n"
            "- 새 context가 생성됨.\n\n"
            "## Next Actions\n\n"
            "- 다음 작업을 기록한다.\n\n"
            "## Blockers\n\n"
            "- 없음.\n"
        )
        card = ContextCard(
            id=card_id,
            project_id=project_id,
            batch_id=batch_id,
            status="active" if switch else "paused",
            source_surface=source_surface,
            session_id=session_id,
            handoff_path=handoff_path,
            created_at=now,
            updated_at=now,
            body=body,
        )
        if switch:
            current = self.active_card()
            if current and current.id != card.id and current.status == "active":
                current.status = "paused"
                current.paused_at = now
                self.save_card(current)
        self.save_card(card)
        self._write_index(card.id if switch else self._read_index().get("active_card_id"))
        return card

    def switch_context(self, project_id: str, *, batch_id: str | None = None, session_id: str | None = None) -> ContextCard:
        target_id = _slug(batch_id or project_id)
        existing = self.get_card(target_id)
        if existing is None:
            return self.create_card(project_id, batch_id=batch_id, session_id=session_id, switch=True)
        now = _now()
        current = self.active_card()
        if current and current.id != existing.id and current.status == "active":
            current.status = "paused"
            current.paused_at = now
            self.save_card(current)
        existing.status = "active"
        existing.paused_at = None
        if session_id:
            existing.session_id = session_id
        self.save_card(existing)
        self._write_index(existing.id)
        return existing

    def mark_done(self, card_id: str, *, summary: str = "", resume_prompt: str | None = None) -> ContextCard:
        card = self.get_card(card_id)
        if card is None:
            raise KeyError(f"Context card not found: {card_id}")
        now = _now()
        prompt = resume_prompt or (
            f"Q가 `{card.project_id} / {card.batch_id or card.id}`를 재개하면, "
            "이 Context Card와 handoff를 읽고 다음 승인 지점부터 진행한다."
        )
        card.status = "done"
        card.done_at = now
        card.resume_prompt = prompt
        content = prompt if not summary else f"{prompt}\n\n완료 메모: {summary}"
        card.body = _replace_or_append_section(card.body, "Resume Prompt", content)
        self.save_card(card)
        if self._read_index().get("active_card_id") == card.id:
            self._write_index(None)
        return card

    def pause_active(self, note: str = "") -> ContextCard:
        card = self.active_card()
        if card is None:
            raise KeyError("No active context card")
        card.status = "paused"
        card.paused_at = _now()
        if note:
            card.body = _replace_or_append_section(card.body, "Pause Note", note)
        self.save_card(card)
        self._write_index(None)
        return card

    def generate_handoff(self, card_id: str, *, target: str | None = None) -> str:
        card = self.get_card(card_id)
        if card is None:
            raise KeyError(f"Context card not found: {card_id}")
        worker = target or "Spark"
        return (
            "Spark handoff 초안:\n\n"
            f"Project: {card.project_id}\n"
            f"Batch: {card.batch_id or card.id}\n"
            f"Worker: {worker}\n"
            "목표: Context Control Layer 구현/검증을 이어간다.\n"
            "금지: DB/schema 최종 결정, core architecture 독단 변경, secrets/API/auth 작업, 같은 파일 동시 수정\n"
            "산출물: 구현 diff, 테스트 결과, 스모크테스트, QA 메모\n"
        )

    def format_status(self, card: ContextCard | None = None) -> str:
        card = card or self.active_card()
        if card is None:
            return "현재 고정된 Context가 없어.\n\n추천: `/context new <project> <batch>`로 새 Batch를 열어."
        return (
            f"Project: {card.project_id}\n"
            f"Batch: {card.batch_id or card.id}\n"
            f"상태: {card.status}\n"
            f"정본: {card.handoff_path or str(self._card_path(card.id))}\n"
            "다음 액션: Context Card를 기준으로 작업을 이어간다."
        )


def handle_context_command(args: str = "", *, store: ContextCardStore | None = None, session_id: str | None = None, source_surface: str | None = None) -> str:
    store = store or ContextCardStore()
    raw = (args or "").strip()
    parts = raw.split()
    action = parts[0].lower() if parts else "status"
    rest = parts[1:]
    if action in {"status", "show", "current", "now"}:
        return store.format_status()
    if action == "new":
        project = rest[0] if rest else "hermes-agent"
        batch = rest[1] if len(rest) > 1 else None
        card = store.create_card(project, batch_id=batch, session_id=session_id, source_surface=source_surface)
        return "새 Context 고정:\n\n" + store.format_status(card)
    if action == "switch":
        project = rest[0] if rest else "hermes-agent"
        batch = rest[1] if len(rest) > 1 else None
        card = store.switch_context(project, batch_id=batch, session_id=session_id)
        return "Context 전환 완료:\n\n" + store.format_status(card)
    if action == "pause":
        card = store.pause_active(" ".join(rest))
        return f"현재 Batch를 paused로 표시했어.\n\nProject: {card.project_id}\nBatch: {card.batch_id or card.id}\n상태: paused"
    if action == "done":
        active = store.active_card()
        if active is None:
            return store.format_status(None)
        card = store.mark_done(active.id, summary=" ".join(rest))
        return f"Batch 종료 완료:\n\nProject: {card.project_id}\nBatch: {card.batch_id or card.id}\n상태: done\n\nResume Prompt:\n{card.resume_prompt}"
    if action == "handoff":
        active = store.active_card()
        if active is None:
            return store.format_status(None)
        target = rest[0] if rest else "Spark"
        return store.generate_handoff(active.id, target=target)
    raise ValueError(f"Unknown context action: {action}")
