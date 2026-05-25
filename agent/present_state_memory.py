"""Live present-state memory for active Hermes conversations.

This module stores compact, current facts separately from the curated
``MEMORY.md`` / ``USER.md`` snapshot.  The state is profile-scoped under
``HERMES_HOME/memories`` and is rendered into the per-turn user-message
context so newly learned facts can affect the current conversation without
rebuilding the cached system prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
STATE_FILE_NAME = "PRESENT_STATE.json"


@dataclass
class PresentStateFact:
    """A compact fact that is true or relevant right now."""

    id: str
    scope: str
    target: str
    content: str
    source: str
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_present_state_path() -> Path:
    """Return the profile-scoped present-state JSON path."""
    return get_hermes_home() / "memories" / STATE_FILE_NAME


def _now() -> float:
    return time.time()


def _normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", content.strip()).lower()


def _fact_id(scope: str, target: str, content: str, session_id: str = "") -> str:
    raw = "|".join([scope, target, session_id or "", _normalize_content(content)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _clean_fact_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.strip(" -:\t\r\n")
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "."
    return cleaned


class PresentStateMemory:
    """Profile-scoped live state store for current facts."""

    def __init__(
        self,
        *,
        max_profile_facts: int = 80,
        max_session_facts: int = 80,
        max_fact_chars: int = 360,
        render_char_budget: int = 5000,
        auto_capture: bool = True,
    ) -> None:
        self.max_profile_facts = int(max_profile_facts)
        self.max_session_facts = int(max_session_facts)
        self.max_fact_chars = int(max_fact_chars)
        self.render_char_budget = int(render_char_budget)
        self.auto_capture = bool(auto_capture)
        self._session_id = ""

    def initialize(self, session_id: str = "") -> None:
        """Create the state file if needed and mark the active session."""
        self._session_id = session_id or ""
        with self._file_lock():
            state = self._load_unlocked()
            state["active_session_id"] = self._session_id
            state["updated_at"] = _now()
            self._write_unlocked(state)

    def render_context(self, *, session_id: str = "") -> str:
        """Render present facts for injection into the next model call."""
        sid = session_id or self._session_id
        state = self._load()
        profile_facts = self._coerce_facts(state.get("profile_facts", []))
        session_facts = self._session_facts_from_state(state, sid)

        if not profile_facts and not session_facts:
            return ""

        lines = ["ACTIVE PRESENT STATE (live facts for this conversation)"]
        if profile_facts:
            lines.append("Profile facts:")
            for fact in profile_facts:
                lines.append(f"- {fact.content}")
        if session_facts:
            lines.append("Current session facts:")
            for fact in session_facts:
                lines.append(f"- {fact.content}")

        rendered = "\n".join(lines)
        if self.render_char_budget > 0 and len(rendered) > self.render_char_budget:
            rendered = rendered[: self.render_char_budget].rstrip()
        return rendered

    def capture_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        *,
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror explicit long-term memory writes into live state."""
        if action not in {"add", "replace"}:
            return
        clean = _clean_fact_text(content or "", self.max_fact_chars)
        if not clean:
            return
        scope = "profile"
        fact_target = "user" if target == "user" else "memory"
        self._upsert_fact(
            PresentStateFact(
                id=_fact_id(scope, fact_target, clean),
                scope=scope,
                target=fact_target,
                content=clean,
                source="memory_tool",
                session_id=session_id or self._session_id,
                metadata=dict(metadata or {}),
            )
        )

    def capture_turn(
        self,
        user_content: Any,
        assistant_content: Any,
        *,
        session_id: str = "",
    ) -> None:
        """Extract and store obvious key facts from a completed turn."""
        if not self.auto_capture:
            return
        user_text = user_content if isinstance(user_content, str) else ""
        assistant_text = assistant_content if isinstance(assistant_content, str) else ""
        sid = session_id or self._session_id
        facts = list(self._extract_user_facts(user_text, sid))
        facts.extend(self._extract_session_facts(user_text, assistant_text, sid))
        for fact in facts:
            self._upsert_fact(fact)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        reason: str = "",
    ) -> None:
        """Update active session and optionally carry state across lineage."""
        if not new_session_id:
            return
        with self._file_lock():
            state = self._load_unlocked()
            sessions = state.setdefault("sessions", {})
            sessions.setdefault(new_session_id, {"facts": []})
            if parent_session_id and not reset:
                parent_facts = list(
                    sessions.get(parent_session_id, {}).get("facts", [])
                )
                existing = list(sessions[new_session_id].get("facts", []))
                merged = self._merge_fact_dicts(existing, parent_facts)
                sessions[new_session_id]["facts"] = self._trim_facts(
                    merged, self.max_session_facts
                )
            state["active_session_id"] = new_session_id
            state["updated_at"] = _now()
            state["last_switch_reason"] = reason
            self._session_id = new_session_id
            self._write_unlocked(state)

    def _extract_user_facts(self, text: str, session_id: str) -> Iterable[PresentStateFact]:
        for content in self._remember_statements(text):
            yield self._make_fact("profile", "memory", content, "user_remembered", session_id)

        patterns = [
            (r"\bcall me\s+([^.\n!?]+)", "user", "User wants to be called {0}."),
            (r"\bmy\s+([a-z][a-z0-9 _-]{1,32})\s+is\s+([^.\n!?]+)", "user", "User's {0} is {1}."),
            (r"\bi\s+prefer\s+([^.\n!?]+)", "user", "User prefers {0}."),
            (r"\bi\s+like\s+([^.\n!?]+)", "user", "User likes {0}."),
            (r"\bi\s+use\s+([^.\n!?]+)", "user", "User uses {0}."),
            (r"\bi(?:'m| am)\s+using\s+([^.\n!?]+)", "user", "User is using {0}."),
        ]
        for pattern, target, template in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if len(match.groups()) == 2:
                    content = template.format(match.group(1).strip(), match.group(2).strip())
                else:
                    content = template.format(match.group(1).strip())
                clean = _clean_fact_text(content, self.max_fact_chars)
                if clean:
                    yield self._make_fact("profile", target, clean, "user_statement", session_id)

        shared_patterns = [
            r"\bwe\s+(?:use|are using|prefer|need|run)\s+([^.\n!?]+)",
            r"\bthis project\s+(?:uses|needs|runs|is)\s+([^.\n!?]+)",
        ]
        for pattern in shared_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                content = f"Project/conversation fact: {match.group(0).strip()}."
                yield self._make_fact("profile", "memory", content, "user_statement", session_id)

    def _extract_session_facts(
        self,
        user_text: str,
        assistant_text: str,
        session_id: str,
    ) -> Iterable[PresentStateFact]:
        current_goal_patterns = [
            r"\bwe\s+are\s+going\s+to\s+([^.\n!?]+)",
            r"\bwe(?:'re| are)\s+working\s+on\s+([^.\n!?]+)",
            r"\bcurrent\s+(?:goal|task)\s+is\s+([^.\n!?]+)",
            r"\bthe\s+(?:goal|task)\s+is\s+([^.\n!?]+)",
        ]
        for pattern in current_goal_patterns:
            for match in re.finditer(pattern, user_text, flags=re.IGNORECASE):
                content = f"Current goal: {match.group(1).strip()}."
                yield self._make_fact("session", "conversation", content, "user_goal", session_id)

        if re.search(r"\b(done|implemented|added|fixed|changed|updated)\b", assistant_text, re.IGNORECASE):
            summary = self._first_sentence(assistant_text)
            if summary:
                content = f"Latest assistant outcome: {summary}"
                yield self._make_fact("session", "conversation", content, "assistant_outcome", session_id)

    def _remember_statements(self, text: str) -> Iterable[str]:
        for match in re.finditer(
            r"\bremember(?:\s+(?:that|this))?:?\s+([^.\n!?]+)",
            text,
            flags=re.IGNORECASE,
        ):
            yield match.group(1).strip()

    def _first_sentence(self, text: str) -> str:
        stripped = re.sub(r"\s+", " ", text.strip())
        if not stripped:
            return ""
        match = re.search(r"^(.{20,260}?[.!?])(?:\s|$)", stripped)
        if match:
            return _clean_fact_text(match.group(1), self.max_fact_chars)
        return _clean_fact_text(stripped[:260], self.max_fact_chars)

    def _make_fact(
        self,
        scope: str,
        target: str,
        content: str,
        source: str,
        session_id: str,
    ) -> PresentStateFact:
        clean = _clean_fact_text(content, self.max_fact_chars)
        return PresentStateFact(
            id=_fact_id(scope, target, clean, session_id if scope == "session" else ""),
            scope=scope,
            target=target,
            content=clean,
            source=source,
            session_id=session_id,
            confidence=0.75 if source == "assistant_outcome" else 0.9,
        )

    def _upsert_fact(self, fact: PresentStateFact) -> None:
        with self._file_lock():
            state = self._load_unlocked()
            if fact.scope == "session":
                sessions = state.setdefault("sessions", {})
                session = sessions.setdefault(fact.session_id or self._session_id, {"facts": []})
                facts = self._upsert_in_list(session.get("facts", []), fact)
                session["facts"] = self._trim_facts(facts, self.max_session_facts)
            else:
                facts = self._upsert_in_list(state.get("profile_facts", []), fact)
                state["profile_facts"] = self._trim_facts(facts, self.max_profile_facts)
            state["active_session_id"] = fact.session_id or self._session_id
            state["updated_at"] = _now()
            self._write_unlocked(state)

    def _upsert_in_list(
        self,
        raw_facts: List[Dict[str, Any]],
        fact: PresentStateFact,
    ) -> List[Dict[str, Any]]:
        facts = self._coerce_facts(raw_facts)
        normalized = _normalize_content(fact.content)
        now = _now()
        for existing in facts:
            if existing.id == fact.id or _normalize_content(existing.content) == normalized:
                existing.content = fact.content
                existing.source = fact.source
                existing.updated_at = now
                existing.session_id = fact.session_id or existing.session_id
                existing.metadata.update(fact.metadata)
                return [asdict(item) for item in facts]
        facts.append(fact)
        return [asdict(item) for item in facts]

    def _merge_fact_dicts(
        self,
        base: List[Dict[str, Any]],
        extra: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        facts = list(base)
        seen = {
            _normalize_content(str(item.get("content") or ""))
            for item in facts
            if isinstance(item, dict)
        }
        for item in extra:
            if not isinstance(item, dict):
                continue
            key = _normalize_content(str(item.get("content") or ""))
            if key and key not in seen:
                facts.append(dict(item))
                seen.add(key)
        return facts

    def _trim_facts(self, raw_facts: List[Dict[str, Any]], max_facts: int) -> List[Dict[str, Any]]:
        facts = self._coerce_facts(raw_facts)
        facts.sort(key=lambda item: item.updated_at, reverse=True)
        if max_facts > 0:
            facts = facts[:max_facts]
        return [asdict(item) for item in facts]

    def _session_facts_from_state(self, state: Dict[str, Any], session_id: str) -> List[PresentStateFact]:
        sessions = state.get("sessions", {})
        if not isinstance(sessions, dict) or not session_id:
            return []
        session = sessions.get(session_id, {})
        if not isinstance(session, dict):
            return []
        return self._coerce_facts(session.get("facts", []))

    def _coerce_facts(self, raw_facts: Any) -> List[PresentStateFact]:
        facts = []
        if not isinstance(raw_facts, list):
            return facts
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            content = _clean_fact_text(str(item.get("content") or ""), self.max_fact_chars)
            if not content:
                continue
            facts.append(
                PresentStateFact(
                    id=str(item.get("id") or _fact_id(
                        str(item.get("scope") or "profile"),
                        str(item.get("target") or "memory"),
                        content,
                        str(item.get("session_id") or ""),
                    )),
                    scope=str(item.get("scope") or "profile"),
                    target=str(item.get("target") or "memory"),
                    content=content,
                    source=str(item.get("source") or "unknown"),
                    session_id=str(item.get("session_id") or ""),
                    created_at=float(item.get("created_at") or _now()),
                    updated_at=float(item.get("updated_at") or item.get("created_at") or _now()),
                    confidence=float(item.get("confidence") or 1.0),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        facts.sort(key=lambda item: item.updated_at, reverse=True)
        return facts

    def _default_state(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "active_session_id": self._session_id,
            "profile_facts": [],
            "sessions": {},
            "updated_at": _now(),
        }

    def _load(self) -> Dict[str, Any]:
        with self._file_lock():
            return self._load_unlocked()

    def _load_unlocked(self) -> Dict[str, Any]:
        path = get_present_state_path()
        if not path.exists():
            return self._default_state()
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to read present-state memory: %s", exc)
            return self._default_state()
        if not isinstance(data, dict):
            return self._default_state()
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("profile_facts", [])
        data.setdefault("sessions", {})
        return data

    def _write_unlocked(self, state: Dict[str, Any]) -> None:
        path = get_present_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".present_state_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @contextmanager
    def _file_lock(self):
        path = get_present_state_path()
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_EX)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            handle.close()


__all__ = ["PresentStateFact", "PresentStateMemory", "get_present_state_path"]
