"""Memory Forest read evaluation and professor-gated draft creation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Protocol

from .models import MeetingMinutes, MemoryEvaluation, MemoryMatch
from .state import StateStore


class MemoryBackend(Protocol):
    def search(self, query: str, limit: int = 5) -> dict[str, Any]: ...

    def create_draft(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


class MCPMemoryBackend:
    """Use Hermes's existing MCP discovery and tool registry."""

    def __init__(
        self,
        *,
        read_server: str,
        search_tool: str = "",
        draft_server: str = "",
        draft_tool: str = "",
    ):
        self.read_server = read_server
        self.search_name = search_tool or self._name(read_server, "memory_search")
        self.draft_name = (
            draft_tool
            or (
                self._name(draft_server, "memory_create_stm_draft")
                if draft_server
                else ""
            )
        )
        self._discovered = False

    @staticmethod
    def _name(server: str, tool: str) -> str:
        from tools.mcp_tool import mcp_prefixed_tool_name

        return mcp_prefixed_tool_name(server, tool)

    def _ensure(self) -> None:
        if self._discovered:
            return
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()
        self._discovered = True

    @staticmethod
    def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from tools.registry import registry

        raw = registry.dispatch(name, arguments)
        if not isinstance(raw, str):
            raise RuntimeError(f"MCP tool {name} returned unsupported response")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError(f"MCP tool {name} returned non-object JSON")
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        self._ensure()
        return self._dispatch(
            self.search_name, {"query": query, "limit": limit, "include_body": False}
        )

    def create_draft(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.draft_name:
            raise RuntimeError("Memory draft MCP tool이 설정되지 않았습니다.")
        self._ensure()
        return self._dispatch(self.draft_name, arguments)


def memory_queries(minutes: MeetingMinutes) -> list[str]:
    candidates = [minutes.title]
    candidates.extend(concept.name for concept in minutes.new_concepts)
    candidates.extend(minutes.agreements)
    candidates.extend(minutes.research_direction)
    unique: list[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique[:8]


def _extract_result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "matches", "memories", "items"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    nested = payload.get("structuredContent")
    if isinstance(nested, dict):
        return _extract_result_rows(nested)
    return []


class MemoryEvaluator:
    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    def evaluate(self, minutes: MeetingMinutes) -> MemoryEvaluation:
        queries = memory_queries(minutes)
        matches: list[MemoryMatch] = []
        seen: set[str] = set()
        for query in queries:
            payload = self.backend.search(query)
            for row in _extract_result_rows(payload):
                memory_id = str(
                    row.get("id") or row.get("path") or row.get("memory_id") or row.get("title")
                )
                if not memory_id or memory_id in seen:
                    continue
                seen.add(memory_id)
                relation = str(row.get("relation", "부분 중복"))
                if relation not in {"높은 중복", "부분 중복", "낮은 중복", "신규"}:
                    relation = "부분 중복"
                matches.append(
                    MemoryMatch(
                        memory_id=memory_id,
                        title=str(row.get("title") or row.get("path") or memory_id),
                        summary=str(row.get("summary") or row.get("snippet") or ""),
                        relation=relation,  # type: ignore[arg-type]
                        raw=row,
                    )
                )
        has_durable_content = bool(
            minutes.new_concepts or minutes.agreements or minutes.research_direction
        )
        if not has_durable_content:
            recommendation = "no_memory"
            reasons = ["새 개념·합의·연구 방향이 없어 장기 기억 가치가 낮음"]
        elif matches:
            recommendation = "merge_existing"
            reasons = ["Memory Forest 검색에서 관련 기억이 발견되어 병합 검토가 필요함"]
        elif minutes.confidence < 0.6 or minutes.disagreements:
            recommendation = "needs_professor_review"
            reasons = ["의미는 있으나 이견 또는 분석 불확실성이 있어 교수 검토가 필요함"]
        else:
            recommendation = "create_stm_draft"
            reasons = ["새로운 지속성 높은 연구 판단이 있고 기존 관련 기억이 발견되지 않음"]
        return MemoryEvaluation(
            searched_queries=queries,
            matched_memories=matches,
            recommendation=recommendation,  # type: ignore[arg-type]
            candidate_memory_title=minutes.title if has_durable_content else None,
            candidate_memory_summary=(
                " / ".join(minutes.agreements + minutes.research_direction)[:2000]
                if has_durable_content
                else None
            ),
            reasons=reasons,
        )


_COMMANDS = {
    "기억해": "remember",
    "초안 만들어": "draft",
    "기존 기억에 합쳐": "merge",
    "기억하지 마": "reject",
}


def parse_approval_command(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    for phrase, command in _COMMANDS.items():
        if phrase in normalized:
            return command
    return None


class DraftGate:
    def __init__(
        self,
        state: StateStore,
        backend: MemoryBackend,
        *,
        professor_user_ids: list[str],
    ):
        self.state = state
        self.backend = backend
        self.professor_user_ids = {str(user_id) for user_id in professor_user_ids}

    def approve(
        self,
        *,
        meeting_id: str,
        text: str,
        user_id: str,
        platform_message_id: str | None,
    ) -> str:
        command = parse_approval_command(text)
        if command is None:
            raise ValueError("지원하는 헤기 승인 명령이 아닙니다.")
        if not self.professor_user_ids or str(user_id) not in self.professor_user_ids:
            raise PermissionError("설정된 교수 계정만 Memory 승인을 할 수 있습니다.")
        inserted = self.state.record_approval(
            meeting_id, command, str(user_id), text, platform_message_id
        )
        if not inserted:
            raise ValueError("이미 처리한 승인 메시지입니다.")
        return command

    def create_draft_after_recheck(
        self,
        minutes: MeetingMinutes,
        evaluation: MemoryEvaluation,
        *,
        project: str,
    ) -> dict[str, Any]:
        if not evaluation.searched_queries:
            raise PermissionError("Draft 전에 Memory 중복 검색이 필요합니다.")
        row = self.state.episode_by_id(minutes.meeting_id)
        if row is None:
            raise ValueError("meeting_id를 찾을 수 없습니다.")
        with self.state.connect() as connection:
            approval = connection.execute(
                """
                SELECT command FROM approval_events WHERE meeting_id=?
                ORDER BY approved_at DESC LIMIT 1
                """,
                (minutes.meeting_id,),
            ).fetchone()
        if approval is None or approval["command"] not in {"remember", "draft", "merge"}:
            raise PermissionError("교수의 명시적 승인 없이는 Draft를 생성할 수 없습니다.")
        for query in evaluation.searched_queries:
            self.backend.search(query)
        sources = [
            f"message:{message_id}" for item in minutes.action_items for message_id in item.source_message_ids
        ] or [f"meeting:{minutes.meeting_id}"]
        return self.backend.create_draft(
            {
                "project": project,
                "title": evaluation.candidate_memory_title or minutes.title,
                "observed_facts": "\n".join(minutes.agreements) or minutes.background,
                "current_judgment": "\n".join(minutes.research_direction)
                or minutes.recommendation,
                "status": "provisional",
                "unresolved_risks": "\n".join(minutes.unresolved_questions),
                "sources": sources,
                "conversation_id": minutes.meeting_id,
                "rationale": "\n".join(evaluation.reasons),
            }
        )
