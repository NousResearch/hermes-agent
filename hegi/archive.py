"""Atomic Markdown/JSON archive with revision and NAS spool semantics."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import MeetingEpisode, MeetingMinutes, as_jsonable
from .quality import enforce_quality_gate


def _slug(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣]+", "-", text).strip("-").lower()
    return value[:60] or "meeting"


def _revision_path(path: Path) -> Path:
    if not path.exists():
        return path
    revision = 2
    while True:
        candidate = path.with_name(f"{path.stem}.r{revision}{path.suffix}")
        if not candidate.exists():
            return candidate
        revision += 1


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_markdown(minutes: MeetingMinutes, episode: MeetingEpisode) -> str:
    def bullets(values: list[str]) -> str:
        safe = [value.strip() for value in values if isinstance(value, str) and value.strip()]
        return "\n".join(f"- {value}" for value in safe) if safe else "- 없음 또는 미확정"

    def ids(values: list[int]) -> str:
        return ", ".join(map(str, values)) or "미기재"

    positions = "\n\n".join(
        f"### {position.agent}\n{position.position}\n\n"
        f"연구적 기여:\n{bullets(position.contributions)}\n\n"
        f"근거: {ids(position.source_message_ids)}"
        for position in minutes.agent_positions
    ) or "### 에이전트 기여\n- 분석 결과 없음"
    activities = "\n".join(
        f"- {item.agent}: {item.activity} → {item.result} "
        f"(근거: {ids(item.source_message_ids)})"
        for item in minutes.agent_activity_log
    ) or "- 기록된 행동 없음"
    flow = "\n\n".join(
        f"### {stage.heading}\n{stage.summary}\n\n"
        f"근거: {ids(stage.source_message_ids)}"
        for stage in minutes.discussion_flow
    ) or "- 구조화 분석 결과 없음"
    actions = "\n".join(
        f"- [{item.priority}] {item.title} (담당: {item.owner or '미정'}, "
        f"기한: {item.deadline or '미정'}, 근거: {ids(item.source_message_ids)})"
        for item in minutes.action_items
    ) or "- 없음"
    conflicts = "\n".join(
        f"- {item.subject}: 이전 `{item.earlier_state}` → 현재 `{item.current_state}` "
        f"[{item.resolution_status}] (근거: {ids(item.source_message_ids)})"
        for item in minutes.temporal_conflicts
    ) or "- 확인된 시간상 상태 충돌 없음"
    memory = minutes.memory_evaluation
    memory_text = (
        f"- 추천: {memory.recommendation}\n"
        f"- 검색 쿼리:\n{bullets(memory.searched_queries)}\n"
        f"- 검색 결과:\n{bullets(memory.search_findings)}\n"
        f"- 중복 대상:\n{bullets(memory.duplicate_targets)}\n"
        f"- 신규성 근거:\n{bullets(memory.novelty_basis)}\n"
        f"- 판정 이유:\n{bullets(memory.reasons)}"
        if memory
        else "- Memory Evaluation 미실행"
    )
    started = datetime.fromtimestamp(episode.started_at).astimezone().isoformat()
    ended = datetime.fromtimestamp(episode.ended_at).astimezone().isoformat()
    metadata = f"""---
meeting_id: {episode.meeting_id}
episode_hash: {episode.episode_hash}
source_message_ids: {ids(episode.source_message_ids)}
source_session_ids: {", ".join(episode.source_session_ids) or "없음"}
model: {minutes.metadata.get("model", "")}
prompt_version: {minutes.metadata.get("prompt_version", "")}
generated_at: {minutes.metadata.get("generated_at", "")}
hegi_version: {minutes.metadata.get("hegi_version", "2.0.2")}
"""
    if minutes.meeting_type == "operational_incident":
        rendered = f"""# 운영 장애 기록

## 장애 정보
- 기록명: {minutes.title}
- 유형: {minutes.meeting_type}
- 일시: {started} ~ {ended}
- 참석: {", ".join(episode.participants)}
- 메시지 수: {len(episode.messages)}
- 분석 범위: source message {ids(episode.source_message_ids)}
- Meeting ID: {episode.meeting_id}

## 1. 장애 개요와 영향
{minutes.background}

## 2. 증상·점검 항목
{bullets(minutes.agenda)}

## 3. 장애 전개와 복구 과정
{flow}

## 4. 교수의 운영 판단
{bullets(minutes.professor_positions)}

## 5. 에이전트 행동 로그
{activities}

## 6. 연구적 의견과 해석
{positions}

## 7. 시간 순서와 해결 상태
{conflicts}

## 8. 현재 해결 상태
{bullets(minutes.agreements)}

## 9. 미해결 장애·위험
{bullets(minutes.unresolved_questions)}

## 10. 복구·예방 Action Items
{actions}

## 11. Memory Evaluation
{memory_text}

## 12. 헤기 운영 권고
{minutes.recommendation or "교수 검토 필요"}

{metadata}"""
    else:
        rendered = f"""# 회의록

## 회의 정보
- 회의명: {minutes.title}
- 유형: {minutes.meeting_type}
- 일시: {started} ~ {ended}
- 참석: {", ".join(episode.participants)}
- 메시지 수: {len(episode.messages)}
- 분석 범위: source message {ids(episode.source_message_ids)}
- Meeting ID: {episode.meeting_id}

## 1. 논의 배경
{minutes.background}

## 2. 핵심 의제
{bullets(minutes.agenda)}

## 3. 논의 전개
{flow}

## 4. 교수의 핵심 판단
{bullets(minutes.professor_positions)}

## 5. 에이전트별 연구적 의견
{positions}

## 6. 에이전트 행동 로그
{activities}

## 7. 합의된 사항
{bullets(minutes.agreements)}

## 8. 논쟁 및 이견
{bullets(minutes.disagreements)}

## 9. 시간 순서와 해결 상태
{conflicts}

## 10. 남은 문제
{bullets(minutes.unresolved_questions)}

## 11. 새로 정의된 개념
{bullets([f"{item.name}: {item.definition} ({item.status})" for item in minutes.new_concepts])}

## 12. 근거·자료·서지 확인
{bullets([f"{item.claim} — {item.source or '출처 미기재'} [{item.verification}]" for item in minutes.evidence_and_sources])}

## 13. 연구 방향
{bullets(minutes.research_direction)}

## 14. Action Items
{actions}

## 15. Memory Evaluation
{memory_text}

## 16. 헤기 권고
{minutes.recommendation or "교수 검토 필요"}

{metadata}"""
    enforce_quality_gate(minutes, episode, rendered=rendered)
    return rendered


class ArchiveManager:
    def __init__(self, local_spool: Path, nas_root: Path | None = None):
        self.local_spool = local_spool
        self.nas_root = nas_root

    def archive(
        self, minutes: MeetingMinutes, episode: MeetingEpisode
    ) -> dict[str, str | None]:
        moment = datetime.fromtimestamp(episode.started_at).astimezone()
        relative = Path("MeetingMinutes") / f"{moment:%Y}" / f"{moment:%m}"
        stem = f"{moment:%Y-%m-%d_%H%M}_{_slug(minutes.title)}_{episode.meeting_id}"
        directory = self.local_spool / relative
        markdown_path = _revision_path(directory / f"{stem}.md")
        json_path = _revision_path(directory / f"{stem}.json")
        payload: dict[str, Any] = {
            "metadata": minutes.metadata,
            "episode": as_jsonable(episode),
            "minutes": as_jsonable(minutes),
        }
        _atomic_write(markdown_path, render_markdown(minutes, episode))
        _atomic_write(
            json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        nas_markdown: Path | None = None
        nas_json: Path | None = None
        if self.nas_root and self.nas_root.is_dir():
            nas_directory = self.nas_root / relative
            nas_directory.mkdir(parents=True, exist_ok=True)
            nas_markdown = _revision_path(nas_directory / markdown_path.name)
            nas_json = _revision_path(nas_directory / json_path.name)
            shutil.copy2(markdown_path, nas_markdown)
            shutil.copy2(json_path, nas_json)
        return {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "nas_markdown": str(nas_markdown) if nas_markdown else None,
            "nas_json": str(nas_json) if nas_json else None,
        }
