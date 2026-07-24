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
        return "\n".join(f"- {value}" for value in values) if values else "- 없음 또는 미확정"

    positions = "\n\n".join(
        f"### {position.agent}\n{position.position}\n\n"
        f"근거: {', '.join(map(str, position.source_message_ids)) or '미기재'}"
        for position in minutes.agent_positions
    ) or "### 에이전트 기여\n- 분석 결과 없음"
    flow = "\n\n".join(
        f"### {stage.heading}\n{stage.summary}\n\n"
        f"근거: {', '.join(map(str, stage.source_message_ids)) or '미기재'}"
        for stage in minutes.discussion_flow
    ) or "- 구조화 분석 결과 없음"
    actions = "\n".join(
        f"- [{item.priority}] {item.title} (담당: {item.owner or '미정'}, "
        f"기한: {item.deadline or '미정'}, 근거: {item.source_message_ids})"
        for item in minutes.action_items
    ) or "- 없음"
    memory = minutes.memory_evaluation
    memory_text = (
        f"- 추천: {memory.recommendation}\n"
        f"- 검색 쿼리: {memory.searched_queries}\n"
        f"- 이유: {'; '.join(memory.reasons)}"
        if memory
        else "- Memory Evaluation 미실행"
    )
    started = datetime.fromtimestamp(episode.started_at).astimezone().isoformat()
    ended = datetime.fromtimestamp(episode.ended_at).astimezone().isoformat()
    return f"""# 회의록

## 회의 정보
- 회의명: {minutes.title}
- 일시: {started} ~ {ended}
- 참석: {", ".join(episode.participants)}
- 메시지 수: {len(episode.messages)}
- 분석 범위: source message {episode.source_message_ids}
- Meeting ID: {episode.meeting_id}

## 1. 논의 배경
{minutes.background}

## 2. 핵심 의제
{bullets(minutes.agenda)}

## 3. 논의 전개
{flow}

## 4. 교수의 핵심 판단
{bullets(minutes.professor_positions)}

## 5. 에이전트별 기여
{positions}

## 6. 합의된 사항
{bullets(minutes.agreements)}

## 7. 논쟁 및 이견
{bullets(minutes.disagreements)}

## 8. 남은 문제
{bullets(minutes.unresolved_questions)}

## 9. 새로 정의된 개념
{bullets([f"{item.name}: {item.definition} ({item.status})" for item in minutes.new_concepts])}

## 10. 근거·자료·서지 확인
{bullets([f"{item.claim} — {item.source or '출처 미기재'} [{item.verification}]" for item in minutes.evidence_and_sources])}

## 11. 연구 방향
{bullets(minutes.research_direction)}

## 12. Action Items
{actions}

## 13. Memory Evaluation
{memory_text}

## 14. 헤기 권고
{minutes.recommendation or "교수 검토 필요"}

---
meeting_id: {episode.meeting_id}
episode_hash: {episode.episode_hash}
source_message_ids: {episode.source_message_ids}
source_session_ids: {episode.source_session_ids}
model: {minutes.metadata.get("model", "")}
prompt_version: {minutes.metadata.get("prompt_version", "")}
generated_at: {minutes.metadata.get("generated_at", "")}
hegi_version: {minutes.metadata.get("hegi_version", "2.0.0")}
"""


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
