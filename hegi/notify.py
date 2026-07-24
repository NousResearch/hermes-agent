"""Checkpointed Telegram reporting through Hermes's existing sender."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

from .models import MeetingMinutes
from .state import StateStore


def load_env_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return ""


def _bullets(items: list[str], limit: int = 8) -> str:
    return "\n".join(f"- {item}" for item in items[:limit]) or "- 없음 또는 미확정"


def telegram_parts(minutes: MeetingMinutes) -> list[str]:
    meeting = minutes.meeting_id
    report_kind = (
        "운영 장애 기록"
        if minutes.meeting_type == "operational_incident"
        else "회의록"
    )
    positions = [
        f"{item.agent}: {item.position}" for item in minutes.agent_positions
    ]
    activities = [
        f"{item.agent}: {item.activity} → {item.result}"
        for item in minutes.agent_activity_log
    ]
    actions = [
        f"[{item.priority}] {item.title} · 담당 {item.owner or '미정'} · 기한 {item.deadline or '미정'}"
        for item in minutes.action_items
    ]
    memory = minutes.memory_evaluation
    memory_text = (
        f"추천: {memory.recommendation}\n"
        f"검색 결과\n{_bullets(memory.search_findings, 6)}\n"
        f"중복 대상\n{_bullets(memory.duplicate_targets, 6)}\n"
        f"신규성 근거\n{_bullets(memory.novelty_basis, 6)}\n"
        f"판정 이유\n{_bullets(memory.reasons, 6)}"
        if memory
        else "Memory Evaluation 미실행"
    )
    return [
        f"1/4 HEGI {report_kind} 개요 · {meeting}\n\n{minutes.title}\n"
        f"유형: {minutes.meeting_type}\n\n"
        f"핵심 의제\n{_bullets(minutes.agenda)}\n\n"
        f"교수 판단\n{_bullets(minutes.professor_positions)}",
        f"2/4 논의 전개·연구적 의견·행동 로그 · {meeting}\n\n"
        f"{_bullets([stage.summary for stage in minutes.discussion_flow])}\n\n"
        f"에이전트별 연구적 의견\n{_bullets(positions)}\n\n"
        f"에이전트 행동 로그\n{_bullets(activities)}",
        f"3/4 합의·이견·Action Items · {meeting}\n\n"
        f"합의\n{_bullets(minutes.agreements)}\n\n"
        f"이견\n{_bullets(minutes.disagreements)}\n\n"
        f"Action Items\n{_bullets(actions)}",
        f"4/4 Memory Evaluation · {meeting}\n\n{memory_text}\n\n"
        "@헤기 기억해\n@헤기 초안 만들어\n@헤기 기존 기억에 합쳐\n@헤기 기억하지 마\n\n"
        "자동 Commit은 수행하지 않습니다.",
    ]


class TelegramReporter:
    def __init__(
        self,
        state: StateStore,
        *,
        token: str,
        chat_id: str,
        sender: Callable[..., Any] | None = None,
        attempts: int = 3,
    ):
        self.state = state
        self.token = token
        self.chat_id = chat_id
        self.sender = sender
        self.attempts = attempts

    def send(self, minutes: MeetingMinutes, *, dry_run: bool = False) -> list[str]:
        sent_ids: list[str] = []
        delivered = self.state.delivered_parts(minutes.meeting_id)
        for index, content in enumerate(telegram_parts(minutes), start=1):
            if index in delivered:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if dry_run:
                sent_ids.append(f"dry-run-{index}")
                continue
            if not self.token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN이 없습니다.")
            error: Exception | None = None
            for attempt in range(self.attempts):
                try:
                    result = self._send(content)
                    message_id = str(
                        result.get("message_id")
                        if isinstance(result, dict)
                        else getattr(result, "message_id", "")
                    )
                    self.state.record_delivery(
                        minutes.meeting_id,
                        index,
                        digest,
                        status="sent",
                        platform_message_id=message_id,
                    )
                    sent_ids.append(message_id)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt + 1 < self.attempts:
                        time.sleep(min(2**attempt, 8))
            if error is not None:
                self.state.record_delivery(
                    minutes.meeting_id,
                    index,
                    digest,
                    status="failed",
                    error=str(error),
                )
                self.state.add_dead_letter(
                    "telegram",
                    {"part_index": index, "content": content},
                    str(error),
                    minutes.meeting_id,
                )
                raise error
        return sent_ids

    def _send(self, content: str) -> Any:
        sender = self.sender
        if sender is None:
            from tools.send_message_tool import _send_telegram

            sender = _send_telegram
        result = sender(
            self.token,
            self.chat_id,
            content,
            disable_link_previews=True,
        )
        return asyncio.run(result) if hasattr(result, "__await__") else result
