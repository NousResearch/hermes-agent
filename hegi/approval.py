"""Durable Telegram approval processing for professor-authorized STM drafts."""

from __future__ import annotations

import json
from typing import Any, Callable

from .config import HegiConfig
from .memory import DraftGate, MCPMemoryBackend, MemoryEvaluator
from .notify import load_env_value
from .pipeline import minutes_from_dict
from .state import StateStore


def _backend(config: HegiConfig) -> MCPMemoryBackend:
    memory = config.section("memory")
    return MCPMemoryBackend(
        read_server=str(memory.get("read_server", "memory-forest-read")),
        search_tool=str(memory.get("search_tool", "")),
        draft_server=str(memory.get("draft_server", "")),
        draft_tool=str(memory.get("draft_tool", "")),
    )


def _send_status(
    config: HegiConfig,
    text: str,
    *,
    sender: Callable[..., Any] | None = None,
) -> None:
    token = load_env_value(config.curator_env, "TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN이 없습니다.")
    if sender is None:
        from tools.send_message_tool import _send_telegram

        sender = _send_telegram
    result = sender(
        token,
        config.chat_id,
        text,
        disable_link_previews=True,
    )
    if hasattr(result, "__await__"):
        import asyncio

        asyncio.run(result)


def process_pending_approvals(
    config: HegiConfig,
    *,
    backend: Any | None = None,
    sender: Callable[..., Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim queued approvals exactly once and create draft-only Memory records."""
    state = StateStore(config.state_db)
    memory_backend = backend or _backend(config)
    memory = config.section("memory")
    gate = DraftGate(
        state,
        memory_backend,
        professor_user_ids=[
            str(item) for item in memory.get("professor_user_ids", [])
        ],
    )
    results: list[dict[str, Any]] = []
    for _ in range(limit):
        job = state.claim_approval_job()
        if job is None:
            break
        meeting_id = str(job["meeting_id"])
        try:
            row = state.episode_by_id(meeting_id)
            if row is None or not row.get("minutes_json"):
                raise ValueError("승인 대상 회의록이 없습니다.")
            minutes = minutes_from_dict(json.loads(row["minutes_json"]))
            evaluation = MemoryEvaluator(memory_backend).evaluate(minutes)
            draft = gate.create_draft_after_recheck(
                minutes,
                evaluation,
                project=str(job["project"]),
            )
            result = {
                "meeting_id": meeting_id,
                "status": "draft_created",
                "project": str(job["project"]),
                "draft": draft,
                "commit": "not_performed",
            }
            state.complete_approval_job(int(job["id"]), status="completed", result=result)
            results.append(result)
            try:
                _send_status(
                    config,
                    "✅ HEGI STM Draft 생성 완료\n\n"
                    f"회의: {meeting_id}\n"
                    f"프로젝트: {job['project']}\n"
                    "Memory Forest 재검색: 완료\n"
                    "Commit: 수행하지 않음",
                    sender=sender,
                )
            except Exception as notify_exc:
                state.add_dead_letter(
                    "approval_notification",
                    {"job_id": int(job["id"])},
                    str(notify_exc),
                    meeting_id,
                )
        except Exception as exc:
            state.complete_approval_job(
                int(job["id"]), status="failed", error=str(exc)
            )
            state.add_dead_letter(
                "approval",
                {
                    "job_id": int(job["id"]),
                    "platform_message_id": str(job["platform_message_id"]),
                },
                str(exc),
                meeting_id,
            )
            try:
                _send_status(
                    config,
                    "⚠️ HEGI Draft 생성 실패\n\n"
                    f"회의: {meeting_id}\n"
                    f"오류: {exc}\n"
                    "Commit은 수행하지 않았습니다.",
                    sender=sender,
                )
            except Exception:
                pass
            results.append(
                {
                    "meeting_id": meeting_id,
                    "status": "failed",
                    "error": str(exc),
                    "commit": "not_performed",
                }
            )
    return results
