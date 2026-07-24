from __future__ import annotations

import json
from types import SimpleNamespace

from hegi.analyzer import build_minutes
from hegi.llm import HermesLLMClient, chunk_messages
from hegi.models import MeetingEpisode, SourceMessage


def _episode() -> MeetingEpisode:
    messages = [
        SourceMessage(
            source_agent=agent,
            source_db=f"/{agent}.db",
            message_id=index,
            session_id=f"s-{agent}",
            platform_message_id=str(index),
            chat_id="-1001",
            chat_type="group",
            role="assistant",
            content="가" * 20,
            timestamp=float(index),
        )
        for index, agent in enumerate(("헤헤", "헤코", "헤클"), start=1)
    ]
    return MeetingEpisode(
        meeting_id="meeting",
        chat_id="-1001",
        started_at=1,
        ended_at=3,
        participants=["헤헤", "헤코", "헤클"],
        messages=messages,
        episode_hash="hash",
        status="quiet",
    )


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_structured_output_repairs_invalid_json_once():
    calls = iter([_response("```json\n{broken}\n```"), _response('{"title":"복구"}')])

    def fake_call(**_kwargs):
        return next(calls)

    client = HermesLLMClient(call=fake_call)
    assert client.structured("system", "user") == {"title": "복구"}


def test_chunker_never_splits_message_content():
    episode = _episode()
    chunks = chunk_messages(episode, max_chars=80)
    assert len(chunks) == 3
    assert all("가" * 20 in chunk for chunk in chunks)


def test_build_minutes_drops_untraced_action_and_preserves_nulls():
    episode = _episode()
    payload = {
        "title": "회의",
        "background": "배경",
        "agenda": ["의제"],
        "discussion_flow": [
            {"heading": "전개", "summary": "요약", "source_message_ids": [1]}
        ],
        "agent_positions": [],
        "professor_positions": [],
        "agreements": [],
        "disagreements": [],
        "unresolved_questions": [],
        "new_concepts": [],
        "evidence_and_sources": [],
        "research_direction": [],
        "action_items": [
            {
                "title": "근거 있음",
                "description": "실행",
                "source_message_ids": [2],
                "owner": None,
                "deadline": None,
                "priority": "high",
                "rationale": "발언 2",
            },
            {"title": "근거 없음", "source_message_ids": [999]},
        ],
        "confidence": 0.8,
    }
    minutes = build_minutes(payload, episode)
    assert [item.title for item in minutes.action_items] == ["근거 있음"]
    assert minutes.action_items[0].owner is None
    assert minutes.action_items[0].deadline is None
    assert any("근거가 없는" in warning for warning in minutes.warnings)
