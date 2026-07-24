from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hegi.approval import process_pending_approvals
from hegi.config import load_config
from hegi.gateway_plugin import intercept_telegram_approval
from hegi.memory import DraftGate
from hegi.models import MeetingMinutes
from hegi.state import StateStore


class RecordingMemory:
    def __init__(self):
        self.searches: list[str] = []
        self.drafts: list[dict] = []

    def search(self, query: str, limit: int = 5):
        self.searches.append(query)
        return {"results": []}

    def create_draft(self, arguments):
        self.drafts.append(arguments)
        return {"draft_id": "draft-1", "status": "pending"}


def _minutes() -> MeetingMinutes:
    return MeetingMinutes(
        meeting_id="meeting-1",
        title="매체미학 연구회의",
        background="교수와 세 에이전트가 개념을 검토함",
        agenda=["인공자연 개념"],
        discussion_flow=[],
        agent_positions=[],
        professor_positions=["기존 장과 연결"],
        agreements=["인공자연을 매체적 조건으로 정의"],
        disagreements=[],
        unresolved_questions=["사례 범위"],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=["논문 장에 반영"],
        action_items=[],
        memory_evaluation=None,
        confidence=0.9,
        warnings=[],
    )


def _config(tmp_path, monkeypatch):
    home = tmp_path / "runtime"
    hegi = home / "hegi"
    hegi.mkdir(parents=True)
    env_path = home / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=fake-token\n", encoding="utf-8")
    config_path = hegi / "config.yaml"
    config_path.write_text(
        f"""
enabled: true
state_db: "{hegi / 'state.db'}"
telegram:
  chat_id: "-1001"
  curator_env: "{env_path}"
  enabled: true
agents: []
archive:
  local_spool: "{hegi / 'archive'}"
memory:
  enabled: true
  read_server: memory-forest-read
  draft_server: memory-forest-curator-draft
  auto_commit: false
  auto_draft: false
  require_professor_approval: true
  professor_user_ids: ["42"]
  default_project: media_aesthetics
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_config(config_path)
    state = StateStore(config.state_db)
    state.save_episode(
        "meeting-1",
        "hash-1",
        {"meeting_id": "meeting-1"},
        "reported",
    )
    state.update_episode("meeting-1", status="reported", minutes=_minutes().to_dict())
    state.record_delivery(
        "meeting-1",
        4,
        "digest",
        status="sent",
        platform_message_id="400",
    )
    return config, state


def test_approval_job_rechecks_memory_creates_draft_and_never_commits(
    tmp_path, monkeypatch
):
    config, state = _config(tmp_path, monkeypatch)
    backend = RecordingMemory()
    gate = DraftGate(state, backend, professor_user_ids=["42"])
    assert gate.approve(
        meeting_id="meeting-1",
        text="기억해",
        user_id="42",
        platform_message_id="500",
    ) == "remember"
    assert state.enqueue_approval_job(
        meeting_id="meeting-1",
        platform_message_id="500",
        project="media_aesthetics",
    )
    sent: list[str] = []

    result = process_pending_approvals(
        config,
        backend=backend,
        sender=lambda _token, _chat, text, **_kwargs: sent.append(text),
    )

    assert result[0]["status"] == "draft_created"
    assert result[0]["commit"] == "not_performed"
    assert backend.searches
    assert len(backend.drafts) == 1
    assert backend.drafts[0]["status"] == "provisional"
    assert "commit" not in backend.drafts[0]
    assert state.approval_job_counts() == {"completed": 1}
    assert any("Commit: 수행하지 않음" in item for item in sent)


@pytest.mark.asyncio
async def test_gateway_plugin_intercepts_professor_reply_before_curator_agent(
    tmp_path, monkeypatch
):
    _config(tmp_path, monkeypatch)
    processed = []
    monkeypatch.setattr(
        "hegi.gateway_plugin.process_pending_approvals",
        lambda config: processed.append(config.chat_id),
    )
    adapter = SimpleNamespace(send=AsyncMock(return_value={"message_id": "ack"}))
    gateway = SimpleNamespace(_adapter_for_source=lambda _source: adapter)
    event = SimpleNamespace(
        text="초안 만들어",
        message_id="501",
        reply_to_message_id="400",
        source=SimpleNamespace(
            platform=SimpleNamespace(value="telegram"),
            chat_id="-1001",
            user_id="42",
        ),
    )

    decision = intercept_telegram_approval(
        event=event, gateway=gateway, session_store=None
    )
    await asyncio.sleep(0.05)

    assert decision == {"action": "skip", "reason": "hegi-approval-queued"}
    adapter.send.assert_awaited()
    assert processed == ["-1001"]
    state = StateStore(tmp_path / "runtime" / "hegi" / "state.db")
    assert state.approval_job_counts() == {"pending": 1}
