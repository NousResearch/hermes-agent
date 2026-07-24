from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from hegi.approval import process_pending_approvals
from hegi.config import load_config
from hegi.llm import HermesLLMClient
from hegi.memory import DraftGate
from hegi.pipeline import HegiPipeline


class FlowMemory:
    def __init__(self):
        self.searches: list[str] = []
        self.drafts: list[dict] = []

    def search(self, query: str, limit: int = 5):
        self.searches.append(query)
        return {"results": []}

    def create_draft(self, arguments):
        self.drafts.append(arguments)
        return {"id": "stm-1", "state": "pending"}


def _agent_db(path: Path, assistant_id: int, timestamp: float, text: str) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, chat_id TEXT, chat_type TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, platform_message_id TEXT, active INTEGER,
            compacted INTEGER
        );
        INSERT INTO sessions VALUES ('s', '-1009', 'group');
        INSERT INTO messages VALUES
        (1, 's', 'user', '교수: 인공자연 개념을 논의합시다', 100, 'prof-1', 1, 0);
        """
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, 's', 'assistant', ?, ?, ?, 1, 0)",
        (assistant_id, text, timestamp, f"agent-{assistant_id}"),
    )
    connection.commit()
    connection.close()


def test_professor_three_agents_quiet_minutes_report_approval_draft_no_commit(
    tmp_path
):
    sources = [
        ("HeHe", 2, 110, "현상학적 배경을 분석함"),
        ("HeCo", 3, 120, "반례와 개념 경계를 검토함"),
        ("HeClaude", 4, 130, "세 관점을 통합해 연구 방향을 제안함"),
    ]
    agent_entries = []
    for name, message_id, timestamp, text in sources:
        path = tmp_path / f"{name}.db"
        _agent_db(path, message_id, timestamp, text)
        agent_entries.append(f'  - name: {name}\n    db_path: "{path}"')
    env_path = tmp_path / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=fake-token\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
enabled: true
state_db: "{tmp_path / 'state.db'}"
telegram:
  chat_id: "-1009"
  curator_env: "{env_path}"
  enabled: true
agents:
{chr(10).join(agent_entries)}
episode:
  quiet_minutes: 10
  max_gap_minutes: 30
  minimum_agents: 2
  minimum_messages: 4
archive:
  local_spool: "{tmp_path / 'archive'}"
memory:
  enabled: true
  auto_commit: false
  auto_draft: false
  require_professor_approval: true
  professor_user_ids: ["9001"]
  default_project: media_aesthetics
""",
        encoding="utf-8",
    )
    minutes_payload = {
        "title": "인공자연 연구회의",
        "background": "교수가 세 에이전트와 개념을 검토함",
        "agenda": ["인공자연 개념의 경계"],
        "discussion_flow": [
            {
                "heading": "세 관점 비교",
                "summary": "배경, 반례, 통합 방향을 검토함",
                "source_message_ids": [2, 3, 4],
            }
        ],
        "agent_positions": [
            {
                "agent": name,
                "position": text,
                "contributions": [],
                "source_message_ids": [message_id],
            }
            for name, message_id, _timestamp, text in sources
        ],
        "professor_positions": ["개념 논의 요청"],
        "agreements": ["인공자연을 매체적 조건으로 다룸"],
        "disagreements": [],
        "unresolved_questions": ["사례 범위"],
        "new_concepts": [],
        "evidence_and_sources": [],
        "research_direction": ["논문 장의 핵심 개념으로 발전"],
        "action_items": [
            {
                "title": "사례 범위 정리",
                "description": "반례를 포함한 사례 목록 작성",
                "source_message_ids": [3],
                "owner": "HeCo",
                "deadline": None,
                "priority": "high",
                "project_id": "media_aesthetics",
                "rationale": "개념 경계 검토",
            }
        ],
        "confidence": 0.9,
        "warnings": [],
        "recommendation": "STM Draft 후보",
    }

    def llm_call(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(minutes_payload, ensure_ascii=False)
                    )
                )
            ]
        )

    sent: list[str] = []
    memory = FlowMemory()
    pipeline = HegiPipeline(
        load_config(config_path),
        llm_client=HermesLLMClient(call=llm_call),
        memory_backend=memory,
        telegram_sender=lambda _token, _chat, text, **_kwargs: {
            "message_id": str(1000 + len(sent) + 1)
        }
        if not sent.append(text)
        else {},
    )

    reports = pipeline.run_once(dry_run=False, now=730)
    assert reports[0]["status"] == "reported"
    assert reports[0]["participants"] == ["HeClaude", "HeCo", "HeHe"]
    assert len(sent) == 4
    assert "Action Items" in sent[2]
    assert "Memory Evaluation" in sent[3]

    meeting_id = reports[0]["meeting_id"]
    gate = DraftGate(pipeline.state, memory, professor_user_ids=["9001"])
    assert gate.approve(
        meeting_id=meeting_id,
        text="기억해",
        user_id="9001",
        platform_message_id="approval-1",
    ) == "remember"
    assert pipeline.state.enqueue_approval_job(
        meeting_id=meeting_id,
        platform_message_id="approval-1",
        project="media_aesthetics",
    )
    approval_results = process_pending_approvals(
        pipeline.config,
        backend=memory,
        sender=lambda _token, _chat, text, **_kwargs: sent.append(text),
    )

    assert approval_results[0]["status"] == "draft_created"
    assert approval_results[0]["commit"] == "not_performed"
    assert len(memory.drafts) == 1
    assert "commit" not in memory.drafts[0]
