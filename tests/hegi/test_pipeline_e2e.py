from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from hegi.config import load_config
from hegi.llm import HermesLLMClient
from hegi.pipeline import HegiPipeline


def _source_db(path: Path, assistant_id: int, agent_text: str) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, chat_id TEXT, chat_type TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, platform_message_id TEXT, active INTEGER, compacted INTEGER
        );
        INSERT INTO sessions VALUES ('s', '-1001', 'group');
        INSERT INTO messages VALUES
            (1, 's', 'user', '이 논의를 정리해', 100, 'u-1', 1, 0);
        """
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, 's', 'assistant', ?, ?, ?, 1, 0)",
        (assistant_id, agent_text, 100 + assistant_id, f"a-{assistant_id}"),
    )
    connection.commit()
    connection.close()


class FakeMemory:
    def search(self, query: str, limit: int = 5):
        return {"results": []}

    def create_draft(self, arguments):
        raise AssertionError("pipeline must not create a draft")


def test_full_v2_pipeline_archives_reports_and_consumes_buffer(tmp_path, monkeypatch):
    sources = []
    for index, (name, text) in enumerate(
        [("헤헤", "첫 분석"), ("헤코", "반론과 보완"), ("헤클", "종합 의견")],
        start=2,
    ):
        path = tmp_path / f"{name}.db"
        _source_db(path, index, text)
        sources.append((name, path))
    env_path = tmp_path / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=fake-test-token\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    agent_yaml = "\n".join(
        f'  - name: "{name}"\n    db_path: "{path}"' for name, path in sources
    )
    config_path.write_text(
        f"""
enabled: true
state_db: "{tmp_path / 'state.db'}"
telegram:
  chat_id: "-1001"
  curator_env: "{env_path}"
  enabled: true
agents:
{agent_yaml}
episode:
  quiet_minutes: 1
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
""",
        encoding="utf-8",
    )
    payload = {
        "title": "공진 연구회의",
        "background": "교수가 정리를 요청함",
        "agenda": ["개념 정리"],
        "discussion_flow": [
            {"heading": "분석", "summary": "세 관점을 비교함", "source_message_ids": [2, 3, 4]}
        ],
        "agent_positions": [
            {
                "agent": "헤헤",
                "position": "첫 분석",
                "contributions": [],
                "source_message_ids": [2],
            }
        ],
        "professor_positions": ["논의 정리 요청"],
        "agreements": ["후속 검토가 필요함"],
        "disagreements": [],
        "unresolved_questions": ["개념 범위"],
        "new_concepts": [],
        "evidence_and_sources": [],
        "research_direction": ["개념 범위를 보완함"],
        "action_items": [
            {
                "title": "개념 범위 보완",
                "description": "비교 내용을 정리한다",
                "source_message_ids": [3],
                "owner": None,
                "deadline": None,
                "priority": "high",
                "project_id": None,
                "rationale": "반론 발언",
            }
        ],
        "confidence": 0.8,
        "warnings": [],
        "recommendation": "교수 검토",
    }

    def fake_llm(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload, ensure_ascii=False)
                    )
                )
            ]
        )

    sends: list[str] = []

    def fake_send(_token, _chat_id, content, **_kwargs):
        sends.append(content)
        return {"message_id": len(sends)}

    pipeline = HegiPipeline(
        load_config(config_path),
        llm_client=HermesLLMClient(call=fake_llm),
        memory_backend=FakeMemory(),
        telegram_sender=fake_send,
    )
    result = pipeline.run_once(dry_run=False, now=1000)

    assert result[0]["status"] == "reported"
    assert len(sends) == 4
    assert list((tmp_path / "archive").rglob("*.md"))
    assert list((tmp_path / "archive").rglob("*.json"))
    with pipeline.state.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM message_buffer WHERE consumed=0"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM action_items"
        ).fetchone()[0] == 1
    assert pipeline.run_once(dry_run=False, now=1100) == []
