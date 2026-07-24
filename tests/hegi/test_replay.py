from __future__ import annotations

import json
import os

from hegi.cli import main
from hegi.models import MeetingMinutes
from hegi.state import StateStore


def test_replay_uses_checkpointed_telegram_delivery(tmp_path, monkeypatch, capsys):
    runtime_home = tmp_path / "memory-curator"
    config_dir = runtime_home / "hegi"
    config_dir.mkdir(parents=True)
    env_path = runtime_home / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=fake-token\n", encoding="utf-8")
    state_path = config_dir / "state.db"
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        f"""
enabled: true
state_db: "{state_path}"
telegram:
  chat_id: "-1001"
  curator_env: "{env_path}"
  enabled: true
agents: []
archive:
  local_spool: "{tmp_path / 'archive'}"
memory:
  auto_commit: false
  auto_draft: false
  require_professor_approval: true
  professor_user_ids: ["42"]
  default_project: research
""",
        encoding="utf-8",
    )
    minutes = MeetingMinutes(
        meeting_id="meeting",
        title="회의",
        background="배경",
        agenda=[],
        discussion_flow=[],
        agent_positions=[],
        professor_positions=[],
        agreements=[],
        disagreements=[],
        unresolved_questions=[],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=[],
        action_items=[],
        memory_evaluation=None,
        confidence=0.8,
        warnings=[],
    )
    state = StateStore(state_path)
    state.save_episode("meeting", "hash", {"meeting_id": "meeting"}, "reported")
    state.update_episode("meeting", status="reported", minutes=minutes.to_dict())
    sends: list[str] = []

    monkeypatch.setattr(
        "tools.send_message_tool._send_telegram",
        lambda _token, _chat, text, **_kwargs: (
            sends.append(text) or {"message_id": str(len(sends))}
        ),
    )
    monkeypatch.delenv("HERMES_HOME", raising=False)

    assert main(
        [
            "--config",
            str(config_path),
            "replay",
            "--meeting-id",
            "meeting",
            "--send",
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    assert len(first["message_ids"]) == 4
    assert len(sends) == 4
    assert os.environ["HERMES_HOME"] == str(runtime_home)

    assert main(
        [
            "--config",
            str(config_path),
            "replay",
            "--meeting-id",
            "meeting",
            "--send",
        ]
    ) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["message_ids"] == []
    assert len(sends) == 4
