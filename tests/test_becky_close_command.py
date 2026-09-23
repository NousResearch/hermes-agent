from __future__ import annotations

from pathlib import Path

import pytest

from gateway.becky_loops import (
    BeckyLoopsConfig,
    SessionDBBeckyLoopsStore,
    load_becky_loops_config,
)


class _TopicDB:
    def __init__(self) -> None:
        self.messages = {
            "session-general": [],
            "session-system": [],
            "session-loop": [
                {
                    "role": "user",
                    "content": "Please compare the two options.",
                    "timestamp": 1_755_104_400.0,
                }
            ],
        }

    def list_sessions_rich(self, **kwargs: object) -> list[dict[str, object]]:
        del kwargs
        return [
            {
                "id": "session-general",
                "chat_id": "-1004476874933",
                "thread_id": "1",
                "title": "General",
                "preview": "General topic",
                "message_count": 1,
                "started_at": "2026-08-14T12:00:00+00:00",
                "last_active": "2026-08-14T12:01:00+00:00",
                "ended_at": None,
            },
            {
                "id": "session-system",
                "chat_id": "-1004476874933",
                "thread_id": "2",
                "title": "System",
                "preview": "System topic for Hermes commands and status.",
                "message_count": 1,
                "started_at": "2026-08-14T12:00:00+00:00",
                "last_active": "2026-08-14T12:01:00+00:00",
                "ended_at": None,
            },
            {
                "id": "session-loop",
                "chat_id": "-1004476874933",
                "thread_id": "3",
                "title": "Energy audit",
                "preview": "Compare the two options.",
                "message_count": 1,
                "started_at": "2026-08-14T12:00:00+00:00",
                "last_active": "2026-08-14T12:01:00+00:00",
                "ended_at": None,
            },
        ]

    def get_messages(
        self, session_id: str, **kwargs: object
    ) -> list[dict[str, object]]:
        del kwargs
        return list(self.messages[session_id])


def test_store_excludes_general_and_managed_system_topics() -> None:
    store = SessionDBBeckyLoopsStore(_TopicDB(), managed_topic_ids={"2"})

    rows = store.list_topics("-1004476874933")

    assert [row["thread_id"] for row in rows] == ["3"]
    assert all(row["title"] not in {"General", "System"} for row in rows)


def test_config_collects_configured_system_topic_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
gateway:
  becky_loops:
    enabled: true
    telegram_chat_id: -1004476874933
    telegram_topic_id: 2
    proven_topic_control: bot_api_private_topic
platforms:
  telegram:
    extra:
      dm_topics:
        - chat_id: '-1004476874933'
          topics:
            - name: System
              thread_id: 2
            - name: Energy audit
              thread_id: 3
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "bridge-token")
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_CONTROL", "1")

    config = load_becky_loops_config(config_path)

    assert config is not None
    assert config.managed_topic_ids == frozenset({"2"})


class _PaginatedTopicDB(_TopicDB):
    def __init__(self) -> None:
        super().__init__()
        self.offsets: list[int] = []
        self.ended: list[tuple[str, str]] = []

    def list_sessions_rich(self, **kwargs: object) -> list[dict[str, object]]:
        offset = int(kwargs["offset"])
        self.offsets.append(offset)
        if offset == 0:
            return [
                {
                    "id": f"unrelated-{index}",
                    "chat_id": "-1004476874933",
                    "thread_id": str(index + 10),
                    "ended_at": None,
                }
                for index in range(200)
            ]
        if offset == 200:
            return [{
                "id": "session-loop",
                "chat_id": "-1004476874933",
                "thread_id": "3",
                "ended_at": None,
            }]
        return []

    def end_session(self, session_id: str, reason: str) -> None:
        self.ended.append((session_id, reason))


def test_store_end_topic_session_scans_past_first_page() -> None:
    db = _PaginatedTopicDB()
    store = SessionDBBeckyLoopsStore(db)

    assert store.end_topic_session(
        chat_id="-1004476874933", thread_id="3", reason="telegram_topic_closed"
    ) is True
    assert db.offsets == [0, 200]
    assert db.ended == [("session-loop", "telegram_topic_closed")]
