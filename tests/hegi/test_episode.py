from __future__ import annotations

from hegi.episode import EpisodeDetector, episode_hash
from hegi.models import SourceMessage


def _message(index: int, agent: str, role: str, timestamp: float, content: str = "본문"):
    return SourceMessage(
        source_agent=agent,
        source_db=f"/{agent}.db",
        message_id=index,
        session_id=f"s-{agent}",
        platform_message_id=str(index),
        chat_id="-1001",
        chat_type="group",
        role=role,
        content=content,
        timestamp=timestamp,
    )


def test_episode_requires_two_assistant_agents_and_quiet_period():
    messages = [
        _message(1, "헤헤", "user", 100),
        _message(2, "헤헤", "assistant", 110),
        _message(3, "헤코", "assistant", 120),
        _message(4, "헤클", "assistant", 130),
    ]
    detector = EpisodeDetector(quiet_minutes=10)
    episode = detector.detect(messages, now=731)[0]
    assert episode.status == "quiet"
    assert episode.participants == ["헤코", "헤클", "헤헤"]


def test_episode_splits_on_gap_and_professor_end_expression():
    detector = EpisodeDetector(
        quiet_minutes=1, max_gap_minutes=30, minimum_messages=1, minimum_agents=1
    )
    messages = [
        _message(1, "헤헤", "assistant", 0),
        _message(2, "헤헤", "user", 1, "여기까지."),
        _message(3, "헤코", "assistant", 2),
        _message(4, "헤클", "assistant", 4000),
    ]
    groups = detector.split(messages)
    assert [len(group) for group in groups] == [2, 1, 1]


def test_hash_is_stable_and_order_sensitive():
    messages = [
        _message(1, "헤헤", "assistant", 100),
        _message(2, "헤코", "assistant", 101),
    ]
    assert episode_hash(messages) == episode_hash(list(messages))
    assert episode_hash(messages) != episode_hash(list(reversed(messages)))
