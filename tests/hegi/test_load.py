from __future__ import annotations

from hegi.episode import EpisodeDetector
from hegi.models import SourceMessage


def test_detector_handles_ten_thousand_messages():
    messages = [
        SourceMessage(
            source_agent=("헤헤" if index % 2 else "헤코"),
            source_db=f"/{index % 2}.db",
            message_id=index,
            session_id=f"s-{index % 2}",
            platform_message_id=str(index),
            chat_id="-1",
            chat_type="group",
            role="assistant",
            content=f"발언 {index}",
            timestamp=float(index),
        )
        for index in range(1, 10001)
    ]
    episodes = EpisodeDetector(
        quiet_minutes=1,
        max_gap_minutes=30,
        minimum_agents=2,
        minimum_messages=4,
        maximum_messages=10000,
    ).detect(messages, now=20000)
    assert len(episodes) == 1
    assert len(episodes[0].messages) == 10000
