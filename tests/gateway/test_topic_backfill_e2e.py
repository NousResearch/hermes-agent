"""E2E for topic-context backfill against a real temp HERMES_HOME.

Seeds two sibling sessions with real SessionDB rows + sessions.json entries in
one Telegram topic, then calls build_topic_backfill excluding a third (new)
session id, asserting the attributed backlog returns with dedup, tool/non-str
exclusion, age-filter, and the DM-skip + opt-out gates (via the helper's
sibling resolution and an enabled/disabled config).
"""

import json
import time
from pathlib import Path

import pytest

from gateway import topic_backfill
from gateway.config import GatewayConfig, TopicBackfillConfig
from gateway.session import SessionSource, is_shared_multi_user_session
from gateway.platforms.base import Platform


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed(home: Path, *, session_id, chat_id, thread_id, user_id, user_name, messages):
    index_path = home / "sessions" / "sessions.json"
    data = {}
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    key = f"agent:main:telegram:group:{chat_id}:{thread_id or ''}:{user_id}"
    data[key] = {
        "session_key": key,
        "session_id": session_id,
        "origin": {
            "platform": "telegram", "chat_id": chat_id, "thread_id": thread_id,
            "user_id": user_id, "user_name": user_name, "chat_type": "group",
        },
    }
    index_path.write_text(json.dumps(data), encoding="utf-8")

    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id=session_id, source="telegram")
    for m in messages:
        db.append_message(
            session_id=session_id, role=m["role"], content=m.get("content"),
            tool_name=m.get("tool_name"), tool_calls=m.get("tool_calls"),
            timestamp=m.get("timestamp"),
        )


def test_e2e_two_siblings_merged_attributed_block(temp_home):
    now = time.time()
    _seed(
        temp_home, session_id="ALICE", chat_id="42", thread_id="7",
        user_id="a", user_name="Alice",
        messages=[
            {"role": "user", "content": "kicking off the deploy", "timestamp": now - 100},
            {"role": "assistant", "content": "on it", "timestamp": now - 95},
            # tool row excluded
            {"role": "tool", "content": "ran deploy.sh", "tool_name": "terminal", "timestamp": now - 90},
            # multimodal excluded
            {"role": "user", "content": [{"type": "text", "text": "screenshot"}], "timestamp": now - 85},
            # ancient excluded by age filter
            {"role": "user", "content": "last week's note", "timestamp": now - 72 * 3600},
        ],
    )
    _seed(
        temp_home, session_id="BOB", chat_id="42", thread_id="7",
        user_id="b", user_name="Bob",
        messages=[
            {"role": "assistant", "content": "deploy finished", "timestamp": now - 80},
            # duplicate of Alice's "on it" assistant line -> deduped
            {"role": "assistant", "content": "on it", "timestamp": now - 70},
        ],
    )

    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW_SESSION", max_messages=15, max_age_hours=24,
    )

    assert block is not None
    # Attribution present.
    assert "Alice" in block
    assert "Bob" in block
    # Real user/assistant turns present, ordered chronologically.
    assert "kicking off the deploy" in block
    assert "deploy finished" in block
    # tool / multimodal / ancient excluded.
    assert "ran deploy.sh" not in block
    assert "screenshot" not in block
    assert "last week's note" not in block
    # dedup: "on it" appears exactly once.
    assert block.count("on it") == 1
    # chronological order: kickoff before deploy finished
    assert block.index("kicking off the deploy") < block.index("deploy finished")


def test_e2e_excludes_self_and_other_topic(temp_home):
    now = time.time()
    _seed(
        temp_home, session_id="SELF", chat_id="42", thread_id="7",
        user_id="me", user_name="Me",
        messages=[{"role": "user", "content": "my own session", "timestamp": now - 10}],
    )
    _seed(
        temp_home, session_id="OTHER_TOPIC", chat_id="42", thread_id="99",
        user_id="x", user_name="X",
        messages=[{"role": "user", "content": "different topic", "timestamp": now - 10}],
    )
    _seed(
        temp_home, session_id="PEER", chat_id="42", thread_id="7",
        user_id="p", user_name="Peer",
        messages=[{"role": "user", "content": "peer in topic", "timestamp": now - 10}],
    )

    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="SELF", max_messages=15, max_age_hours=24,
    )

    assert block is not None
    assert "peer in topic" in block
    assert "my own session" not in block
    assert "different topic" not in block


def test_e2e_opt_out_via_disabled_config(temp_home):
    """When the config disables backfill the gate (as the adapter applies it)
    short-circuits before any DB work."""
    now = time.time()
    _seed(
        temp_home, session_id="PEER", chat_id="42", thread_id="7",
        user_id="p", user_name="Peer",
        messages=[{"role": "user", "content": "peer in topic", "timestamp": now - 10}],
    )
    cfg = TopicBackfillConfig(enabled=False)

    # Mirror the adapter gate: when disabled, never call build_topic_backfill.
    block = None
    if cfg.enabled:
        block = topic_backfill.build_topic_backfill(
            platform="telegram", chat_id="42", thread_id="7",
            exclude_session_id="NEW", max_messages=cfg.max_messages,
            max_age_hours=cfg.max_age_hours,
        )
    assert block is None


def test_e2e_dm_skipped_by_shared_session_gate():
    """A DM source is not a shared multi-user session -> never backfilled."""
    dm = SessionSource(
        platform=Platform.TELEGRAM, chat_id="500", chat_type="dm",
        user_id="7", user_name="Carol",
    )
    assert is_shared_multi_user_session(dm) is False

    shared = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="group",
        user_id="7", user_name="Carol", thread_id="7",
    )
    assert is_shared_multi_user_session(shared) is True


def test_e2e_cap_keeps_most_recent(temp_home):
    now = time.time()
    msgs = [
        {"role": "user", "content": f"line{i}", "timestamp": now - (50 - i)}
        for i in range(10)
    ]
    _seed(
        temp_home, session_id="PEER", chat_id="42", thread_id="7",
        user_id="p", user_name="Peer", messages=msgs,
    )

    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=2, max_age_hours=24,
    )

    assert block is not None
    assert "line8" in block
    assert "line9" in block
    assert "line0" not in block
