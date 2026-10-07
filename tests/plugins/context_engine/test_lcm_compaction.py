"""Compaction contract: lineage, recoverability, and output validity.

The acceptance criterion is exact: "raw messages remain recoverable through
bounded pages and every summary node retains source lineage".  These tests
assert the relationship between the compacted list and the store, not the
wording of the digest.
"""

from __future__ import annotations

from plugins.context_engine.lcm import compaction
from plugins.context_engine.lcm.compaction import (
    LCM_NODE_METADATA_KEY,
    LCM_SUMMARY_METADATA_KEY,
    choose_summary_role,
    plan_compaction,
)
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


def _engine(tmp_path, **overrides) -> LCMEngine:
    settings = {"db_path": str(tmp_path / "lcm.db"), "page_size": 3, "max_page_size": 5}
    settings.update(overrides)
    return LCMEngine(config=LCMConfig(settings))


def _transcript(head: int = 3, middle: int = 8, tail: int = 6) -> list[dict]:
    messages = [{"role": "system", "content": "system prompt"}]
    messages += [{"role": "user", "content": f"head {index}"} for index in range(head)]
    messages += [
        {"role": "assistant", "content": f"middle {index}"} for index in range(middle)
    ]
    messages += [{"role": "user", "content": f"tail {index}"} for index in range(tail)]
    return messages


def test_plan_protects_system_head_and_tail():
    messages = _transcript()
    plan = plan_compaction(messages, protect_first_n=3, protect_last_n=6)
    assert plan == (4, 12)

    assert plan_compaction(messages[:5], protect_first_n=3, protect_last_n=6) is None
    assert plan_compaction([], protect_first_n=3, protect_last_n=6) is None


def test_summary_role_alternates_for_the_common_shapes():
    user_tail = [{"role": "user", "content": "t"}]
    assistant_tail = [{"role": "assistant", "content": "t"}]
    assert choose_summary_role([{"role": "user", "content": "h"}], user_tail) == "assistant"
    assert (
        choose_summary_role([{"role": "assistant", "content": "h"}], assistant_tail)
        == "user"
    )
    # Tool/system rows are invisible to a template, so they must not decide it.
    assert (
        choose_summary_role(
            [{"role": "assistant", "content": "h"}, {"role": "tool", "content": "x"}],
            user_tail,
        )
        == "assistant",
    )


def test_compress_reports_the_node_and_shrinks_the_list(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    messages = _transcript()

    compacted = engine.compress(messages)

    assert len(compacted) < len(messages)
    assert compacted[0] is messages[0], "the protected head is reused verbatim"
    assert compacted[-1] is messages[-1], "the protected tail is reused verbatim"

    marker = compacted[4]
    assert marker[LCM_SUMMARY_METADATA_KEY] is True
    node_id = marker[LCM_NODE_METADATA_KEY]
    assert node_id
    assert node_id in marker["content"], "the digest cites its own node id"
    assert engine._last_node_id == node_id
    assert engine.compression_count == 1


def test_every_summary_node_retains_source_lineage(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    messages = _transcript()
    middle = messages[4:12]

    engine.compress(messages)
    store = engine._store
    node = store.node(engine._last_node_id)

    assert node["source_count"] == len(middle)
    sources = store.node_sources(node["node_id"])
    assert len(sources) == len(middle)
    assert {kind for kind, _sid in sources} == {"message"}

    # The lineage resolves to the *actual* original messages, in order.
    bodies = [store.message("s1", sid)["body"] for _kind, sid in sources]
    assert bodies == [message["content"] for message in middle]
    assert store.stats("s1")["lineage_edges"] == len(middle)


def test_raw_messages_stay_recoverable_after_compaction(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    messages = _transcript()
    engine.compress(messages)

    store = engine._store
    total = store.stats("s1")["messages"]
    assert total == len(messages), "no raw message is deleted by compaction"

    recovered = []
    cursor = 0
    while True:
        window = store.page_window("s1", after_seq=cursor, page_size=5)
        recovered.extend(row["seq"] for row in window["items"])
        if not window["has_more"]:
            break
        cursor = window["next_cursor"]
    assert len(recovered) == len(messages), "every raw message is reachable by paging"

    # And the compacted-away detail is still findable by search.
    assert store.search("s1", "middle 5"), "compacted content remains searchable"


def test_second_compaction_descends_the_dag(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    once = engine.compress(_transcript())
    first_node = engine._last_node_id

    twice = engine.compress(once)
    second = engine._last_node_id
    assert second != first_node

    store = engine._store
    sources = store.node_sources(second)
    assert ("node", first_node) in sources, "a re-summarized range points at the node"
    assert store.node(second)["level"] == store.node(first_node)["level"] + 1
    assert len(twice) == len(once), "re-compaction does not grow the transcript"


def test_compress_leaves_short_transcripts_alone(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    messages = _transcript(head=1, middle=0, tail=2)

    assert engine.compress(messages) == messages
    assert engine.compression_count == 0


def test_compress_never_raises_when_storage_fails(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    messages = _transcript()

    class Boom:
        is_open = True

        def __getattr__(self, name):
            raise RuntimeError(f"storage unavailable: {name}")

    engine._store = Boom()

    assert engine.compress(messages) == messages, (
        "a failing engine must leave the context untouched"
    )


def test_digest_is_bounded_and_keeps_the_recovery_pointer(tmp_path):
    messages = [{"role": "user", "content": "x" * 500} for _ in range(40)]
    summary = compaction.build_summary(
        messages, node_id="n-test", level=1, max_chars=600, page_size=5
    )
    assert len(summary) <= 600
    assert 'node_id="n-test"' in summary, "the pointer survives truncation"
    assert "lcm_expand" in summary
