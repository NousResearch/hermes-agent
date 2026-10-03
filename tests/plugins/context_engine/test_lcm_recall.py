"""Recall contract: bounded pages, session scope, and honest truncation."""

from __future__ import annotations

import json

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm import recall


def _engine(tmp_path, **overrides) -> LCMEngine:
    settings = {
        "db_path": str(tmp_path / "lcm.db"),
        "page_size": 3,
        "max_page_size": 5,
        "max_search_results": 2,
        "body_chars": 400,
    }
    settings.update(overrides)
    engine = LCMEngine(config=LCMConfig(settings))
    engine.on_session_start("s1")
    return engine


def _tool(engine: LCMEngine, name: str, **args):
    return json.loads(engine.handle_tool_call(name, args))


def test_search_is_session_scoped_and_bounded(tmp_path):
    engine = _engine(tmp_path)
    engine.on_turn_complete(
        [
            {"role": "user", "content": "alpha ledger entry one"},
            {"role": "assistant", "content": "alpha ledger entry two"},
            {"role": "user", "content": "alpha ledger entry three"},
        ],
        session_id="s1",
    )
    engine.on_turn_complete([{"role": "user", "content": "alpha other session"}], session_id="s2")

    payload = _tool(engine, "lcm_search", query="alpha ledger")
    assert payload["bounded"] is True
    assert payload["returned"] == 2, "capped by max_search_results"
    assert payload["has_more"] is True
    assert all("alpha ledger" in hit["body"] for hit in payload["results"])
    assert "other session" not in json.dumps(payload)


def test_expand_pages_a_node_and_reports_more(tmp_path):
    engine = _engine(tmp_path)
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": f"head {i}"} for i in range(3)]
    messages += [{"role": "assistant", "content": f"body {i}"} for i in range(8)]
    messages += [{"role": "user", "content": f"tail {i}"} for i in range(6)]
    engine.compress(messages)
    node_id = engine._last_node_id

    first = _tool(engine, "lcm_expand", node_id=node_id, page=0)
    assert first["node"]["source_count"] == 8
    assert first["returned"] == 3, "page_size bounds the lineage page"
    assert first["total"] == 8
    assert first["has_more"] is True
    assert first["next_page"] == 1

    seen = []
    page = 0
    while True:
        chunk = _tool(engine, "lcm_expand", node_id=node_id, page=page)
        seen.extend(item["body"] for item in chunk["lineage"] if item["kind"] == "message")
        if not chunk["has_more"]:
            break
        page = chunk["next_page"]

    assert seen == [f"body {i}" for i in range(8)], "every source is recoverable in order"


def test_expand_page_size_request_is_clamped(tmp_path):
    engine = _engine(tmp_path, page_size=3, max_page_size=5)
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": f"head {i}"} for i in range(3)]
    messages += [{"role": "assistant", "content": f"body {i}"} for i in range(8)]
    messages += [{"role": "user", "content": f"tail {i}"} for i in range(6)]
    engine.compress(messages)

    payload = _tool(engine, "lcm_expand", node_id=engine._last_node_id, page_size=10_000)
    assert payload["page_size"] == 5
    assert payload["returned"] <= 5


def test_expand_single_message_truncates_honestly(tmp_path):
    engine = _engine(tmp_path, body_chars=200)
    long_body = "padded " + "y" * 5_000
    engine.on_turn_complete([{"role": "user", "content": long_body}], session_id="s1")
    message_id = _tool(engine, "lcm_search", query="padded")["results"][0]["message_id"]

    payload = _tool(engine, "lcm_expand", message_id=message_id)
    assert payload["message"]["truncated"] is True
    assert payload["body_chars"] == 200
    assert len(payload["message"]["body"]) == 200

    # A larger cap is honored up to the hard maximum, then clamped.
    bigger = _tool(engine, "lcm_expand", message_id=message_id, max_chars=10_000)
    assert bigger["body_chars"] <= recall.HARD_MAX_BODY_CHARS


def test_expand_rejects_unknown_and_ambiguous_refs(tmp_path):
    engine = _engine(tmp_path)
    engine.on_turn_complete([{"role": "user", "content": "hello"}], session_id="s1")

    assert "error" in _tool(engine, "lcm_expand", node_id="n-missing")
    assert "error" in _tool(engine, "lcm_expand", message_id="m-missing")
    assert "error" in _tool(engine, "lcm_expand", node_id="n-a", message_id="m-b")
    assert "error" in _tool(engine, "lcm_expand")


def test_page_walks_the_session_and_never_returns_it_whole(tmp_path):
    engine = _engine(tmp_path, page_size=3, max_page_size=5)
    engine.on_turn_complete(
        [{"role": "user", "content": f"row {i}"} for i in range(11)], session_id="s1"
    )

    payload = _tool(engine, "lcm_page")
    assert payload["returned"] == 3
    assert payload["total"] == 11
    assert payload["has_more"] is True

    oversized = _tool(engine, "lcm_page", page_size=100)
    assert oversized["page_size"] == 5, "the ceiling wins over the request"
    assert oversized["total"] == 11

    cursor = payload["next_cursor"]
    seen = payload["returned"]
    while cursor:
        chunk = _tool(engine, "lcm_page", cursor=cursor)
        seen += chunk["returned"]
        cursor = chunk["next_cursor"] if chunk["has_more"] else None
    assert seen == 11


def test_status_reports_policy_and_bounds(tmp_path):
    engine = _engine(tmp_path)
    payload = _tool(engine, "lcm_status")
    assert payload["session_id"] == "s1"
    assert payload["stats"]["schema_version"] >= 1
    assert payload["policy"]["unbounded_transcript_loads"] is False
    assert payload["policy"]["scope"] == recall.SCOPE_SESSION
    assert payload["bounds"]["max_page_size"] == 5


def test_recall_requires_an_open_store(tmp_path):
    engine = LCMEngine(config=LCMConfig({"db_path": str(tmp_path / "lcm.db")}))
    assert "error" in _tool(engine, "lcm_search", query="anything")
