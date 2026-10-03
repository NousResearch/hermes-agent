"""Lifecycle contract: session start/ingest/reset/end, tools, and diagnostics."""

from __future__ import annotations

import json

from agent.context_engine import ContextEngine
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


def _engine(tmp_path, **overrides) -> LCMEngine:
    settings = {"db_path": str(tmp_path / "lcm.db")}
    settings.update(overrides)
    return LCMEngine(config=LCMConfig(settings))


def _tool(engine: LCMEngine, name: str, **args):
    return json.loads(engine.handle_tool_call(name, args))


def test_engine_satisfies_the_context_engine_abc(tmp_path):
    engine = _engine(tmp_path)
    assert isinstance(engine, ContextEngine)
    assert engine.name == "lcm"
    for method in ("compress", "should_compress", "update_from_response"):
        assert callable(getattr(engine, method))
    assert engine.is_available() is True


def test_session_lifecycle_opens_ingests_and_closes(tmp_path):
    engine = _engine(tmp_path)
    assert engine._store is None

    engine.on_session_start("s1")
    assert engine._store is not None and engine._store.is_open
    assert engine._session_id == "s1"

    engine.on_turn_complete(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}],
        session_id="s1",
    )
    assert engine._store.stats("s1")["messages"] == 2

    engine.on_session_end("s1", [{"role": "user", "content": "third"}])
    assert engine._store is not None
    assert engine._store.is_open is False, "session end closes the store"
    # A later read reopens the same durable store (the record survives the close).
    assert engine._store.stats("s1")["messages"] == 3, "the final turn is flushed"


def test_on_session_reset_clears_counters_but_keeps_data(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    engine.on_turn_complete([{"role": "user", "content": "kept"}], session_id="s1")
    engine.compression_count = 3
    engine.last_prompt_tokens = 123

    engine.on_session_reset()

    assert engine.compression_count == 0
    assert engine.last_prompt_tokens == 0
    assert engine._last_node_id is None
    assert engine._store.stats("s1")["messages"] == 1, "reset does not drop history"


def test_token_accounting_and_should_compress(tmp_path):
    engine = _engine(tmp_path)
    engine.update_model("m", 10_000)
    engine.update_from_response(
        {"prompt_tokens": 9_000, "completion_tokens": 10, "total_tokens": 9_010}
    )
    assert engine.threshold_tokens == int(10_000 * engine.threshold_percent)
    assert engine.last_total_tokens == 9_010
    assert engine.should_compress() is True
    assert engine.should_compress(prompt_tokens=1) is False

    # Canonical buckets are accepted too (newer hosts send them).
    engine.update_from_response({"input_tokens": 10, "output_tokens": 5})
    assert engine.last_prompt_tokens == 10
    assert engine.last_total_tokens == 15


def test_engine_tools_are_declared_and_dispatched(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")

    names = {schema["name"] for schema in engine.get_tool_schemas()}
    assert names == {"lcm_search", "lcm_expand", "lcm_page", "lcm_status"}
    assert all("description" in schema for schema in engine.get_tool_schemas())

    assert _tool(engine, "lcm_status")["session_id"] == "s1"
    unknown = _tool(engine, "lcm_not_a_tool")
    assert "error" in unknown


def test_tool_errors_are_json_not_exceptions(tmp_path):
    engine = LCMEngine(config=LCMConfig({"db_path": str(tmp_path / "lcm.db")}))
    raw = engine.handle_tool_call("lcm_search", {"query": "x"})
    assert isinstance(raw, str)
    assert "error" in json.loads(raw)


def test_ingest_failure_does_not_break_the_turn(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")

    class Boom:
        is_open = True

        def __getattr__(self, name):
            raise RuntimeError("boom")

    engine._store = Boom()
    engine.on_turn_complete([{"role": "user", "content": "x"}], session_id="s1")
    assert engine._ingest_failures == 1


def test_diagnostics_expose_store_shape_without_leaking_content(tmp_path):
    engine = _engine(tmp_path)
    engine.on_session_start("s1")
    engine.on_turn_complete([{"role": "user", "content": "secret-ish"}], session_id="s1")

    diagnostics = engine.diagnostics()
    assert diagnostics["engine"] == "lcm"
    assert diagnostics["store"]["stats"]["messages"] == 1
    assert diagnostics["store"]["bounds"]["page_size"] >= 1
    assert "secret-ish" not in json.dumps(diagnostics)

    status = engine.get_status()
    assert status["engine"] == "lcm"
    assert status["recall_policy"]["bounded"] is True


def test_ingest_can_be_disabled_by_config(tmp_path):
    engine = _engine(tmp_path, ingest_on_turn_complete=False)
    engine.on_session_start("s1")
    engine.on_turn_complete([{"role": "user", "content": "x"}], session_id="s1")
    assert engine._store.stats("s1")["messages"] == 0
