"""Rollback and fail-safe behaviour (H05, part 2).

The acceptance criterion is: "plugin disabled or failed startup leaves the
built-in Hermes context path usable".  Every route to that outcome is asserted
here — unset config, an unknown engine name, a plugin that raises on import,
and a plugin that fails mid-compaction.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from plugins.context_engine import load_context_engine
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


def _agent_with_engine_config(cfg: dict):
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
        patch("agent.context_compressor.get_model_context_length", return_value=128_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_unknown_engine_name_loads_nothing():
    assert load_context_engine("definitely-not-an-engine") is None


def test_engine_directory_missing_returns_none():
    assert load_context_engine("") is None
    assert load_context_engine("..") is None


def test_a_plugin_that_raises_on_import_falls_back_to_the_builtin(tmp_path, monkeypatch):
    """A broken third-party engine must not take the context path down with it."""
    import plugins.context_engine as ce

    broken = tmp_path / "engines" / "broken"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text(
        "raise ImportError('engine exploded at import time')\n", encoding="utf-8"
    )
    monkeypatch.setattr(ce, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path / "engines")

    assert load_context_engine("broken") is None, "a raising engine is not returned"
    assert ce.discover_context_engines()[0][2] is False, "and it reports unavailable"


def test_config_unset_uses_the_builtin_compressor():
    agent = _agent_with_engine_config({"agent": {}})
    assert isinstance(agent.context_compressor, ContextCompressor)
    assert agent.context_compressor.name == "compressor"


def test_config_explicit_compressor_uses_the_builtin():
    agent = _agent_with_engine_config({"context": {"engine": "compressor"}, "agent": {}})
    assert isinstance(agent.context_compressor, ContextCompressor)


def test_config_naming_a_missing_engine_falls_back_to_the_builtin():
    agent = _agent_with_engine_config({"context": {"engine": "no-such-engine"}, "agent": {}})
    assert isinstance(agent.context_compressor, ContextCompressor), (
        "an unavailable plugin leaves the built-in context path usable"
    )
    assert not hasattr(agent.context_compressor, "lcm_search")


def test_mid_compaction_failure_leaves_the_message_list_untouched(tmp_path):
    engine = LCMEngine(config=LCMConfig({"db_path": str(tmp_path / "lcm.db")}))
    engine.on_session_start("s1")
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": f"head {i}"} for i in range(3)]
    messages += [{"role": "assistant", "content": f"body {i}"} for i in range(8)]
    messages += [{"role": "user", "content": f"tail {i}"} for i in range(6)]

    class Boom:
        is_open = True

        def __getattr__(self, name):
            raise RuntimeError("disk went away")

    engine._store = Boom()
    assert engine.compress(messages) == messages


def test_failed_engine_start_does_not_raise(tmp_path, monkeypatch):
    engine = LCMEngine(config=LCMConfig({"db_path": str(tmp_path / "lcm.db")}))

    def _explode():
        raise OSError("cannot create database")

    monkeypatch.setattr(engine, "_ensure_store", _explode)
    engine.on_session_start("s1")  # must not raise
    assert engine.handle_tool_call("lcm_status", {}).startswith("{")


def test_disabling_the_plugin_keeps_the_session_usable(tmp_path):
    """With the engine disabled, sessions still work through the built-in path."""
    agent = _agent_with_engine_config({"context": {"engine": "compressor"}})
    assert agent.compression_enabled is not None
    assert hasattr(agent.context_compressor, "compress")
    assert agent.context_compressor.get_status()["context_length"] > 0
