"""Agent config-load failures must be visible without losing defaults (#125919)."""

import builtins
import logging
from unittest.mock import patch

import pytest


@pytest.fixture
def make_agent(monkeypatch):
    from hermes_constants import get_hermes_home
    import run_agent

    home = get_hermes_home()
    (home / "config.yaml").write_text("", encoding="utf-8")
    (home / "memories" / "MEMORY.md").write_text("The notebook is blue.", encoding="utf-8")
    # run_agent may have been imported before this test's home was selected.
    monkeypatch.setattr(run_agent, "_hermes_home", home)
    agents = []

    def create():
        agent = run_agent.AIAgent(
            model="test-model",
            provider="openrouter",
            api_mode="chat_completions",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            skip_background_review=True,
            save_trajectories=False,
        )
        agents.append(agent)
        return agent

    with (
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata_http.get") as metadata_get,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("tools.env_probe.warm_environment_probe_async"),
    ):
        # The prewarm thread and compressor keep imported metadata aliases; stop
        # their shared HTTP boundary while retaining the real metadata resolution.
        metadata_get.return_value.json.return_value = {
            "data": [{"id": "test-model", "context_length": 204_800}],
        }
        try:
            yield create
        finally:
            for agent in reversed(agents):
                agent.close()


@pytest.mark.parametrize("failure_at", ["import", "load"])
@pytest.mark.parametrize("reader", ["cache", "reasoning", "sections", "all"])
def test_config_failure_warns_without_exposing_values_and_preserves_defaults(
    make_agent, monkeypatch, caplog, failure_at, reader,
):
    from hermes_cli import config
    from hermes_constants import get_hermes_home
    import sys

    home = get_hermes_home()
    configured = (
        "prompt_caching:\n  cache_ttl: 1h\n"
        "model:\n  reasoning_echo: true\n"
        "compression:\n  enabled: false\n"
        "skills:\n  creation_nudge_interval: 99\n"
    )
    config_path = home / "config.yaml"
    config_path.write_text(configured, encoding="utf-8")
    reader_functions = {
        "cache": "_init_prompt_cache_config",
        "reasoning": "_read_reasoning_echo_from_config",
        "sections": "init_agent",
    }
    affected = set(reader_functions) if reader == "all" else {reader}
    targets = {reader_functions[name] for name in affected}
    secret = "private-config-value-do-not-log"
    error_type = ModuleNotFoundError if failure_at == "import" else RuntimeError
    error = error_type(f"Could not load config.yaml: api_key: {secret}")
    intercepted = set()

    with monkeypatch.context() as failure:
        if failure_at == "import":
            real_import = builtins.__import__

            def import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
                caller = sys._getframe(1).f_code.co_name
                if (
                    name == "hermes_cli.config"
                    and "load_config_readonly" in fromlist
                    and (globals or {}).get("__name__")
                    in {"agent.agent_init", "agent.reasoning_params"}
                    and caller in targets
                ):
                    intercepted.add(caller)
                    raise error
                return real_import(name, globals, locals, fromlist, level)

            failure.setattr(builtins, "__import__", import_with_failure)
        else:
            real_load = config.load_config_readonly

            def load_with_failure(*args, **kwargs):
                caller = sys._getframe(1).f_code.co_name
                if caller in targets:
                    intercepted.add(caller)
                    raise error
                return real_load(*args, **kwargs)

            failure.setattr(config, "load_config_readonly", load_with_failure)

        # The failing object is the first construction, so an early diagnostic
        # must reach the real file handler, not one initialized by a reference.
        with caplog.at_level(logging.WARNING, logger="run_agent"):
            fallback = make_agent()

    assert intercepted == targets
    assert config_path.read_text(encoding="utf-8-sig") == configured
    warnings = [
        record for record in caplog.records
        if record.name == "run_agent" and record.levelno >= logging.WARNING
        and "config" in record.getMessage().lower()
        and "default" in record.getMessage().lower()
    ]
    assert len(warnings) == len(affected)
    expected_sections = {
        "cache": "prompt_caching.cache_ttl",
        "reasoning": "model.reasoning_echo",
        "sections": "memory",
    }
    messages = [record.getMessage() for record in warnings]
    persisted = (home / "logs" / "errors.log").read_text(encoding="utf-8-sig")
    for scope in affected:
        assert any(expected_sections[scope] in message for message in messages)
        assert expected_sections[scope] in persisted
    assert all(error_type.__name__ in message for message in messages)
    assert secret not in caplog.text
    assert secret not in persisted
    assert all(secret not in logging.Formatter().format(record) for record in warnings)

    caplog.clear()
    healthy = make_agent()
    config_path.write_text("", encoding="utf-8")
    defaults = make_agent()
    assert healthy._cache_ttl != defaults._cache_ttl
    assert healthy._reasoning_echo_flag is not defaults._reasoning_echo_flag
    assert healthy.compression_enabled is not defaults.compression_enabled
    assert healthy._skill_nudge_interval != defaults._skill_nudge_interval
    for scope, attrs in {
        "cache": ("_cache_ttl", "_cache_disabled", "_use_prompt_caching"),
        "reasoning": ("_reasoning_echo_flag",),
        "sections": ("compression_enabled", "_skill_nudge_interval"),
    }.items():
        reference = defaults if scope in affected else healthy
        for attr in attrs:
            assert getattr(fallback, attr) == getattr(reference, attr), attr
    assert fallback._memory_manager is healthy._memory_manager is defaults._memory_manager is None
    for attr in (
        "_memory_enabled", "_user_profile_enabled", "_memory_nudge_interval",
        "compression_in_place", "max_compression_attempts",
    ):
        assert getattr(fallback, attr) == getattr(healthy, attr) == getattr(defaults, attr), attr
    assert (
        fallback.context_compressor.threshold_percent
        == healthy.context_compressor.threshold_percent
        == defaults.context_compressor.threshold_percent
    )
    assert fallback._memory_store.memory_entries == healthy._memory_store.memory_entries
    assert fallback._memory_store.memory_entries == defaults._memory_store.memory_entries
    assert not any(
        record.name == "run_agent" and record.levelno >= logging.WARNING
        and "config" in record.getMessage().lower()
        and "default" in record.getMessage().lower()
        for record in caplog.records
    )


def test_empty_config_initializes_memory_without_a_load_warning(make_agent, caplog):
    from hermes_constants import get_hermes_home

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent = make_agent()

    assert agent._memory_manager is None
    assert agent._memory_store is not None
    assert agent._memory_store.memory_entries == ["The notebook is blue."]
    assert (get_hermes_home() / "config.yaml").read_bytes() == b""
    assert not any(
        record.name == "run_agent" and record.levelno >= logging.WARNING
        and "config" in record.getMessage().lower()
        and "default" in record.getMessage().lower()
        for record in caplog.records
    )
