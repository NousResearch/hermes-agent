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


@pytest.mark.parametrize(
    ("failure_at", "error_type"),
    [("import", ModuleNotFoundError), ("load", RuntimeError)],
)
def test_config_failure_warns_without_exposing_values_and_preserves_defaults(
    make_agent, monkeypatch, caplog, failure_at, error_type,
):
    from hermes_cli import config

    reference = make_agent()
    secret = "private-config-value-do-not-log"
    error = error_type(f"Could not load config.yaml: api_key: {secret}")
    caplog.clear()

    with monkeypatch.context() as failure:
        if failure_at == "import":
            real_import = builtins.__import__

            def import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
                if (
                    name == "hermes_cli.config"
                    and (globals or {}).get("__name__") == "agent.agent_init"
                    and fromlist == ("load_config_readonly",)
                ):
                    raise error
                return real_import(name, globals, locals, fromlist, level)

            failure.setattr(builtins, "__import__", import_with_failure)
        else:
            def load_with_failure():
                raise error

            failure.setattr(config, "load_config_readonly", load_with_failure)

        with caplog.at_level(logging.WARNING, logger="run_agent"):
            fallback = make_agent()

    warnings = [
        record for record in caplog.records
        if record.name == "run_agent" and record.levelno >= logging.WARNING
        and "config" in record.getMessage().lower()
        and "default" in record.getMessage().lower()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert error_type.__name__ in message
    assert all(section in message.lower() for section in ("memory", "skills", "compression"))
    # Format the whole record: exc_info can leak values even when getMessage() is safe.
    assert secret not in logging.Formatter().format(warnings[0])

    assert fallback._memory_manager is reference._memory_manager is None
    assert fallback._memory_store is not None
    assert fallback._memory_store.memory_entries == reference._memory_store.memory_entries
    for attr in (
        "_memory_enabled", "_user_profile_enabled", "_memory_nudge_interval",
        "_skill_nudge_interval", "compression_enabled", "compression_in_place",
        "max_compression_attempts",
    ):
        assert getattr(fallback, attr) == getattr(reference, attr), attr
    assert (
        fallback.context_compressor.threshold_percent
        == reference.context_compressor.threshold_percent
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
