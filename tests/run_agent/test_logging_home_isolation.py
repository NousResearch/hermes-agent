"""Regression tests: agent logging must follow the LIVE ``HERMES_HOME``.

``run_agent._hermes_home`` is resolved once at import time. Under pytest,
test modules import ``run_agent`` during collection — BEFORE the per-test
conftest fixture redirects ``HERMES_HOME`` — so that frozen path points at
the developer's real Hermes home. ``agent_init`` used to pass it to
``setup_logging()``, attaching root-logger file handlers to the real
``~/.hermes/logs/agent.log`` (``%LOCALAPPDATA%\\hermes\\logs\\agent.log``
on native Windows). Every mock-provider test in the process then appended
records to the live install's log — which reads like the agent silently
calling real provider APIs during a test run.

The fix is twofold:
  * ``agent_init`` calls ``setup_logging()`` with no override, so the log
    directory is resolved from the environment at call time.
  * ``tests/conftest.py`` sandboxes ``HERMES_HOME`` at import time as
    defense-in-depth for other import-time consumers.
"""

import os
from pathlib import Path
from unittest.mock import patch

import run_agent
from run_agent import AIAgent


def _queued_file_handler_paths() -> set:
    from hermes_logging import rotating_file_handlers

    return {
        Path(h.baseFilename).resolve()
        for h in rotating_file_handlers()
        if getattr(h, "baseFilename", None)
    }


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def test_agent_logging_ignores_import_time_frozen_home(tmp_path, monkeypatch):
    """setup_logging() must use the current HERMES_HOME, not the value
    ``run_agent._hermes_home`` froze at import time."""
    stale_home = tmp_path / "stale_import_time_home"
    stale_home.mkdir()
    live_home = Path(os.environ["HERMES_HOME"])  # per-test sandbox

    # Simulate the original failure mode: run_agent was imported while
    # HERMES_HOME pointed somewhere else (the real install).
    monkeypatch.setattr(run_agent, "_hermes_home", stale_home)

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    paths = _queued_file_handler_paths()
    live_agent_log = (live_home / "logs" / "agent.log").resolve()
    stale_resolved = stale_home.resolve()

    assert live_agent_log in paths, (
        f"agent.log handler not attached under live HERMES_HOME; "
        f"handlers: {sorted(str(p) for p in paths)}"
    )
    offenders = [p for p in paths if stale_resolved in p.parents]
    assert not offenders, (
        "log handlers attached under the import-time frozen home "
        f"instead of the live HERMES_HOME: {offenders}"
    )


def test_logging_queue_does_not_retain_previous_test_home(tmp_path, monkeypatch):
    import hermes_logging

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(first))
    hermes_logging.setup_logging()
    assert all(first.resolve() in p.parents for p in _queued_file_handler_paths())

    # This is the same lifecycle boundary used by the per-test fixture.
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False
    monkeypatch.setenv("HERMES_HOME", str(second))
    hermes_logging.setup_logging()

    paths = _queued_file_handler_paths()
    assert paths
    assert all(second.resolve() in p.parents for p in paths)
    assert not any(first.resolve() in p.parents for p in paths)


def test_conftest_sandboxes_hermes_home_at_import_time():
    """The import-time HERMES_HOME sandbox in tests/conftest.py must keep
    module-level ``get_hermes_home()`` callers away from the real install."""
    from hermes_constants import _get_platform_default_hermes_home

    real_home = _get_platform_default_hermes_home().resolve()
    frozen = Path(run_agent._hermes_home).resolve()
    assert frozen != real_home and real_home not in frozen.parents, (
        f"run_agent._hermes_home froze the REAL Hermes home ({frozen}); "
        "tests/conftest.py must export a sandbox HERMES_HOME before any "
        "test module is imported"
    )
