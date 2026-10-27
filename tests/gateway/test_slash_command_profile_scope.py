"""Gateway slash commands must do their blocking work inside the routed profile.

The multiplexed inbound handler wraps the whole message in
``_profile_runtime_scope``, which installs the routed profile's ``HERMES_HOME``
override and its secret scope as **contextvars**. A bare
``loop.run_in_executor(None, ...)`` starts the worker with an EMPTY context, so
``SessionDB()`` / ``get_hermes_home()`` inside the worker resolve the LAUNCH
home — /insights reported the default profile's conversations from another
profile's chat. ``/compress`` already routes through
``_run_in_executor_with_context``; every other hop in the mixin must too.

Drives the real mixin methods and the real ``_profile_runtime_scope``: the
contextvar loss is a property of the hop, so mocking the hop away would test
nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "coder"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return home


@pytest.fixture
def runner():
    """Minimal host exposing the mixin plus the runner's executor helpers."""
    from gateway.run import GatewayRunner
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Runner(GatewaySlashCommandsMixin):
        _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context
        _get_executor = GatewayRunner._get_executor

    r = _Runner()
    r.adapters = {}
    r._pending_skills_reload_notes = {}
    return r


class _Event:
    def __init__(self, args: str = ""):
        self._args = args
        self.source = None

    def get_command_args(self) -> str:
        return self._args


@pytest.mark.asyncio
async def test_insights_opens_session_db_under_the_routed_home(
    runner, profile_home, monkeypatch
):
    import agent.insights as insights_mod
    import hermes_state
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home

    seen: dict = {}

    class _RecordingDB:
        def __init__(self, *a, **kw):
            seen["home"] = str(get_hermes_home())

        def close(self):
            pass

    class _Engine:
        def __init__(self, db):
            pass

        def generate(self, **kw):
            return {}

        def format_gateway(self, report):
            return "ok"

    monkeypatch.setattr(hermes_state, "SessionDB", _RecordingDB)
    monkeypatch.setattr(insights_mod, "InsightsEngine", _Engine)

    with _profile_runtime_scope(profile_home):
        result = await runner._handle_insights_command(_Event(""))

    assert result == "ok"
    assert seen["home"] == str(profile_home)
