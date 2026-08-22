"""Gateway slash commands must do their blocking work inside the routed profile.

The multiplexed inbound handler wraps the whole message in
``_profile_runtime_scope``, which installs the routed profile's ``HERMES_HOME``
override and its secret scope as **contextvars**.

``GatewaySlashCommandsMixin`` already knows a bare executor hop drops them — it
routes ``/compress`` through ``_run_in_executor_with_context`` and says so at
the call site. Three siblings in the same file still used
``loop.run_in_executor(None, ...)``, which starts the worker with an EMPTY
context:

* ``/insights`` — ``SessionDB()`` with no explicit path resolves
  ``get_hermes_home()`` at call time, so it read the DEFAULT profile's
  ``state.db`` and reported another profile's conversations.
* ``/debug`` — collects that home's logs/config and uploads them to a public
  paste, so it published the wrong profile's diagnostics.
* ``/goal draft`` — calls the auxiliary LLM, whose credential resolution reads
  the profile secret scope.

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

    return _Runner()


class _Event:
    def __init__(self, args: str = ""):
        self._args = args

    def get_command_args(self) -> str:
        return self._args


class TestInsightsReadsTheRoutedProfilesDatabase:
    @pytest.mark.asyncio
    async def test_session_db_opens_under_the_profile_home(
        self, runner, profile_home, monkeypatch
    ):
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
        import agent.insights as insights_mod

        monkeypatch.setattr(insights_mod, "InsightsEngine", _Engine)

        with _profile_runtime_scope(profile_home):
            result = await runner._handle_insights_command(_Event(""))

        assert result == "ok"
        assert seen["home"] == str(profile_home)

    @pytest.mark.asyncio
    async def test_without_a_scope_it_still_uses_the_launch_home(
        self, runner, profile_home, tmp_path, monkeypatch
    ):
        """Single-profile gateways never enter the scope — behaviour unchanged."""
        import hermes_state
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
        import agent.insights as insights_mod

        monkeypatch.setattr(insights_mod, "InsightsEngine", _Engine)

        await runner._handle_insights_command(_Event(""))

        assert seen["home"] == str(tmp_path / ".hermes")


class TestExecutorHelperContract:
    """The mechanism all three call sites now rely on.

    ``/insights`` is driven end-to-end above. ``/debug`` and ``/goal draft``
    take the identical one-line substitution but sit behind adapter and
    goal-manager scaffolding, so their shared guarantee is pinned here rather
    than through a fake deep enough to stop testing the real thing.
    """

    @pytest.mark.asyncio
    async def test_helper_preserves_the_override_a_bare_hop_drops(
        self, runner, profile_home
    ):
        import asyncio

        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home

        def _probe():
            return str(get_hermes_home())

        loop = asyncio.get_running_loop()
        with _profile_runtime_scope(profile_home):
            via_helper = await runner._run_in_executor_with_context(_probe)
            via_bare = await loop.run_in_executor(None, _probe)

        assert via_helper == str(profile_home)
        # The defect this fix closes: the bare hop cannot see the override.
        assert via_bare != str(profile_home)
