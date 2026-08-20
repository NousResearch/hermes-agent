"""Pre-turn hygiene compression must run inside the routed profile's scope.

The multiplexed inbound handler wraps the whole message in
``_profile_runtime_scope``, which installs the profile's ``HERMES_HOME``
override and its secret scope as **contextvars** — its own docstring notes they
reach the agent worker thread "via ``copy_context()``".

The automatic pre-turn hygiene compression handed ``_compress_context`` to a
bare ``loop.run_in_executor(None, fn)``. A bare executor hop starts the worker
with an empty context, so inside it ``get_hermes_home()`` fell back to the
default profile and ``get_secret`` read process-global ``os.environ`` — which
under multiplexing may hold a *different* profile's credentials. That is
exactly the failure ``gateway/slash_commands.py`` already routes ``/compress``
through ``_run_in_executor_with_context`` to avoid; the automatic path had not
been given the same treatment.

Drives the real ``_profile_runtime_scope`` and the real spawn helper — the
contextvar loss is a property of the hop itself, so mocking it away would test
nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "coder"
    home.mkdir(parents=True)
    (home / ".env").write_text("PROFILE_MARKER_KEY=coder-secret\n", encoding="utf-8")
    root.mkdir(exist_ok=True)
    (root / ".env").write_text("PROFILE_MARKER_KEY=root-secret\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The unscoped answer: what a context-less worker thread would see.
    monkeypatch.setenv("PROFILE_MARKER_KEY", "root-secret")
    return home


class _RecordingAgent:
    """Stands in for the hygiene AIAgent; records what the worker thread sees."""

    def __init__(self):
        self.seen_home = None
        self.seen_secret = None
        self.called_with = None

    def _compress_context(self, messages, _unused, *, approx_tokens, commit_fence):
        from hermes_constants import get_hermes_home

        self.seen_home = str(get_hermes_home())
        try:
            from agent.secret_scope import get_secret

            self.seen_secret = get_secret("PROFILE_MARKER_KEY")
        except Exception:
            self.seen_secret = None
        self.called_with = (messages, approx_tokens, commit_fence)
        return ["compressed"], True


async def _spawn_and_wait(agent, profile_home):
    from gateway.run import GatewayRunner, _profile_runtime_scope

    loop = asyncio.get_running_loop()
    with _profile_runtime_scope(profile_home):
        future = GatewayRunner._spawn_hygiene_compression(
            loop, agent, ["m1", "m2"], 1234, object()
        )
        return await future


class TestHygieneCompressionRunsUnderTheRoutedProfile:
    @pytest.mark.asyncio
    async def test_worker_sees_the_profile_home(self, profile_home):
        agent = _RecordingAgent()
        await _spawn_and_wait(agent, profile_home)

        assert agent.seen_home == str(profile_home)

    @pytest.mark.asyncio
    async def test_worker_sees_the_profile_secret_not_the_process_env(
        self, profile_home
    ):
        agent = _RecordingAgent()
        await _spawn_and_wait(agent, profile_home)

        assert agent.seen_secret == "coder-secret"

    @pytest.mark.asyncio
    async def test_arguments_and_result_are_passed_through_unchanged(
        self, profile_home
    ):
        agent = _RecordingAgent()
        fence = object()

        from gateway.run import GatewayRunner, _profile_runtime_scope

        loop = asyncio.get_running_loop()
        with _profile_runtime_scope(profile_home):
            result = await GatewayRunner._spawn_hygiene_compression(
                loop, agent, ["a", "b"], 99, fence
            )

        assert result == (["compressed"], True)
        assert agent.called_with == (["a", "b"], 99, fence)

    @pytest.mark.asyncio
    async def test_returns_a_future_the_progress_wait_can_poll(self, profile_home):
        """The caller polls .done()/.result(); keep that interface."""
        from gateway.run import GatewayRunner, _profile_runtime_scope

        agent = _RecordingAgent()
        loop = asyncio.get_running_loop()
        with _profile_runtime_scope(profile_home):
            future = GatewayRunner._spawn_hygiene_compression(
                loop, agent, ["m"], 1, object()
            )
            assert hasattr(future, "done") and hasattr(future, "cancel")
            await future
            assert future.done()
            assert future.result() == (["compressed"], True)


class TestSingleProfileGatewayUnchanged:
    @pytest.mark.asyncio
    async def test_no_scope_installed_still_resolves_the_launch_home(
        self, profile_home, tmp_path
    ):
        """Non-multiplexed gateways never enter the scope — behaviour must not shift."""
        from gateway.run import GatewayRunner

        agent = _RecordingAgent()
        loop = asyncio.get_running_loop()
        await GatewayRunner._spawn_hygiene_compression(
            loop, agent, ["m"], 1, object()
        )

        assert agent.seen_home == str(tmp_path / ".hermes")
