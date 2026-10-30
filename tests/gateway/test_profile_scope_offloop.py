"""Routed-profile scopes must resolve their files off the event loop too.

``_async_profile_runtime_scope`` already offloaded the *secret* hydration; the terminal policy
behind it (``.env`` + ``config.yaml``) was still built on the loop, on every routed turn and on
every handoff-watcher tick that enters a profile scope. Same blocking read, same failure mode as
the launch seam: a stalled disk read holds the loop past the liveness probe and
``gateway.shutdown_watchdog`` exits the gateway with 75.

These tests pin both halves together: the build runs in a worker thread, and the scope a body sees
is identical to the synchronous seam's.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tools import terminal_scope
from tools.terminal_scope import (
    TerminalPolicyRefusal,
    TerminalPolicyUnavailable,
    get_terminal_scope,
)


@pytest.fixture
def profile_home(tmp_path):
    home = tmp_path / "profiles" / "beta"
    home.mkdir(parents=True)
    (home / ".env").write_text("ROUTED_TOKEN=from-dotenv\n", encoding="utf-8")
    (home / "config.yaml").write_text("terminal:\n  env: local\n", encoding="utf-8")
    return home


@pytest.fixture
def gateway_run():
    from gateway import run as gateway_run
    return gateway_run


def test_terminal_policy_is_built_off_the_loop(gateway_run, profile_home, monkeypatch):
    sentinel = {"TERMINAL_ENV": "local", "TERMINAL_CWD": "/tmp/routed"}
    threads: dict[str, int] = {}

    def fake_build(_home, *, env_overlay=None):
        threads["build"] = threading.get_ident()
        return dict(sentinel)

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", fake_build)

    observed: dict[str, object] = {}

    async def body():
        threads["loop"] = threading.get_ident()
        async with gateway_run._async_profile_runtime_scope(profile_home):
            observed["scope"] = dict(get_terminal_scope() or {})
            observed["thread"] = threading.get_ident()

    asyncio.run(body())

    assert threads["build"] != threads["loop"], "terminal policy was built on the event loop"
    assert observed["thread"] == threads["loop"]
    assert observed["scope"] == sentinel


def test_loop_keeps_running_while_the_terminal_policy_stalls(gateway_run, profile_home, monkeypatch):
    real_build = terminal_scope.build_profile_terminal_scope
    release = threading.Event()

    def stalled_build(*args, **kwargs):
        # Blocks the worker until the loop proves it can still schedule work (no wall-clock wait).
        if not release.wait(2.0):
            raise AssertionError("event loop did not keep running while the build was in flight")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", stalled_build)

    async def body():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while not release.is_set():
                await asyncio.sleep(0)  # yield only
                ticks += 1
                if ticks >= 5:
                    release.set()

        task = asyncio.create_task(ticker())
        try:
            async with gateway_run._async_profile_runtime_scope(profile_home):
                pass
        finally:
            task.cancel()
        return ticks

    assert asyncio.run(body()) >= 5, "event loop was blocked for the whole build"


def test_routed_scope_matches_the_synchronous_seam(gateway_run, profile_home):
    with gateway_run._profile_runtime_scope(profile_home):
        sync_scope = dict(get_terminal_scope() or {})

    async def body():
        async with gateway_run._async_profile_runtime_scope(profile_home):
            return dict(get_terminal_scope() or {})

    assert asyncio.run(body()) == sync_scope


def test_unreadable_policy_still_installs_the_refusal_scope(gateway_run, profile_home, monkeypatch):
    def broken_build(*_args, **_kwargs):
        raise TerminalPolicyUnavailable("cannot read config.yaml")

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", broken_build)

    async def body():
        async with gateway_run._async_profile_runtime_scope(profile_home):
            return get_terminal_scope()

    installed = asyncio.run(body())

    assert isinstance(installed, TerminalPolicyRefusal)
    assert "cannot read config.yaml" in installed.reason


def test_prepared_policy_is_not_rebuilt(gateway_run, profile_home, monkeypatch):
    """An explicit ``prepared_terminal_scope`` is installed as-is (no second read)."""
    calls: list[int] = []

    def counting_build(*_args, **_kwargs):
        calls.append(1)
        return {"TERMINAL_ENV": "docker"}

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", counting_build)

    prepared = (True, {"TERMINAL_ENV": "ssh"})
    with gateway_run._profile_runtime_scope(profile_home, prepared_terminal_scope=prepared):
        assert get_terminal_scope() == {"TERMINAL_ENV": "ssh"}

    assert calls == []
