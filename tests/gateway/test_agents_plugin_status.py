"""Bounded plugin status lines for /agents: malformed output ignored, session isolation."""

import pytest

from gateway import slash_commands_status as scs


@pytest.fixture(autouse=True)
def _clean_providers():
    scs.reset_agents_status_providers_for_tests()
    yield
    scs.reset_agents_status_providers_for_tests()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: "agent:main:test:dm:1"
    return runner


class _Event:
    source = None


@pytest.mark.asyncio
async def test_malformed_provider_output_ignored():
    scs.register_agents_status_provider(lambda session_key: "all good")
    scs.register_agents_status_provider(lambda session_key: None)
    scs.register_agents_status_provider(lambda session_key: 12345)
    scs.register_agents_status_provider(lambda session_key: {"line": "x"})
    out = await _make_runner()._handle_agents_command(_Event())
    assert "all good" in out
    assert "12345" not in out


@pytest.mark.asyncio
async def test_provider_list_filtering_and_errors_ignored():
    scs.register_agents_status_provider(lambda session_key: ["fine", None, 42, "  "])

    def _bad(session_key):
        raise RuntimeError("boom")

    scs.register_agents_status_provider(_bad)
    out = await _make_runner()._handle_agents_command(_Event())
    assert "fine" in out


@pytest.mark.asyncio
async def test_provider_receives_session_key_only():
    seen: list = []
    scs.register_agents_status_provider(lambda session_key: seen.append(session_key) or "ok")
    out = await _make_runner()._handle_agents_command(_Event())
    assert seen == ["agent:main:test:dm:1"]
    assert "ok" in out


def test_provider_lines_bounded():
    for i in range(10):
        scs.register_agents_status_provider(
            lambda session_key, i=i: [f"line-{i}-{j}" for j in range(10)]
        )
    lines = scs.agents_plugin_status_lines("agent:main:test:dm:1")
    assert len(lines) <= scs.AGENTS_PLUGIN_LINES_CAP
    assert all(isinstance(line, str) and line for line in lines)
