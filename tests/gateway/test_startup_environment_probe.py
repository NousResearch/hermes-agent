"""The gateway boot warm-up primes the local toolchain probe the first prompt build reads (#106064)."""

import pytest

import model_tools
from gateway import run as gateway_run
from tools import env_probe
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope


@pytest.fixture(autouse=True)
def _probe_cache(monkeypatch):
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_: [])
    env_probe._reset_cache_for_tests()
    yield
    env_probe._reset_cache_for_tests()


def _warm(tmp_path, monkeypatch, agent_section: dict, backend: str) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"agent: {agent_section!r}\n", encoding="utf-8")
    token = set_terminal_scope({"TERMINAL_ENV": backend})
    try:
        gateway_run._warm_turn_machinery_sync()
    finally:
        reset_terminal_scope(token)


def test_warmup_leaves_probe_cached_for_first_prompt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(env_probe, "_build_probe_line", lambda: calls.append(1) or "Python toolchain: fixture.")

    _warm(tmp_path, monkeypatch, {}, "local")

    assert env_probe._PROBE_DONE.is_set()
    assert env_probe.get_environment_probe_line() == "Python toolchain: fixture."
    assert calls == [1]  # single worker; the first turn reuses the cache


@pytest.mark.parametrize("agent_section,backend", [({"environment_probe": False}, "local"), ({}, "ssh")])
def test_warmup_skips_probe_when_disabled_or_remote(tmp_path, monkeypatch, agent_section, backend):
    monkeypatch.setattr(env_probe, "_build_probe_line", lambda: pytest.fail("host inspected"))

    _warm(tmp_path, monkeypatch, agent_section, backend)

    assert not env_probe._PROBE_DONE.is_set()
    assert env_probe._PROBE_THREAD is None
