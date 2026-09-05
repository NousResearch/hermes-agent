import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    session_source_scope,
    set_session_vars,
)
from run_agent import _session_source_for_agent


@pytest.fixture(autouse=True)
def _reset_contextvars():
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_session_source_context_overrides_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")
    try:
        assert _session_source_for_agent("tui") == "tool"
    finally:
        clear_session_vars(tokens)


def test_session_source_falls_back_to_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert _session_source_for_agent("tui") == "tui"


def test_session_source_scope_overrides_then_restores(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    # Inside the scope the persisted source is "voice" even though the agent
    # platform is "api_server"; outside it, the platform is used again.
    assert _session_source_for_agent("api_server") == "api_server"
    with session_source_scope("voice"):
        assert _session_source_for_agent("api_server") == "voice"
    assert _session_source_for_agent("api_server") == "api_server"


def test_session_source_scope_restores_prior_bound_value(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")
    try:
        assert _session_source_for_agent("api_server") == "tool"
        with session_source_scope("voice"):
            assert _session_source_for_agent("api_server") == "voice"
        # reset() restores the previously-bound source, not _UNSET/platform.
        assert _session_source_for_agent("api_server") == "tool"
    finally:
        clear_session_vars(tokens)


