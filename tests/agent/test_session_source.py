import pytest

from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
from agent.session_source import session_source_for


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
        assert session_source_for("tui") == "tool"
    finally:
        clear_session_vars(tokens)


def test_session_source_falls_back_to_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert session_source_for("tui") == "tui"




@pytest.mark.parametrize("inherited", ["", "tui", "desktop"])
def test_oneshot_run_gets_distinct_source(monkeypatch, inherited):
    """A finite `hermes chat -q` / `hermes -z` run is tagged `oneshot`, whether launched from a plain shell
    or spawned inside a TUI/Desktop session (whose transport label it inherits but is not) (#112550)."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", inherited)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    assert session_source_for("cli") == "oneshot"


def test_oneshot_marker_does_not_relabel_subagents(monkeypatch):
    """Delegate children inside a one-shot process share its env but keep their own platform."""
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    assert session_source_for("subagent") == "subagent"


@pytest.mark.parametrize("inherited", ["kanban", "tool", "a2a"])
def test_oneshot_child_keeps_inherited_automation_source(monkeypatch, inherited):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", inherited)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    assert session_source_for("cli") == inherited


@pytest.mark.parametrize("explicit", ["tui", "desktop"])
def test_oneshot_keeps_explicit_source_flag(monkeypatch, explicit):
    """`hermes chat -q --source tui` is a documented flag, not an inherited transport label: main.py
    marks it HERMES_SESSION_SOURCE_EXPLICIT=1 and the one-shot drop must not override it."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", explicit)
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    assert session_source_for("cli") == explicit
