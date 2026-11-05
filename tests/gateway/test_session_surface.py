"""Session-source labels must not change CLI transport classification."""

import pytest

import gateway.session_context as session_context


@pytest.fixture(autouse=True)
def _reset_contextvars():
    session_context.reset_session_vars()
    yield
    session_context.reset_session_vars()


def test_explicit_cli_source_label_is_not_a_messaging_surface(monkeypatch):
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "my-orchestrator")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")

    assert session_context.session_is_messaging_surface() is False


def test_platform_identity_still_wins_over_explicit_source_label(monkeypatch):
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "my-orchestrator")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")

    assert session_context.session_is_messaging_surface() is True


def test_contextvar_platform_wins_without_environment_platform(monkeypatch):
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "my-orchestrator")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")
    token = session_context._SESSION_PLATFORM.set("telegram")
    try:
        assert session_context.session_is_messaging_surface() is True
    finally:
        session_context._SESSION_PLATFORM.reset(token)


def test_contextvar_source_survives_inherited_explicit_marker(monkeypatch):
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "my-orchestrator")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")
    token = session_context._SESSION_SOURCE.set("telegram")
    try:
        assert session_context.session_is_messaging_surface() is True
    finally:
        session_context._SESSION_SOURCE.reset(token)
