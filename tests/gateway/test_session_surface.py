"""Session-source labels must not change CLI transport classification."""

import gateway.session_context as session_context


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
