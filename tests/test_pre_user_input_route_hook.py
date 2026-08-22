"""Behavior contract for the pre-user-input routing plugin hook."""

from hermes_cli import lifecycle
from hermes_cli.plugins import VALID_HOOKS


def test_pre_user_input_route_is_public_plugin_hook():
    assert "pre_user_input_route" in VALID_HOOKS


def test_pre_user_input_route_uses_first_valid_rewrite(monkeypatch):
    seen = {}

    def invoke(name, **payload):
        seen.update(payload)
        return [
            {"action": "rewrite", "text": ""},
            {"action": "allow"},
            {"action": "rewrite", "text": "/goal ship it", "notice": "Routed"},
            {"action": "rewrite", "text": "ignored"},
        ]

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke)

    text, notice = lifecycle.route_pre_user_input(
        surface="cli",
        text="ship it",
        session_key="session-1",
        platform="cli",
        goal_active=False,
        has_attachments=False,
    )

    assert (text, notice) == ("/goal ship it", "Routed")
    assert seen == {
        "surface": "cli",
        "text": "ship it",
        "session_key": "session-1",
        "platform": "cli",
        "goal_active": False,
        "has_attachments": False,
    }


def test_pre_user_input_route_fails_open_on_invalid_or_failed_hook(monkeypatch):
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda *_args, **_kwargs: [None, "rewrite", {"action": "rewrite", "text": 3}],
    )
    assert lifecycle.route_pre_user_input(
        surface="gateway", text="hello", session_key="s", platform="telegram",
        goal_active=False, has_attachments=False,
    ) == ("hello", None)


def test_pre_user_input_route_bypasses_ineligible_input(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda *_args, **_kwargs: calls.append(True),
    )

    for text, goal_active, has_attachments in [
        (None, False, False),
        ("", False, False),
        ("   ", False, False),
        (" /goal status", False, False),
        ("follow up", True, False),
        ("caption", False, True),
    ]:
        assert lifecycle.route_pre_user_input(
            surface="gateway",
            text=text,
            session_key="s",
            platform="telegram",
            goal_active=goal_active,
            has_attachments=has_attachments,
        ) == (text, None)

    assert calls == []

    def boom(*_args, **_kwargs):
        raise RuntimeError("broken plugin host")

    monkeypatch.setattr(lifecycle, "invoke_hook", boom)
    assert lifecycle.route_pre_user_input(
        surface="gateway", text="hello", session_key="s", platform="telegram",
        goal_active=False, has_attachments=False,
    ) == ("hello", None)
