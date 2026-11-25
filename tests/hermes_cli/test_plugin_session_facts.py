"""``hermes_cli.plugin_session`` answers the running turn's facts: the frozen launch policy
(``session_policy()``) and the gateway turn's message event (``turn_event()``).

Both are what a plugin needs to act like the turn it runs in, without importing a gateway internal.
"""
import contextlib
from types import SimpleNamespace


@contextlib.contextmanager
def _turn_of(tmp_path, monkeypatch, **request):
    """The task-local state a local session's turn runs under (the policy its launch froze)."""
    from gateway import session_policy
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    policy = session_policy.build_policy(
        {"source": "tui", "cwd": str(tmp_path), **request}, {"model": {"default": "m"}})
    token = session_policy._ACTIVE_POLICY.set(policy)
    try:
        yield policy
    finally:
        session_policy._ACTIVE_POLICY.reset(token)


def test_session_policy_answers_the_running_turns_launch_facts(tmp_path, monkeypatch):
    from hermes_cli.plugin_session import session_policy

    assert session_policy() is None  # outside a turn
    with _turn_of(tmp_path, monkeypatch) as policy:
        assert session_policy().source == "tui"
        assert session_policy().cwd == str(tmp_path)
        assert session_policy().platform == policy.platform
    assert session_policy() is None


def test_turn_event_answers_only_inside_a_bound_turn():
    from gateway.config import Platform
    from hermes_cli.plugin_session import turn_event

    assert turn_event() is None  # outside a gateway turn
    undo = _bound_event(_event(Platform.TELEGRAM))
    try:
        assert turn_event().user_id == "u-1"
    finally:
        undo()
    assert turn_event() is None


def _bound_event(event):
    from gateway.session_context import clear_session_vars, set_session_vars
    tokens = set_session_vars(platform="x", chat_id="1", session_key="k", turn_event=event)
    return lambda: clear_session_vars(tokens)


def _event(platform, raw=None, **fields):
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    source = SessionSource(platform=platform, chat_id="chat-9", user_id="u-1", is_bot=fields.pop("is_bot", False))
    return MessageEvent(text="hi", source=source, user_id="u-1", message_id="m-1", raw_message=raw, **fields)


def test_session_policy_is_a_frozen_view_without_the_policys_internals(tmp_path, monkeypatch):
    import dataclasses

    import pytest

    from hermes_cli.plugin_session import SessionPolicyView, session_policy

    with _turn_of(tmp_path, monkeypatch, toolsets=["web"]) as policy:
        view = session_policy()
        assert isinstance(view, SessionPolicyView) and view is not policy
        assert (view.source, view.platform, view.cwd, view.model) == (
            policy.source, policy.platform, policy.cwd, policy.model)
        assert view.toolsets == tuple(policy.toolsets)
        assert not hasattr(view, "config_json") and not hasattr(view, "config")
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.cwd = "/elsewhere"


def test_turn_event_is_a_frozen_snapshot_not_the_live_event():
    import dataclasses

    import pytest

    from gateway.config import Platform
    from hermes_cli.plugin_session import TurnEventView, turn_event

    raw = SimpleNamespace(token="bot-token")
    event = _event(Platform.TELEGRAM, raw, reply_to_message_id="m-0", reply_to_is_own_message=True,
                   internal=True, inherit_channel_pin=True)
    event.source.is_bot = True
    undo = _bound_event(event)
    try:
        view = turn_event()
    finally:
        undo()
    assert isinstance(view, TurnEventView) and view is not event
    assert (view.platform, view.user_id, view.chat_id, view.message_id) == ("telegram", "u-1", "chat-9", "m-1")
    assert (view.reply_to_message_id, view.reply_to_is_own_message, view.is_bot) == ("m-0", True, True)
    assert (view.internal, view.inherit_channel_pin, view.forwarded) == (True, True, False)
    assert not hasattr(view, "raw_message") and "bot-token" not in repr(view)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.user_id = "someone-else"
    event.message_id = "changed"
    assert view.message_id == "m-1"  # a snapshot: later changes to the event do not show


def test_turn_event_says_whether_the_message_is_a_forward_per_platform():
    from gateway.config import Platform
    from hermes_cli.plugin_session import turn_event

    feishu = lambda kind: SimpleNamespace(event=SimpleNamespace(message=SimpleNamespace(message_type=kind)))
    cases = [
        (Platform.DISCORD, SimpleNamespace(message_snapshots=[object()]), True),
        (Platform.DISCORD, SimpleNamespace(message_snapshots=[]), False),
        (Platform.TELEGRAM, SimpleNamespace(forward_origin=object()), True),
        (Platform.TELEGRAM, SimpleNamespace(forward_origin=None), False),
        (Platform.FEISHU, feishu("merge_forward"), True),
        (Platform.FEISHU, feishu("text"), False),
        (Platform.SLACK, SimpleNamespace(message_snapshots=[object()], forward_origin=object()), False),
        (Platform.TELEGRAM, None, False),
    ]
    for platform, raw, expected in cases:
        undo = _bound_event(_event(platform, raw))
        try:
            assert turn_event().forwarded is expected, (platform, raw)
        finally:
            undo()
