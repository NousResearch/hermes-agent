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


def test_session_policy_answers_the_running_turns_launch_policy(tmp_path, monkeypatch):
    from gateway.session_policy import active_policy
    from hermes_cli.plugin_session import session_policy

    assert session_policy() is None  # outside a turn
    with _turn_of(tmp_path, monkeypatch) as policy:
        assert session_policy() is policy
        assert session_policy() is active_policy()
        assert session_policy().source == "tui"
        assert session_policy().cwd == str(tmp_path)
    assert session_policy() is None


def test_turn_event_answers_only_inside_a_bound_turn():
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_event

    assert turn_event() is None  # outside a gateway turn
    event = SimpleNamespace(user_id="u-1")
    tokens = set_session_vars(platform="telegram", chat_id="1", session_key="k", turn_event=event)
    try:
        assert turn_event() is event
        assert turn_event().user_id == "u-1"
    finally:
        clear_session_vars(tokens)
    assert turn_event() is None
