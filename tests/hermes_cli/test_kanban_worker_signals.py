"""Signal routing is independent of callback injection and group lookup failures."""

from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db_worker_signals as signals


def test_injected_sender_does_not_inspect_or_signal_a_group(monkeypatch):
    sender = Mock()
    lookup = Mock(side_effect=AssertionError("callback path must not inspect groups"))
    group_sender = Mock(side_effect=AssertionError("callback path must not signal groups"))
    monkeypatch.setattr(signals.os, "getpgid", lookup, raising=False)
    monkeypatch.setattr(signals.os, "killpg", group_sender, raising=False)

    assert signals.kill_fn(sender) is sender
    target = signals.reclaimed_signal_target(42, sender, allow_group=False)
    target(42, 15)

    sender.assert_called_once_with(42, 15)
    lookup.assert_not_called()
    group_sender.assert_not_called()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("pgid", [42, 7])
def test_only_a_group_leader_uses_group_signalling(monkeypatch, pgid):
    sender, group_sender = Mock(), Mock()
    monkeypatch.setattr(signals.os, "getpgid", lambda pid: pgid)
    monkeypatch.setattr(signals.os, "killpg", group_sender)

    target = signals.reclaimed_signal_target(42, sender, allow_group=True)
    target(42, 15)

    if pgid == 42:
        group_sender.assert_called_once_with(42, 15)
        sender.assert_not_called()
    else:
        sender.assert_called_once_with(42, 15)
        group_sender.assert_not_called()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("error", [PermissionError(), ProcessLookupError()])
def test_group_send_fallback_does_not_signal_a_vanished_pid(monkeypatch, error):
    sender = Mock()
    monkeypatch.setattr(signals.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(signals.os, "killpg", Mock(side_effect=error))
    target = signals.reclaimed_signal_target(42, sender, allow_group=True)

    if isinstance(error, ProcessLookupError):
        with pytest.raises(ProcessLookupError):
            target(42, 15)
        sender.assert_not_called()
    else:
        target(42, 15)
        sender.assert_called_once_with(42, 15)
