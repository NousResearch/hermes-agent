import inspect

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_is_orphaned_true_when_parent_pid_is_recycled(monkeypatch):
    # The PID still reports our original parent — but the process occupying
    # it is a different incarnation (OS recycled the PID after a crash).
    monkeypatch.setattr(slash_worker, "_original_parent_identity", "old-incarnation")
    monkeypatch.setattr(
        slash_worker,
        "_read_parent_identity",
        lambda pid: "new-incarnation",
    )

    assert slash_worker._is_orphaned(1234, getppid=lambda: 1234) is True


def test_is_orphaned_false_when_parent_identity_is_unchanged(monkeypatch):
    monkeypatch.setattr(slash_worker, "_original_parent_identity", "same-incarnation")
    monkeypatch.setattr(
        slash_worker,
        "_read_parent_identity",
        lambda pid: "same-incarnation",
    )

    assert slash_worker._is_orphaned(1234, getppid=lambda: 1234) is False


def test_is_orphaned_false_when_parent_identity_is_unreadable(monkeypatch):
    # Transient identity read failure must never kill — fall back to the
    # legacy PID-only verdict instead of guessing.
    monkeypatch.setattr(slash_worker, "_original_parent_identity", "old-incarnation")
    monkeypatch.setattr(slash_worker, "_read_parent_identity", lambda pid: None)

    assert slash_worker._is_orphaned(1234, getppid=lambda: 1234) is False


def test_parent_death_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
    ]
