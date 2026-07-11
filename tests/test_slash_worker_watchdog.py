import psutil

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_parent_alive_create_time_drifts():
    # Issue #62505: the public create_time() epoch is not a stable
    # process-identity contract (psutil#2526). A live parent whose
    # create_time() drifted must NOT be classified orphaned, otherwise the
    # watchdog SIGTERMs a live MCP server. The kernel parent/child
    # relationship (matching ppid + pid still exists) is sufficient.
    me = psutil.Process()
    assert (
        slash_worker._is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is False
    )


def test_is_orphaned_true_when_parent_pid_gone():
    # The original parent PID no longer exists -> genuinely orphaned.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 1234) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    me = psutil.Process()
    assert (
        slash_worker._is_orphaned(me.pid, me.create_time(), getppid=lambda: me.pid) is False
    )
