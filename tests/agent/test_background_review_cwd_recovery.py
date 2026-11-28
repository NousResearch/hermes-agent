import os
import shutil


def test_fork_recovers_deleted_session_cwd(tmp_path, monkeypatch):
    """Behavior contract (the review-fork workspace bug): a worker process whose cwd is
    rmtree'd underneath it (kanban complete before the /refine/background review fork)
    must resolve an EXISTING cwd through the contextvars-scoped override — never the
    deleted dir [Errno 2], never a process-global chdir."""
    from agent import background_review, runtime_cwd

    doomed = tmp_path / "scratch-ws"
    doomed.mkdir()
    monkeypatch.chdir(doomed)
    # rmtree refuses to delete the cwd itself; renaming it out from under us leaves the
    # process cwd dangling on the original path — exactly the worker's shape.
    os.rename(doomed, tmp_path / "scratch-ws-dead")
    shutil.rmtree(tmp_path / "scratch-ws-dead")  # cwd now dangling
    # sanity: the base defect — the cwd is dangling (os.getcwd raises/None on a deleted cwd)
    try:
        cwd_ok = os.path.isdir(os.getcwd())
    except (FileNotFoundError, OSError):
        cwd_ok = False
    assert not cwd_ok
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    token = runtime_cwd._SESSION_CWD.set(runtime_cwd._UNSET)  # isolate: no stale override
    try:
        background_review._recover_deleted_session_cwd()
        resolved = runtime_cwd.resolve_agent_cwd().resolve()
        assert resolved.is_dir()
    finally:
        runtime_cwd._SESSION_CWD.reset(token)


def test_recovery_noop_on_live_cwd(tmp_path, monkeypatch):
    """When the cwd is alive the helper must not touch the session override."""
    from agent import background_review, runtime_cwd

    live = tmp_path / "live-ws"
    live.mkdir()
    monkeypatch.chdir(live)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert not runtime_cwd.scoped_session_cwd()
    background_review._recover_deleted_session_cwd()
    assert not runtime_cwd.scoped_session_cwd()
    assert runtime_cwd.resolve_agent_cwd().resolve() == live.resolve()