"""CLI shutdown must flush a final process checkpoint so a crash that bypasses
atexit still leaves the most recent background-process state for the next
startup recovery."""


def test_run_cleanup_writes_final_checkpoint(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "tools.process_registry.process_registry._write_checkpoint",
        lambda: writes.append(1),
    )
    # _run_cleanup touches many subsystems; stub the heavy ones so the test
    # only asserts the checkpoint write happened.
    monkeypatch.setattr("cli._cleanup_all_terminals", lambda *a, **k: None)
    monkeypatch.setattr("cli._cleanup_all_browsers", lambda *a, **k: None)
    monkeypatch.setattr("tools.async_delegation.interrupt_all", lambda reason="": 0, raising=False)
    from cli import _run_cleanup
    _run_cleanup(notify_session_finalize=False)
    assert len(writes) == 1
