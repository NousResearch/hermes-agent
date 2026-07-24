"""CLI startup must recover detached background processes from checkpoint.

Regression test for the second root cause of disappearing pollers: the CLI
never called recover_from_checkpoint(), so background processes spawned in a
previous CLI session (pollers, servers) became invisible to
process(action='list') while their OS processes kept running.
"""


def test_cli_run_conversation_recovers_background_processes(monkeypatch):
    """The CLI conversation loop must call process_registry.recover_and_log()
    once at startup, after the agent is built and before the first user turn."""
    calls = []

    def fake_recover_and_log():
        calls.append(1)
        return 0

    monkeypatch.setattr(
        "tools.process_registry.process_registry.recover_and_log",
        fake_recover_and_log,
    )
    from cli import _recover_background_processes_on_startup
    _recover_background_processes_on_startup()
    assert len(calls) == 1


def test_cli_startup_recovery_swallows_errors(monkeypatch):
    """A failure in recover_and_log must never raise into CLI startup."""
    def fake_recover_and_log():
        raise RuntimeError("checkpoint disk exploded")
    monkeypatch.setattr(
        "tools.process_registry.process_registry.recover_and_log",
        fake_recover_and_log,
    )
    from cli import _recover_background_processes_on_startup
    # Must not raise.
    _recover_background_processes_on_startup()
