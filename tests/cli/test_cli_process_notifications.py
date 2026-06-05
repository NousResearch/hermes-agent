import queue


def test_process_notification_drain_does_not_wait_for_empty_input(monkeypatch):
    import cli as cli_mod
    import tools.process_registry as pr_mod

    cli_obj = object.__new__(cli_mod.HermesCLI)
    cli_obj._pending_input = queue.Queue()
    cli_obj._pending_input.put("already queued")

    class _FakeRegistry:
        def drain_notifications(self):
            return [("completed", "synthetic completion")]

    monkeypatch.setattr(pr_mod, "process_registry", _FakeRegistry())

    cli_obj._drain_process_notifications()

    assert cli_obj._pending_input.get_nowait() == "already queued"
    assert cli_obj._pending_input.get_nowait() == "synthetic completion"
