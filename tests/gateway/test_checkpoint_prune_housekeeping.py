"""Checkpoint store pruning rides the gateway housekeeping tick, not the constructor.

Its ``git gc`` repacks the whole store (tens of seconds on a GB store); run at construction it
delayed the control socket, adapters and the code_sha stamp, so the first restart of the day (the
``hermes update`` one) looked hung and failed fleet verification.
"""

import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_runs_the_checkpoint_prune(monkeypatch):
    import tools.checkpoint_maintenance as cm

    calls = []
    monkeypatch.setattr(cm, "auto_prune_from_config", lambda: calls.append(True) or {"skipped": False})

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [True]
