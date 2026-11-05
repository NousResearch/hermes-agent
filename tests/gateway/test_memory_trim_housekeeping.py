"""Memory-trim coverage for the long-lived messaging gateway housekeeper."""

import gateway.run as gateway_run


class _OneTickStopEvent:
    """Run one housekeeping tick without a sleep or background thread."""

    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_calls_periodic_memory_trim(monkeypatch):
    """The tick must ask for the allocator trim only: this lane fires every cooldown, and a forced
    full ``gc.collect()`` here (400-600ms with the GIL held on a live gateway) froze every thread in
    the process once a minute (#139763)."""
    from hermes_cli import mem_trim

    calls = []
    monkeypatch.setattr(
        mem_trim,
        "trim_memory",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert [call.get("reason") for call in calls] == ["messaging gateway housekeeping"]
    assert calls[0].get("collect") is False
