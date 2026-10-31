"""The hourly gateway housekeeping chase must include the peer-delivery queue sweep.

Trigger 4 of spec §3.1: orphaned claims and over-age delivery records are recovered
hourly even on a gateway that sees no peer send in that hour. Upstream later moved the
tag -> cleanup table out of ``gateway/run.py`` into ``gateway/run_housekeeping_disk.py``;
a merge that keeps only the new file silently drops the queue from the chore list, so
the sweep then runs only from the 30s drainer tick. This pins the wiring itself.
"""

from __future__ import annotations


def test_hourly_housekeeping_runs_the_delivery_queue_sweep(monkeypatch):
    from gateway import run_housekeeping_disk as housekeeping
    import tools.bot_delivery_queue as delivery_queue

    chores: dict[str, list] = {}

    # gateway.run is the facade the table lives behind; capture the registrations instead of
    # spawning the housekeeping thread.
    import gateway.run as gateway_run

    def _record(name, chore):
        chores[name] = [chore]

    calls = []

    def _sweep(*, max_age_hours=None):
        calls.append(max_age_hours)
        return 0

    monkeypatch.setattr(gateway_run, "_housekeeping_chore", _record)
    monkeypatch.setattr(delivery_queue, "cleanup_bot_delivery_queue", _sweep)

    housekeeping.housekeeping_media_caches()

    assert "Bot delivery cache cleanup" in chores, sorted(chores)
    chores["Bot delivery cache cleanup"][0]()
    assert len(calls) == 1, "the registered chore must call the queue sweep exactly once"
