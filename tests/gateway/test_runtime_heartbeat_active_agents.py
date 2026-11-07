"""The housekeeping heartbeat re-publishes the live ``active_agents`` aggregate every tick.

``_persist_active_agents`` writes the count only at messaging-turn boundaries, but the aggregate
(``_active_work_count``) also sums cron jobs, API runs and deferred workers, which end with no
write. A turn that ends while a cron job is in flight persists 1; the cron job finishes; the file
keeps saying 1 until the next chat turn, which can be days away. External idle gates (updaters,
quiet-restart gates, ``/health/detailed`` readers) see that file and report the gateway busy
(#122813). The heartbeat, already the one periodic writer, recomputes the count so a missed
boundary self-heals within one tick.
"""

from types import SimpleNamespace

import gateway.run as gateway_run


class _OneTick:
    def __init__(self):
        self.ticks = 0

    def is_set(self):
        return self.ticks >= 1

    def wait(self, timeout=None):
        self.ticks += 1
        return True


def _heartbeat_calls(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **fields: calls.append(fields))
    gateway_run._start_gateway_housekeeping(_OneTick(), interval=0, runner=runner)
    return calls


def _runner(count):
    return SimpleNamespace(_active_work_count=count, adapters={}, _profile_adapters={},
                           _profile_bot_platforms={}, config=None)


def test_heartbeat_republishes_live_count_so_a_pinned_value_heals(monkeypatch):
    calls = _heartbeat_calls(monkeypatch, _runner(lambda: 0))
    assert calls and calls[0] == {"active_agents": 0}


def test_heartbeat_reports_in_flight_work_while_it_runs(monkeypatch):
    calls = _heartbeat_calls(monkeypatch, _runner(lambda: 2))
    assert calls[0] == {"active_agents": 2}


def test_heartbeat_still_stamps_when_the_count_cannot_be_read(monkeypatch):
    def boom():
        raise RuntimeError("cron module half-imported")

    calls = _heartbeat_calls(monkeypatch, _runner(boom))
    # A bare stamp keeps the wedge detector (updated_at age) alive and leaves the last count as-is.
    assert calls[0] == {}


def test_heartbeat_without_a_runner_is_a_bare_stamp(monkeypatch):
    calls = _heartbeat_calls(monkeypatch, None)
    assert calls[0] == {}
