import pytest
from hermes_cli.tunnel_supervisor import reset_idle_on, should_close_now


class TestPolicy:
    def test_reset_when_counter_increases(self):
        assert reset_idle_on(10, 11) is True

    def test_no_reset_when_counter_unchanged(self):
        assert reset_idle_on(10, 10) is False

    def test_no_reset_when_counter_decreases(self):
        # a poll hiccup / counter reset should NOT count as activity
        assert reset_idle_on(11, 10) is False

    def test_close_after_idle_timeout(self):
        # now - last_activity = 1900s >= 1800s idle -> close
        state = {"now": 2000.0, "last_activity": 100.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is True

    def test_open_before_idle_timeout(self):
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is False

    def test_hold_active_keeps_open_past_idle(self):
        state = {"now": 1000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 2000.0}
        assert should_close_now(state) is False

    def test_hold_expired_falls_back_to_idle_not_hard_kill(self):
        # hold_until in the past: fall back to idle rule.
        # last_activity recent -> still open (no hard kill on approval expiry).
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is False

    def test_hold_expired_and_idle_closes(self):
        # hold_until in the past (expired) -> fall back to idle rule.
        # now - last_activity = 2000s >= 1800s idle -> close.
        state = {"now": 2000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is True


from hermes_cli.tunnel_supervisor import TunnelSupervisor


def _sup(tmp_path, *, counters, times, idle=1800.0, hold_request_id=False,
         approved_until=None, denied=False):
    """Build a supervisor with scripted counter + time sequences.

    hold_request_id=False -> no hold request. Pass hold_request_id=True to
    file a real request (capturing its generated id) and optionally
    approve/deny it; the supervisor is wired to that same id.
    """
    cfg = {"idle_timeout_seconds": idle, "drain_seconds": 0,
           "poll_interval_seconds": 5, "metrics_port": 0,
           "zone": "noit2.com", "tunnel_name": "t", "credentials_file": "",
           "admin": ["admin1"], "routes": []}
    it_c = iter(counters)
    it_t = iter(times)
    killed = {"flag": False}

    def counter():
        return next(it_c)
    def ts():
        return next(it_t)
    class P:
        def terminate(self):
            killed["flag"] = True
        def wait(self, timeout=None):
            return 0
    def spawn(*a, **kw):
        return P()
    def sleep(s):
        pass

    path = str(tmp_path / "hold.jsonl")
    filed_id = None
    if hold_request_id:
        from hermes_cli import tunnel_approvals as ta
        filed_id = ta.file_request(path, user="alice",
                                    subdomains=["alice.noit2.com"], reason="demo",
                                    requested_until=None)
        if approved_until is not None:
            ta.approve(path, filed_id, until=approved_until,
                       by="admin1", admin_ids=["admin1"])
        elif denied:
            ta.deny(path, filed_id, reason="no", by="admin1", admin_ids=["admin1"])

    sup = TunnelSupervisor(
        cfg, path, hold_request_id=filed_id,
        time_source=ts, metrics_counter=counter,
        spawn_cloudflared=spawn, sleep=sleep,
    )
    return sup, killed


class TestSupervisor:
    def test_activity_resets_idle_clock(self, tmp_path):
        # All-increasing counters -> each tick is activity -> last_activity
        # resets to now -> stays open despite the long wall-clock gap.
        sup, killed = _sup(tmp_path, counters=[0, 10, 20], times=[0.0, 100.0, 1900.0])
        assert sup.tick() is True      # t=0, counter 0 (flat vs init 0) -> open
        assert sup.tick() is True      # t=100, counter 10 -> activity, reset
        assert sup.last_activity == 100.0
        assert sup.tick() is True      # t=1900, counter 20 -> activity resets to 1900 -> open
        assert sup.last_activity == 1900.0
        assert sup.closed is False
        assert killed["flag"] is False

    def test_idle_expiry_closes(self, tmp_path):
        sup, killed = _sup(tmp_path, counters=[0, 0, 0], times=[0.0, 100.0, 2000.0])
        assert sup.tick() is True      # open
        assert sup.tick() is True      # no activity, 100-0 < 1800 -> open
        assert sup.tick() is False     # 2000-0 >= 1800 -> close
        assert sup.closed is True
        assert killed["flag"] is True

    def test_hold_approved_extends_past_idle(self, tmp_path):
        sup, killed = _sup(tmp_path, counters=[0, 0, 0],
                           times=[0.0, 2000.0, 4000.0],
                           hold_request_id=True, approved_until=5000.0)
        assert sup.tick() is True      # t=0, hold active (approved until 5000)
        assert sup.tick() is True      # t=2000, no activity, but hold until 5000 -> open
        assert sup.hold_until == 5000.0
        assert sup.tick() is True      # t=4000, still within hold -> open
        assert sup.closed is False

    def test_hold_denied_closes_on_idle(self, tmp_path):
        sup, killed = _sup(tmp_path, counters=[0, 0, 0],
                           times=[0.0, 100.0, 2000.0],
                           hold_request_id=True, denied=True)
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.tick() is False     # idle closes; denied hold did not extend
        assert sup.closed is True