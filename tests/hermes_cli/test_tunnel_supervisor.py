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


import hermes_cli.tunnel_supervisor as ts_mod
from hermes_cli.tunnel_supervisor import TunnelSupervisor


_OFFSET = 1_000_000.0  # wall-clock = monotonic + _OFFSET (faked time.time)


def _sup(tmp_path, monkeypatch, *, counters, times, idle=1800.0,
         hold_request_id=False, approved_duration=None, denied=False):
    """Supervisor with scripted counter + monotonic-time sequences.

    Coordinated dual clock (models reality): monotonic = scripted `times`;
    wall (time.time) = monotonic + _OFFSET. Approvals are filed as wall-clock
    deadlines (OFFSET + duration) so _check_hold's wall->monotonic conversion
    is exercised — the integration boundary no per-task test previously hit.

    hold_request_id=True files a request; approved_duration (seconds) approves
    it (wall deadline = OFFSET + duration); denied=True denies it.
    """
    cfg = {"idle_timeout_seconds": idle, "drain_seconds": 0,
           "poll_interval_seconds": 5, "metrics_port": 0,
           "zone": "noit2.com", "tunnel_name": "t", "credentials_file": "",
           "admin": ["admin1"], "routes": []}
    it_c = iter(counters)
    it_t = iter(times)
    mono_now = [0.0]
    killed = {"flag": False}

    def counter():
        return next(it_c)
    def ts():
        v = next(it_t)
        mono_now[0] = v
        return v
    class P:
        def terminate(self):
            killed["flag"] = True
        def wait(self, timeout=None):
            return 0
    def spawn(*a, **kw):
        return P()
    def sleep(s):
        pass

    monkeypatch.setattr(ts_mod.time, "time", lambda: mono_now[0] + _OFFSET)

    path = str(tmp_path / "hold.jsonl")
    filed_id = None
    if hold_request_id:
        from hermes_cli import tunnel_approvals as ta
        filed_id = ta.file_request(path, user="alice",
                                    subdomains=["alice.noit2.com"], reason="demo",
                                    requested_until=None)
        if approved_duration is not None:
            ta.approve(path, filed_id, until=mono_now[0] + _OFFSET + approved_duration,
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
    def test_activity_resets_idle_clock(self, tmp_path, monkeypatch):
        sup, killed = _sup(tmp_path, monkeypatch, counters=[0, 10, 20],
                           times=[0.0, 100.0, 1900.0])
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.last_activity == 100.0
        assert sup.tick() is True
        assert sup.last_activity == 1900.0
        assert sup.closed is False
        assert killed["flag"] is False

    def test_idle_expiry_closes(self, tmp_path, monkeypatch):
        sup, killed = _sup(tmp_path, monkeypatch, counters=[0, 0, 0],
                           times=[0.0, 100.0, 2000.0])
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.tick() is False
        assert sup.closed is True
        assert killed["flag"] is True

    def test_hold_approved_extends_past_idle(self, tmp_path, monkeypatch):
        sup, killed = _sup(tmp_path, monkeypatch, counters=[0, 0, 0],
                           times=[0.0, 2000.0, 4000.0],
                           hold_request_id=True, approved_duration=5000.0)
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.hold_until == 5000.0
        assert sup.tick() is True
        assert sup.closed is False

    def test_hold_denied_closes_on_idle(self, tmp_path, monkeypatch):
        sup, killed = _sup(tmp_path, monkeypatch, counters=[0, 0, 0],
                           times=[0.0, 100.0, 2000.0],
                           hold_request_id=True, denied=True)
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.tick() is False
        assert sup.closed is True

    def test_hold_approval_expires_then_idle_resumes(self, tmp_path, monkeypatch):
        # The clock-domain integration test: approved_until is wall-clock;
        # _check_hold must convert to monotonic so the hold can expire and the
        # idle timer resume (the core dead-man's-switch guarantee). Without
        # the fix, hold_until would be ~1_000_100 (wall) and the hold never
        # expires, so the final tick would NOT close.
        sup, killed = _sup(tmp_path, monkeypatch, counters=[0, 0, 0, 0],
                           times=[0.0, 50.0, 100.0, 2000.0],
                           hold_request_id=True, approved_duration=100.0)
        assert sup.tick() is True              # t=0: hold_until = 0 + 100 = 100
        assert sup.hold_until == 100.0         # monotonic domain, NOT ~1.75e9
        assert sup.tick() is True              # t=50: hold active (50<100)
        assert sup.tick() is True              # t=100: hold expired -> idle (100-0<1800) -> open
        assert sup.tick() is False             # t=2000: idle (2000-0>=1800) -> close
        assert sup.closed is True
        assert killed["flag"] is True
