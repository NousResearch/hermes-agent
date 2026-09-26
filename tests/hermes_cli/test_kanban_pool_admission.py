"""Rate-limit storm mitigation: pool-health admission, escalating backoff, tick circuit.

Context (measured 2026-09-22 on the live board): 3,706 kanban runs ended
``rate_limited`` in 7d; 42% of them ran >2 minutes before dying, and turns whose
start falls inside those windows total ~$17.7k API-equivalent. The existing
``rate_limit_cooldown`` guard DOES work but is FLAT (300s forever), so a task
re-probes an empty pool every 5 minutes indefinitely.

These tests are hermetic: a fake ``/health`` HTTP server and an injected clock.
No real pool, no sleeping, no network beyond loopback.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import hermes_cli.kanban_db as kb
import hermes_cli.kanban_db_connect as kbc
import hermes_cli.kanban_db_dispatch as kbd


# --------------------------------------------------------------------------
# Fake pool /health server
# --------------------------------------------------------------------------

class _HealthServer:
    """Loopback /health returning a caller-controlled payload.

    ``delay`` forces the client-side timeout path; ``status`` forces a non-200.
    """

    def __init__(self, payload, *, status: int = 200, delay: float = 0.0):
        self.payload = payload
        self.status = status
        self.delay = delay
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.hits += 1
                if outer.delay:
                    import time as _t
                    _t.sleep(outer.delay)
                body = json.dumps(outer.payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):  # silence
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._srv.server_port}/health"

    def __enter__(self):
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()


# --------------------------------------------------------------------------
# GATE 1 — pool-health admission
# --------------------------------------------------------------------------

def test_pool_health_zero_eligible_blocks_spawn():
    """eligible_count=0 -> pool-bound profile is not admitted."""
    with _HealthServer({"status": "all_capped", "eligible_count": 0, "pool_size": 15}) as srv:
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=1) is False
        assert srv.hits == 1


def test_pool_health_eligible_allows_spawn():
    with _HealthServer({"status": "ok", "eligible_count": 3, "pool_size": 15}) as srv:
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=1) is True


def test_non_pool_provider_never_probes_health():
    """openai-codex is unaffected: admitted without an HTTP call at all."""
    with _HealthServer({"eligible_count": 0}) as srv:
        assert kbd.pool_admits_spawn("openai-codex", health_url=srv.url, min_eligible=1) is True
        assert srv.hits == 0, "non-pool provider must not probe the pool"


def test_health_unreachable_fails_open():
    """The gate must never become a new outage: unreachable health = admit."""
    # Port 1 on loopback: connection refused, fast.
    assert kbd.pool_admits_spawn(
        "claude-apr", health_url="http://127.0.0.1:1/health", min_eligible=1
    ) is True


def test_health_timeout_fails_open():
    with _HealthServer({"eligible_count": 0}, delay=2.0) as srv:
        assert kbd.pool_admits_spawn(
            "claude-apr", health_url=srv.url, min_eligible=1, timeout=0.25
        ) is True


def test_health_malformed_body_fails_open():
    with _HealthServer({"unexpected": "shape"}) as srv:
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=1) is True


def test_health_non_200_fails_open():
    with _HealthServer({"eligible_count": 0}, status=503) as srv:
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=1) is True


@pytest.mark.parametrize(
    "provider, is_pool",
    [
        ("claude-apr", True), ("claude-bpr", True),
        ("claude-apx-3", True), ("claude-bpx-22", True),
        ("openai-codex", False), ("anthropic", False),
        ("", False), (None, False),
    ],
)
def test_pool_provider_classification(provider, is_pool):
    assert kbd.is_pool_provider(provider) is is_pool


def test_min_eligible_threshold_is_respected():
    """min_eligible=2 rejects a pool with exactly 1 eligible."""
    with _HealthServer({"eligible_count": 1}) as srv:
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=2) is False
        assert kbd.pool_admits_spawn("claude-apr", health_url=srv.url, min_eligible=1) is True


# --------------------------------------------------------------------------
# GATE 2 — escalating per-task backoff
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "consecutive, expected",
    [(1, 300), (2, 900), (3, 2700), (4, 7200), (5, 7200), (99, 7200)],
)
def test_backoff_ladder(consecutive, expected):
    """5m, 15m, 45m, 2h, then capped at 2h."""
    assert kbd.rate_limit_backoff_seconds(consecutive) == expected


def test_backoff_zero_runs_is_no_delay():
    assert kbd.rate_limit_backoff_seconds(0) == 0


def _seed_rate_limited_runs(conn, tid, *, count, ended_at):
    """Seed ``count`` consecutive rate_limited runs, newest ending at ended_at."""
    for i in range(count):
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        end = ended_at - (count - 1 - i) * 10_000
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "started_at=?, ended_at=? WHERE id=?",
            (end - 100, end, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
    conn.commit()


def test_escalating_backoff_second_failure_holds_past_flat_300(monkeypatch):
    """THE REGRESSION THIS CARD EXISTS FOR.

    After 2 consecutive rate_limited runs the task must still be held at
    t+400s. The OLD flat-300s guard released it (measured: 1,538 retries
    clustered at the 5-minute floor).
    """
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-escalate", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=2, ended_at=now)

        monkeypatch.setattr(kb.time, "time", lambda: now + 400)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # 15m ladder step elapsed -> released.
        monkeypatch.setattr(kb.time, "time", lambda: now + 901)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_first_rate_limit_still_uses_5m(monkeypatch):
    """Backward compatible: a single rate_limited run keeps the 300s behaviour."""
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-first", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=1, ended_at=now)

        monkeypatch.setattr(kb.time, "time", lambda: now + 100)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: now + 301)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_backoff_resets_after_non_rate_limited_run(monkeypatch):
    """A crash (any non-rate_limited close) resets the ladder to step 1."""
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-reset", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=3, ended_at=now - 50_000)
        # Newest run is a crash, then one fresh rate_limited.
        kb.claim_task(conn, tid)
        rid = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='crashed', status='crashed', ended_at=? WHERE id=?",
            (now - 20_000, rid),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?", (tid,),
        )
        conn.commit()
        _seed_rate_limited_runs(conn, tid, count=1, ended_at=now)

        # Streak is 1 (not 4): released after 300s, not 7200s.
        # Assert the STREAK directly too — otherwise this test passes against
        # the old flat-300s guard and proves nothing about the reset rule.
        assert kbd.consecutive_rate_limited_runs(conn, tid) == 1
        monkeypatch.setattr(kb.time, "time", lambda: now + 200)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: now + 301)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_consecutive_streak_counts_only_trailing_rate_limited():
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-streak", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=3, ended_at=now)
        assert kbd.consecutive_rate_limited_runs(conn, tid) == 3


# --------------------------------------------------------------------------
# GATE 3 — tick-level circuit breaker
# --------------------------------------------------------------------------

def test_circuit_trips_at_threshold():
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-circuit", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=5, ended_at=now)
        # All 5 ended inside the 10-minute window.
        conn.execute(
            "UPDATE task_runs SET ended_at=? WHERE task_id=? AND outcome='rate_limited'",
            (now - 60, tid),
        )
        conn.commit()
        assert kbd.rate_limit_circuit_open(conn, now=now, trip=5, window=600) is True
        assert kbd.rate_limit_circuit_open(conn, now=now, trip=6, window=600) is False


def test_circuit_ignores_runs_outside_window():
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-old", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=5, ended_at=now)
        conn.execute(
            "UPDATE task_runs SET ended_at=? WHERE task_id=? AND outcome='rate_limited'",
            (now - 5_000, tid),
        )
        conn.commit()
        assert kbd.rate_limit_circuit_open(conn, now=now, trip=5, window=600) is False


def test_circuit_ignores_non_rate_limited_outcomes():
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-mixed", assignee="a")
        _seed_rate_limited_runs(conn, tid, count=5, ended_at=now)
        conn.execute(
            "UPDATE task_runs SET outcome='crashed', ended_at=? "
            "WHERE task_id=? AND outcome='rate_limited'",
            (now - 60, tid),
        )
        conn.commit()
        assert kbd.rate_limit_circuit_open(conn, now=now, trip=5, window=600) is False


# --------------------------------------------------------------------------
# Config knobs must actually reach the code (inert-knob trap)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key", ["pool_health_url", "pool_min_eligible", "rate_limit_trip"],
)
def test_knob_declared_in_config_defaults(key):
    """A knob absent from DEFAULT_CONFIG makes `hermes config set` warn falsely."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert key in DEFAULT_CONFIG["kanban"], f"kanban.{key} missing from DEFAULT_CONFIG"


def test_knobs_reach_the_code_through_the_real_loader(tmp_path, monkeypatch):
    """THE INERT-KNOB TRAP.

    Built via the PRODUCTION loader, not a constructed config object: a
    hand-built object proves nothing about whether config.yaml actually
    reaches the dispatcher. If this reads back the defaults instead of these
    values, the knobs shipped inert.
    """
    (tmp_path / "config.yaml").write_text(
        "kanban:\n"
        '  pool_health_url: "http://example.invalid/h"\n'
        "  pool_min_eligible: 4\n"
        "  rate_limit_trip: 9\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert kbd._kanban_pool_settings() == ("http://example.invalid/h", 4, 9)


def test_empty_health_url_disables_gate_via_config(tmp_path, monkeypatch):
    """Operator kill switch survives the loader."""
    (tmp_path / "config.yaml").write_text(
        'kanban:\n  pool_health_url: ""\n', encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    url, _min_elig, _trip = kbd._kanban_pool_settings()
    assert url == ""
