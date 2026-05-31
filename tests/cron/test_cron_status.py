"""Tests for `hermes cron status` ticker-health reporting (hermes_cli.cron)."""

import cron.jobs
import hermes_cli.cron as c
import hermes_cli.gateway as gw


def _no_jobs(monkeypatch):
    monkeypatch.setattr(cron.jobs, "list_jobs", lambda include_disabled=False: [])


def test_status_healthy_ticker(monkeypatch, capsys):
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("running", 5.0, None))
    c.cron_status()
    out = capsys.readouterr().out.lower()
    assert "healthy" in out


def test_status_dead_ticker_warns_loudly(monkeypatch, capsys):
    """Gateway up but ticker hasn't ticked in hours: must NOT claim jobs fire."""
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("running", 99999.0, None))
    c.cron_status()
    out = capsys.readouterr().out
    assert "NOT" in out
    assert "hermes gateway restart" in out


def test_status_running_but_stale_beyond_liveness_threshold_is_not_healthy(monkeypatch, capsys):
    """state="running" with an age past the liveness backstop (e.g. the
    supervisor hasn't yet written "stopped") must read NOT-alive, not green.
    Pins the 180s threshold: a 300s-stale ticker is dead, not healthy."""
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("running", 300.0, None))
    c.cron_status()
    out = capsys.readouterr().out
    assert "NOT" in out
    assert "healthy" not in out.lower()


def test_status_running_within_liveness_threshold_is_healthy(monkeypatch, capsys):
    """A running ticker just under the threshold stays green — guards against
    the threshold being set so tight it false-alarms on a slightly slow tick."""
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("running", 120.0, None))
    c.cron_status()
    assert "healthy" in capsys.readouterr().out.lower()


def test_status_failed_import(monkeypatch, capsys):
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("failed_import", None, "SyntaxError: bad"))
    c.cron_status()
    out = capsys.readouterr().out.lower()
    assert "failed to start" in out
    assert "syntaxerror" in out


def test_status_unknown_health(monkeypatch, capsys):
    """Gateway up but no ticker block yet (e.g. just started): honest 'unknown'."""
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: (None, None, None))
    c.cron_status()
    out = capsys.readouterr().out.lower()
    assert "unknown" in out


def test_status_stopped_ticker_is_not_healthy(monkeypatch, capsys):
    """A cleanly-stopped ticker (thread gone, gateway up) must not read healthy
    even when the last tick was recent — jobs are NOT firing."""
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(c, "_read_cron_ticker_health", lambda: ("stopped", 3.0, None))
    c.cron_status()
    out = capsys.readouterr().out
    assert "NOT firing" in out
    assert "healthy" not in out.lower()


def test_read_health_parses_real_status_block(tmp_path, monkeypatch):
    """End-to-end: a real cron_ticker block written by gateway.status round-trips
    through the reader, yielding a parsed state + finite age."""
    import gateway.status as status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_cron_ticker_status(state="running", last_tick_at=status._utc_now_iso())
    state, age, err = c._read_cron_ticker_health()
    assert state == "running"
    assert age is not None and age < 60
    assert err is None


def test_read_health_degrades_on_bad_timestamp(tmp_path, monkeypatch):
    """A malformed last_tick_at must degrade to unknown, never crash the CLI."""
    import gateway.status as status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_cron_ticker_status(state="running", last_tick_at="not-a-timestamp")
    assert c._read_cron_ticker_health() == (None, None, None)


def test_status_gateway_not_running(monkeypatch, capsys):
    _no_jobs(monkeypatch)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda: [])
    c.cron_status()
    out = capsys.readouterr().out
    assert "not running" in out.lower()
    assert "NOT fire" in out
