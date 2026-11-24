"""Re-probe behaviour when the primary stays throttled for hours.

A 1-token probe decides whether the primary is back before a full request is built, and each
failed probe lengthens the next interval (persisted in the shared ledger, so the streak is not
per-process)."""
import json
import time
from types import SimpleNamespace

import pytest

from agent import fallback_cooldown, model_health_ledger


KEY = "nous/qwen/qwen4-max:free"


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "model_windows.json"
    monkeypatch.setattr(model_health_ledger, "_state_path", lambda: str(path))
    return path


def _seed(path, failures=0, reset_in=-10.0, hits=0):
    now = time.time()
    path.write_text(json.dumps({KEY: {
        "reset_at": now + reset_in, "recorded_at": now, "seconds": max(0.0, reset_in),
        "hits": hits, "failures": failures,
    }}))


def test_failed_probe_streak_survives_expiry(ledger):
    _seed(ledger, failures=2, reset_in=-10.0)
    # An expired window still reports no cooldown, but the streak that widened the backoff remains.
    assert model_health_ledger.model_window_remaining("nous", "qwen/qwen4-max:free") == 0.0
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 2

    # A later window must not erase the streak.
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 60)
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 2

    model_health_ledger.clear_model_window("nous", "qwen/qwen4-max:free")
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 0


def test_probe_failures_increment_streak_and_hits(ledger):
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 30, probe_failure=True)
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 30, probe_failure=True)
    snapshot = model_health_ledger.model_windows_snapshot()[KEY]
    assert snapshot["failures"] == 2
    assert snapshot["hits"] == 2  # a probe failure is a real recorded 429


def test_backoff_doubles_and_caps():
    assert fallback_cooldown.reprobe_backoff(300, 0) == 300
    assert fallback_cooldown.reprobe_backoff(300, 1) == 600
    assert fallback_cooldown.reprobe_backoff(300, 3) == 2400
    assert fallback_cooldown.reprobe_backoff(300, 20) == fallback_cooldown.MAX_REPROBE_SECONDS
    assert fallback_cooldown.reprobe_backoff(0, 5) == 0


def _fallback_agent():
    return SimpleNamespace(
        _provider_fallback_active=True,
        _fallback_activated=True,
        model="Qwen3.8-27B-oQ6e-mtp",
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        api_mode="chat_completions",
        _primary_runtime={
            "provider": "nous",
            "model": "qwen/qwen4-max:free",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_mode": "chat_completions",
            "api_key": "test-key",
        },
        _fallback_reprobe_at=0.0,
        _rate_limited_until=0.0,
    )


def test_throttled_probe_does_not_restore_and_backs_off(ledger, monkeypatch):
    agent = _fallback_agent()
    calls = {"restore": 0}
    monkeypatch.setattr(fallback_cooldown, "probe_primary_route", lambda _a: (False, 120.0))
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.restore_primary_runtime",
        lambda _a: calls.__setitem__("restore", calls["restore"] + 1) or True,
    )

    assert fallback_cooldown.maybe_reprobe_primary(agent) is False
    assert calls["restore"] == 0, "a throttled primary must not be restored"
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 1
    assert model_health_ledger.model_window_remaining("nous", "qwen/qwen4-max:free") > 100
    # Base 300 s doubling for streak=1, jittered +-30%.
    assert agent._fallback_reprobe_at > time.monotonic() + 400


def test_successful_probe_restores_but_parks_the_streak(ledger, monkeypatch):
    """Production thrash: probe 11:51:47 OK, real request 11:51:50 refused.

    The probe restore may not erase the streak: otherwise every bad restore starts the next
    probe from the base interval and the lane flaps between primary and fallback all day.
    """
    _seed(ledger, failures=2, reset_in=-10.0)  # streak present, window already closed
    agent = _fallback_agent()
    monkeypatch.setattr(fallback_cooldown, "probe_primary_route", lambda _a: (True, None))
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", lambda _a: True)

    assert fallback_cooldown.maybe_reprobe_primary(agent) is True
    # Told to callers as available: nothing gates on the old window any more...
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 0
    assert model_health_ledger.model_window_remaining("nous", "qwen/qwen4-max:free") == 0.0
    # ...but the verdict is provisional and remembers what the probe failed to disprove.
    assert json.loads(ledger.read_text())[KEY]["probe_streak"] == 2
    assert agent._fallback_reprobe_at > time.monotonic()

    # The real request is refused seconds later: resume at 3, not restart at 1 (base interval).
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 510)
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 3
    assert "probe_at" not in json.loads(ledger.read_text())[KEY]


def test_false_positive_streak_widens_the_next_probe(ledger, monkeypatch):
    """End to end through the re-probe: a refused restore must not re-arm at the base interval."""
    _seed(ledger, failures=2, reset_in=-10.0)
    agent = _fallback_agent()
    monkeypatch.setattr(fallback_cooldown, "probe_primary_route", lambda _a: (True, None))
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", lambda _a: True)
    assert fallback_cooldown.maybe_reprobe_primary(agent) is True

    # The restored primary is refused on the very next call → streak 3, shared window 510 s.
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 510)
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 3

    # That window ends; the next probe fails → streak 4 → 300*2^4 s, not the 600 s of streak=1.
    entry = json.loads(ledger.read_text())[KEY]
    entry["reset_at"] = time.time() - 1
    ledger.write_text(json.dumps({KEY: entry}))
    monkeypatch.setattr(fallback_cooldown, "probe_primary_route", lambda _a: (False, 30.0))
    agent._fallback_reprobe_at = 0.0
    assert fallback_cooldown.maybe_reprobe_primary(agent) is False
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 4
    assert agent._fallback_reprobe_at > time.monotonic() + 2000


def test_failure_after_the_verify_window_starts_a_fresh_streak(ledger):
    """A marker older than PROBE_VERIFY_SECONDS is no longer evidence about this failure."""
    model_health_ledger.mark_probe_restore("nous", "qwen/qwen4-max:free", 3)
    entry = json.loads(ledger.read_text())[KEY]
    entry["probe_at"] -= model_health_ledger.PROBE_VERIFY_SECONDS + 1
    ledger.write_text(json.dumps({KEY: entry}))

    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 60)
    # A stale marker proves nothing, so this starts a fresh streak (0 failed probes, base interval).
    assert model_health_ledger.model_failure_streak("nous", "qwen/qwen4-max:free") == 0
    assert "probe_at" not in json.loads(ledger.read_text())[KEY]


def test_shared_window_still_defers_before_probing(ledger, monkeypatch):
    agent = _fallback_agent()
    model_health_ledger.record_model_window("nous", "qwen/qwen4-max:free", 90)
    probed = {"n": 0}
    monkeypatch.setattr(
        fallback_cooldown, "probe_primary_route",
        lambda _a: probed.__setitem__("n", probed["n"] + 1) or (True, None),
    )

    assert fallback_cooldown.maybe_reprobe_primary(agent) is False
    assert probed["n"] == 0, "an open shared window must not be probed"
    assert agent._fallback_reprobe_at > time.monotonic() + 60


def test_probe_parses_retry_after_and_tolerates_other_errors():
    """The probe must read Retry-After off a real 429 and not treat a 400 as 'throttled'."""
    import http.server
    import threading

    responses = [
        (429, {"Retry-After": "7", "Content-Type": "application/json"}, {"error": "rate_limited"}),
        (200, {"Content-Type": "application/json"},
         {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}),
        (400, {"Content-Type": "application/json"}, {"error": "bad model"}),
    ]
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            status, headers, payload = responses[min(len(seen), len(responses) - 1)]
            seen.append(status)
            body = json.dumps(payload).encode()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = SimpleNamespace(_primary_runtime={
            "provider": "nous", "model": "qwen/qwen4-max:free",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "api_mode": "chat_completions", "api_key": "test-key",
        })
        assert fallback_cooldown.probe_primary_route(agent) == (False, 7.0)
        assert fallback_cooldown.probe_primary_route(agent) == (True, None)
        # A non-429 error is inconclusive, not evidence of throttling.
        assert fallback_cooldown.probe_primary_route(agent) == (True, None)
    finally:
        server.shutdown()
