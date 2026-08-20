"""Tests for the outbound webhook dispatcher (agent.outbound_webhooks).

Covers config parsing, matcher behaviour, HMAC signing, payload shape,
idempotent registration on the plugin manager, and end-to-end delivery
against a real local HTTP server (no mocks on the network path).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent import outbound_webhooks


# ── helpers ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registration_state():
    _strip_outbound_callbacks()
    outbound_webhooks.reset_for_tests()
    yield
    _strip_outbound_callbacks()
    outbound_webhooks.reset_for_tests()


def _strip_outbound_callbacks():
    """Remove outbound-webhook callbacks from the shared plugin manager.

    ``reset_for_tests`` clears the idempotence set but the manager singleton
    keeps previously-registered callbacks; without this, a target registered
    in one test would fire (real network!) in every later test in this file.
    """
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for event, callbacks in list(manager._hooks.items()):
        manager._hooks[event] = [
            cb for cb in callbacks
            if not getattr(cb, "__name__", "").startswith("outbound_webhook[")
        ]


class _CapturingHandler(BaseHTTPRequestHandler):
    """Records every POST (path, headers, body) on the server instance."""

    def do_POST(self):  # noqa: N802 — http.server naming
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.captured.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "method": "POST",
                "headers": dict(self.headers),
                "body": body,
            }
        )
        status = getattr(self.server, "respond_status", 200)
        self.send_response(status)
        location = getattr(self.server, "respond_location", None)
        if location:
            self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):  # noqa: N802 — records redirect follow-ups
        self.server.captured.append(  # type: ignore[attr-defined]
            {"path": self.path, "method": "GET", "headers": dict(self.headers),
             "body": b""}
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 — http.server naming
        pass


@pytest.fixture()
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    server.captured = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _url(server: HTTPServer, path: str = "/hook") -> str:
    return f"http://127.0.0.1:{server.server_address[1]}{path}"


def _cfg(*entries):
    return {"hooks": {"outbound": list(entries)}}


class _FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


class _Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _RecordingOpener:
    def __init__(self, clock, statuses=None):
        self.clock = clock
        self.statuses = list(statuses or [200])
        self.attempts = []

    def open(self, request, timeout):
        attempt_index = len(self.attempts)
        status = self.statuses[min(attempt_index, len(self.statuses) - 1)]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.attempts.append(
            {
                "url": request.full_url,
                "time": self.clock.monotonic(),
                "timeout": timeout,
                "body": request.data,
                "delivery_id": headers["x-hermes-delivery"],
            }
        )
        return _Response(status)


# ── config parsing ────────────────────────────────────────────────────────


class TestParseConfig:
    def test_missing_block_is_empty(self):
        assert outbound_webhooks.iter_configured_targets({}) == []
        assert outbound_webhooks.iter_configured_targets(None) == []
        assert outbound_webhooks.iter_configured_targets({"hooks": {}}) == []
        assert outbound_webhooks.iter_configured_targets(
            {"hooks": "not-a-dict"}
        ) == []

    def test_non_list_outbound_is_empty(self):
        assert outbound_webhooks.iter_configured_targets(
            {"hooks": {"outbound": {"url": "https://x"}}}
        ) == []

    def test_valid_entry_parses(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_end"],
                    "name": "ci",
                    "timeout": 5,
                }
            )
        )
        assert len(targets) == 1
        t = targets[0]
        assert t.url == "https://example.com/hook"
        assert t.events == ["on_session_end"]
        assert t.name == "ci"
        assert t.timeout == 5

    def test_missing_url_skipped(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"events": ["on_session_end"]})
        ) == []

    def test_non_http_url_skipped(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "ftp://example.com", "events": ["on_session_end"]})
        ) == []
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "file:///etc/passwd", "events": ["on_session_end"]})
        ) == []

    def test_unknown_events_filtered_known_kept(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_end", "not_a_real_event"],
                }
            )
        )
        assert len(targets) == 1
        assert targets[0].events == ["on_session_end"]

    def test_all_unknown_events_skips_entry(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "https://example.com", "events": ["bogus"]})
        ) == []

    def test_empty_events_skips_entry(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "https://example.com", "events": []})
        ) == []

    def test_timeout_clamped(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "timeout": 9999,
                }
            )
        )
        assert targets[0].timeout == outbound_webhooks.MAX_TIMEOUT_SECONDS

    def test_bad_timeout_falls_back_to_default(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "timeout": "soon",
                }
            )
        )
        assert targets[0].timeout == outbound_webhooks.DEFAULT_TIMEOUT_SECONDS

    def test_rate_limit_defaults_and_accepts_positive_int_or_float(self):
        default_target, int_target, float_target = (
            outbound_webhooks.iter_configured_targets(
                _cfg(
                    {"url": f"https://example.com/{index}",
                     "events": ["on_session_end"], **extra},
                )
            )[0]
            for index, extra in enumerate((
                {},
                {"rate_limit_per_second": 25},
                {"rate_limit_per_second": 2.5},
            ))
        )

        assert default_target.rate_limit_per_second == 10.0
        assert int_target.rate_limit_per_second == 25.0
        assert float_target.rate_limit_per_second == 2.5

    @pytest.mark.parametrize(
        "configured",
        [True, False, float("nan"), float("inf"), float("-inf"), 0, -1, "5", None],
    )
    def test_malformed_rate_limit_warns_and_falls_back_to_default(
        self, configured, caplog,
    ):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "rate_limit_per_second": configured,
                }
            )
        )

        assert targets[0].rate_limit_per_second == 10.0
        assert "rate_limit_per_second" in caplog.text

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            (0.01, 0.1),
            (10_000, 1000.0),
            pytest.param(10**1000, 1000.0, id="huge-int"),
        ],
    )
    def test_positive_rate_limit_is_clamped(self, configured, expected):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "rate_limit_per_second": configured,
                }
            )
        )

        assert targets[0].rate_limit_per_second == expected

    def test_matcher_dropped_for_non_tool_events(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "matcher": "terminal",
                }
            )
        )
        assert targets[0].matcher is None

    def test_matcher_kept_for_tool_events(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["post_tool_call"],
                    "matcher": "terminal|delegate_task",
                }
            )
        )
        assert targets[0].matcher == "terminal|delegate_task"

    def test_secret_env_wins_over_literal(self, monkeypatch):
        monkeypatch.setenv("MY_HOOK_SECRET", "from-env")
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "secret_env": "MY_HOOK_SECRET",
                    "secret": "literal",
                }
            )
        )
        assert targets[0].secret == "from-env"

    def test_unset_secret_env_means_unsigned(self, monkeypatch):
        monkeypatch.delenv("MISSING_SECRET_VAR", raising=False)
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "secret_env": "MISSING_SECRET_VAR",
                }
            )
        )
        assert targets[0].secret is None


# ── matcher behaviour ─────────────────────────────────────────────────────


class TestMatcher:
    def _target(self, matcher):
        return outbound_webhooks.WebhookTarget(
            url="https://example.com",
            events=["post_tool_call"],
            matcher=matcher,
        )

    def test_no_matcher_matches_everything(self):
        assert self._target(None).matches_tool("terminal")
        assert self._target(None).matches_tool(None)

    def test_regex_fullmatch(self):
        t = self._target("terminal|delegate_task")
        assert t.matches_tool("terminal")
        assert t.matches_tool("delegate_task")
        assert not t.matches_tool("terminal_extra")
        assert not t.matches_tool(None)

    def test_invalid_regex_falls_back_to_equality(self):
        t = self._target("terminal(")
        assert t.matches_tool("terminal(")
        assert not t.matches_tool("terminal")


# ── payload shape ─────────────────────────────────────────────────────────


class TestPayload:
    def test_top_level_shape_matches_shell_hooks_wire(self):
        body = outbound_webhooks._serialize_payload(
            "post_tool_call",
            {
                "tool_name": "terminal",
                "args": {"command": "ls"},
                "session_id": "sess_1",
                "status": "ok",
                "duration_ms": 42,
            },
            "did_1234",
        )
        payload = json.loads(body)
        assert payload["hook_event_name"] == "post_tool_call"
        assert payload["tool_name"] == "terminal"
        assert payload["tool_input"] == {"command": "ls"}
        assert payload["session_id"] == "sess_1"
        assert payload["extra"]["status"] == "ok"
        assert payload["extra"]["duration_ms"] == 42
        assert payload["delivery_id"] == "did_1234"
        assert payload["timestamp"].endswith("Z")

    def test_unserialisable_values_stringified(self):
        body = outbound_webhooks._serialize_payload(
            "on_session_end", {"weird": object()}, "did_1"
        )
        payload = json.loads(body)
        assert isinstance(payload["extra"]["weird"], str)


# ── registration ──────────────────────────────────────────────────────────


class TestRegistration:
    def test_registration_is_idempotent(self):
        cfg = _cfg(
            {"url": "https://example.com/hook", "events": ["on_session_end"]}
        )
        first = outbound_webhooks.register_from_config(cfg)
        second = outbound_webhooks.register_from_config(cfg)
        assert len(first) == 1
        assert second == []

    def test_safe_mode_skips_registration(self, monkeypatch):
        monkeypatch.setenv("HERMES_SAFE_MODE", "1")
        cfg = _cfg(
            {"url": "https://example.com/hook", "events": ["on_session_end"]}
        )
        assert outbound_webhooks.register_from_config(cfg) == []

    def test_callbacks_never_return_directives(self, http_server):
        """Outbound webhooks are notify-only — invoke_hook must see None."""
        cfg = _cfg({"url": _url(http_server), "events": ["pre_tool_call"]})
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        results = get_plugin_manager().invoke_hook(
            "pre_tool_call", tool_name="terminal", args={"command": "ls"},
        )
        assert outbound_webhooks.flush()
        # No block/context directives from the webhook callback.
        assert results == []
        assert len(http_server.captured) == 1


# ── E2E delivery against a real HTTP server ──────────────────────────────


class TestDelivery:
    def test_default_destination_rate_limit_waits_after_ten_attempts(
        self, monkeypatch,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_rate_limit",
        )

        for _ in range(11):
            outbound_webhooks._deliver(delivery)

        assert len(opener.attempts) == 11
        assert clock.sleeps == [pytest.approx(0.1)]

    def test_at_limit_traffic_does_not_sleep(self, monkeypatch):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_at_limit",
        )

        for _ in range(10):
            outbound_webhooks._deliver(delivery)

        assert len(opener.attempts) == 10
        assert clock.sleeps == []

    def test_above_limit_queue_waits_without_losing_deliveries(
        self, monkeypatch,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        callback = outbound_webhooks._make_callback("on_session_end", target)

        for index in range(11):
            assert callback(session_id=f"session-{index}") is None
        outbound_webhooks._delivery_queue.join()

        assert len(opener.attempts) == 11
        assert clock.sleeps == [pytest.approx(0.1)]
        assert len({attempt["delivery_id"] for attempt in opener.attempts}) == 11
        assert [
            json.loads(attempt["body"])["session_id"]
            for attempt in opener.attempts
        ] == [f"session-{index}" for index in range(11)]

    def test_destination_urls_have_independent_buckets(self, monkeypatch):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target_a = outbound_webhooks.WebhookTarget(
            url="https://a.example.com/hook", events=["on_session_end"],
        )
        target_b = outbound_webhooks.WebhookTarget(
            url="https://b.example.com/hook", events=["on_session_end"],
        )
        delivery_a = outbound_webhooks._build_delivery(
            "on_session_end", target_a, b"{}", "did_a",
        )
        delivery_b = outbound_webhooks._build_delivery(
            "on_session_end", target_b, b"{}", "did_b",
        )

        for _ in range(10):
            outbound_webhooks._deliver(delivery_a)
        outbound_webhooks._deliver(delivery_b)

        assert clock.sleeps == []

        outbound_webhooks._deliver(delivery_a)
        assert clock.sleeps == [pytest.approx(0.1)]

    def test_same_url_across_events_shares_one_bucket(self, monkeypatch):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        session_start = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_start"],
        )
        session_end = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        start_delivery = outbound_webhooks._build_delivery(
            "on_session_start", session_start, b"{}", "did_start",
        )
        end_delivery = outbound_webhooks._build_delivery(
            "on_session_end", session_end, b"{}", "did_end",
        )

        for _ in range(10):
            outbound_webhooks._deliver(start_delivery)
        outbound_webhooks._deliver(end_delivery)

        assert clock.sleeps == [pytest.approx(0.1)]

    def test_conflicting_same_url_rates_keep_strictest_effective_rate(
        self, monkeypatch, caplog,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        fast_target, strict_target = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_start"],
                    "rate_limit_per_second": 100.0,
                },
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_end"],
                    "rate_limit_per_second": 0.5,
                },
            )
        )
        fast_delivery = outbound_webhooks._build_delivery(
            "on_session_start", fast_target, b"{}", "did_fast",
        )
        strict_delivery = outbound_webhooks._build_delivery(
            "on_session_end", strict_target, b"{}", "did_strict",
        )

        assert fast_target.rate_limit_per_second == 0.5
        assert strict_target.rate_limit_per_second == 0.5

        with caplog.at_level("INFO", logger=outbound_webhooks.__name__):
            for _ in range(10):
                outbound_webhooks._deliver(fast_delivery)
            outbound_webhooks._deliver(fast_delivery)
            outbound_webhooks._deliver(strict_delivery)

        assert clock.sleeps == [pytest.approx(2.0), pytest.approx(2.0)]
        assert caplog.text.count("rate=0.5/s") == 2

    def test_destination_bucket_recovers_with_monotonic_time(
        self, monkeypatch,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_refill",
        )

        for _ in range(10):
            outbound_webhooks._deliver(delivery)
        clock.advance(0.2)
        outbound_webhooks._deliver(delivery)
        outbound_webhooks._deliver(delivery)

        assert clock.sleeps == []

        outbound_webhooks._deliver(delivery)
        assert clock.sleeps == [pytest.approx(0.1)]

    def test_retry_attempt_is_rate_limited_after_existing_backoff(
        self, monkeypatch,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock, statuses=[200] * 9 + [500, 200])
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook",
            events=["on_session_end"],
            rate_limit_per_second=0.5,
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_retry",
        )

        for _ in range(9):
            outbound_webhooks._deliver(delivery)
        outbound_webhooks._deliver(delivery)

        assert len(opener.attempts) == 11
        assert clock.sleeps == [
            pytest.approx(outbound_webhooks.RETRY_BACKOFF_SECONDS),
            pytest.approx(1.0),
        ]

    def test_reset_for_tests_clears_destination_buckets(self, monkeypatch):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook", events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_reset",
        )

        for _ in range(10):
            outbound_webhooks._deliver(delivery)
        outbound_webhooks.reset_for_tests()
        outbound_webhooks._deliver(delivery)

        assert clock.sleeps == []

    def test_throttle_info_log_is_emitted_only_when_waiting(
        self, monkeypatch, caplog,
    ):
        clock = _FakeClock()
        opener = _RecordingOpener(clock)
        monkeypatch.setattr(outbound_webhooks, "time", clock)
        monkeypatch.setattr(outbound_webhooks, "_opener", opener)
        target = outbound_webhooks.WebhookTarget(
            url="https://example.com/hook",
            events=["on_session_end"],
            name="named-target",
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_log",
        )

        with caplog.at_level("INFO", logger=outbound_webhooks.__name__):
            for _ in range(10):
                outbound_webhooks._deliver(delivery)
            assert "outbound webhook throttled" not in caplog.text
            outbound_webhooks._deliver(delivery)

        assert "target=named-target" in caplog.text
        assert "rate=10/s" in caplog.text
        assert "wait=0.100s" in caplog.text
        assert "queue_depth=" in caplog.text

    def test_delivery_with_hmac_signature(self, http_server):
        secret = "s3cret"
        cfg = _cfg(
            {
                "url": _url(http_server),
                "events": ["on_session_end"],
                "secret": secret,
                "name": "e2e",
            }
        )
        registered = outbound_webhooks.register_from_config(cfg)
        assert len(registered) == 1

        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().invoke_hook(
            "on_session_end",
            session_id="sess_e2e",
            completed=True,
            interrupted=False,
            model="test-model",
            platform="cli",
        )
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        req = http_server.captured[0]

        payload = json.loads(req["body"])
        assert payload["hook_event_name"] == "on_session_end"
        assert payload["session_id"] == "sess_e2e"
        assert payload["extra"]["completed"] is True
        assert payload["extra"]["model"] == "test-model"

        assert req["headers"]["X-Hermes-Event"] == "on_session_end"
        assert req["headers"]["X-Hermes-Delivery"]
        expected = hmac.new(
            secret.encode(), req["body"], hashlib.sha256
        ).hexdigest()
        assert req["headers"]["X-Hermes-Signature-256"] == f"sha256={expected}"

    def test_unsigned_delivery_has_no_signature_header(self, http_server):
        cfg = _cfg({"url": _url(http_server), "events": ["on_session_end"]})
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().invoke_hook("on_session_end", session_id="s")
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        assert "X-Hermes-Signature-256" not in http_server.captured[0]["headers"]

    def test_matcher_filters_tool_events(self, http_server):
        cfg = _cfg(
            {
                "url": _url(http_server),
                "events": ["post_tool_call"],
                "matcher": "terminal",
            }
        )
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        manager = get_plugin_manager()
        manager.invoke_hook(
            "post_tool_call", tool_name="web_search", args={}, status="ok",
        )
        manager.invoke_hook(
            "post_tool_call", tool_name="terminal", args={}, status="ok",
        )
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        payload = json.loads(http_server.captured[0]["body"])
        assert payload["tool_name"] == "terminal"

    def test_4xx_not_retried(self, http_server):
        http_server.respond_status = 400
        target = outbound_webhooks.WebhookTarget(
            url=_url(http_server), events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_4xx",
        )
        outbound_webhooks._deliver(delivery)
        assert len(http_server.captured) == 1

    def test_5xx_retried_once(self, http_server):
        http_server.respond_status = 500
        target = outbound_webhooks.WebhookTarget(
            url=_url(http_server), events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_5xx",
        )
        outbound_webhooks._deliver(delivery)
        assert len(http_server.captured) == outbound_webhooks.MAX_DELIVERY_ATTEMPTS

    def test_redirect_not_followed(self, http_server):
        """3xx must be treated as failure — urllib's default handler would
        convert a 302'd POST into a body-less GET at the redirect target."""
        http_server.respond_status = 302
        http_server.respond_location = _url(http_server, "/redirected")
        target = outbound_webhooks.WebhookTarget(
            url=_url(http_server), events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_3xx",
        )
        outbound_webhooks._deliver(delivery)
        # One POST hit the server (the redirect response), nothing followed
        # (no GET to /redirected), no retry (misconfiguration, not transient).
        assert [c["method"] for c in http_server.captured] == ["POST"]
        assert http_server.captured[0]["path"] == "/hook"

    def test_delivery_id_matches_header_and_body(self, http_server):
        """The X-Hermes-Delivery header and the signed body's delivery_id
        must be the same value, or receiver-side dedupe breaks."""
        cfg = _cfg(
            {"url": _url(http_server), "events": ["on_session_end"],
             "secret": "s"}
        )
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().invoke_hook("on_session_end", session_id="s1")
        assert outbound_webhooks.flush()

        req = http_server.captured[0]
        payload = json.loads(req["body"])
        assert payload["delivery_id"] == req["headers"]["X-Hermes-Delivery"]

    def test_connection_error_does_not_raise(self):
        target = outbound_webhooks.WebhookTarget(
            # Port 9 (discard) — nothing listening.
            url="http://127.0.0.1:9/unreachable",
            events=["on_session_end"],
            timeout=1,
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}", "did_conn",
        )
        # Must swallow the failure (logged), never raise into the agent loop.
        outbound_webhooks._deliver(delivery)

    def test_events_enqueued_at_exit_still_delivered(self, http_server, tmp_path):
        """A short-lived process (`hermes chat -q`, cron) exits right after
        firing on_session_end.  The delivery worker is a daemon thread, so
        without the atexit flush the final event is silently dropped."""
        import subprocess
        import sys as _sys

        cfg = {"hooks": {"outbound": [
            {"url": _url(http_server), "events": ["on_session_end"]}
        ]}}
        script = tmp_path / "fire_and_exit.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {repr(str(Path(outbound_webhooks.__file__).resolve().parents[1]))})\n"
            "from agent import outbound_webhooks\n"
            "from hermes_cli.plugins import get_plugin_manager\n"
            f"cfg = {repr(cfg)}\n"
            "outbound_webhooks.register_from_config(cfg)\n"
            "get_plugin_manager().invoke_hook('on_session_end', session_id='exit_test')\n"
            "# exit immediately — no explicit flush\n"
        )
        proc = subprocess.run(
            [_sys.executable, str(script)], capture_output=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not http_server.captured:
            time.sleep(0.05)
        assert len(http_server.captured) == 1
        payload = json.loads(http_server.captured[0]["body"])
        assert payload["session_id"] == "exit_test"
