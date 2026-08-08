"""Tests for tools/mcp_trace_propagation.py — opt-in W3C trace propagation.

Covers the three properties the feature promises:

1. Gate — everything is a no-op unless ``mcp.trace_propagation`` is true.
2. Capture — the caller's context is read on the calling thread via the
   standard propagation API (or a registered provider), validated against
   the W3C grammar, and never raises.
3. Injection — the header exists on the shared client exactly for the
   duration of one RPC, restores any pre-existing value, and no-ops for
   stdio transports (client=None).

Plus the design-rationale test: capture on a different thread than the one
owning the span yields nothing, which is why capture must happen before the
call crosses onto the MCP daemon loop.

No live network calls; the OpenTelemetry API is faked through the provider
hook and a stub ``opentelemetry`` module so the suite passes with or
without the real SDK installed.
"""

import sys
import threading
import types

import pytest

from tools import mcp_trace_propagation as mtp


VALID_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    """Isolate HERMES_HOME, clear any registered provider, default gate off."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mtp.register_traceparent_provider(None)
    yield
    mtp.register_traceparent_provider(None)


def _enable(monkeypatch, enabled=True):
    monkeypatch.setattr(
        mtp, "is_enabled", lambda: enabled,
    )


def _fake_config(monkeypatch, cfg):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestGate:
    def test_disabled_by_default(self, monkeypatch):
        _fake_config(monkeypatch, {})
        assert mtp.is_enabled() is False

    def test_enabled_via_config(self, monkeypatch):
        _fake_config(monkeypatch, {"mcp": {"trace_propagation": True}})
        assert mtp.is_enabled() is True

    def test_null_mcp_section_is_off(self, monkeypatch):
        # A config carrying `mcp:` with no body loads as {"mcp": None}.
        _fake_config(monkeypatch, {"mcp": None})
        assert mtp.is_enabled() is False

    def test_config_error_means_off(self, monkeypatch):
        import hermes_cli.config as config_mod

        def boom():
            raise RuntimeError("unreadable config")

        monkeypatch.setattr(config_mod, "load_config_readonly", boom)
        assert mtp.is_enabled() is False

    def test_disabled_gate_short_circuits_even_with_a_provider(self, monkeypatch):
        _enable(monkeypatch, False)
        mtp.register_traceparent_provider(lambda: VALID_TP)
        assert mtp.current_traceparent() is None


# ---------------------------------------------------------------------------
# Capture — provider path
# ---------------------------------------------------------------------------

class TestProviderCapture:
    def test_valid_provider_output_is_used(self, monkeypatch):
        _enable(monkeypatch)
        mtp.register_traceparent_provider(lambda: VALID_TP)
        assert mtp.current_traceparent() == VALID_TP

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "",
            "not-a-traceparent",
            "01-" + "a" * 32 + "-" + "b" * 16 + "-01",   # unknown version
            "00-" + "A" * 32 + "-" + "b" * 16 + "-01",   # uppercase hex
            "00-" + "0" * 32 + "-" + "b" * 16 + "-01",   # all-zero trace-id
            "00-" + "a" * 32 + "-" + "0" * 16 + "-01",   # all-zero parent-id
            "00-" + "a" * 32 + "-" + "b" * 16 + "-01; evil: header",
            42,
        ],
    )
    def test_malformed_provider_output_is_discarded(self, monkeypatch, bad):
        _enable(monkeypatch)
        mtp.register_traceparent_provider(lambda: bad)
        assert mtp.current_traceparent() is None

    def test_raising_provider_is_survived(self, monkeypatch):
        _enable(monkeypatch)

        def boom():
            raise RuntimeError("plugin bug")

        mtp.register_traceparent_provider(boom)
        assert mtp.current_traceparent() is None


# ---------------------------------------------------------------------------
# Capture — standard propagation API path (stubbed opentelemetry)
# ---------------------------------------------------------------------------

class TestAmbientCapture:
    def _stub_otel(self, monkeypatch, inject):
        stub = types.ModuleType("opentelemetry")
        stub.propagate = types.SimpleNamespace(inject=inject)
        monkeypatch.setitem(sys.modules, "opentelemetry", stub)

    def test_ambient_context_is_captured(self, monkeypatch):
        _enable(monkeypatch)
        self._stub_otel(
            monkeypatch,
            lambda carrier: carrier.__setitem__("traceparent", VALID_TP),
        )
        assert mtp.current_traceparent() == VALID_TP

    def test_no_active_span_yields_none(self, monkeypatch):
        _enable(monkeypatch)
        self._stub_otel(monkeypatch, lambda carrier: None)  # injects nothing
        assert mtp.current_traceparent() is None

    def test_missing_sdk_yields_none(self, monkeypatch):
        _enable(monkeypatch)
        monkeypatch.setitem(sys.modules, "opentelemetry", None)  # ImportError
        assert mtp.current_traceparent() is None

    def test_capture_on_the_wrong_thread_sees_nothing(self, monkeypatch):
        """The design constraint: ambient context is thread-local. Capturing
        on any thread but the span's own yields nothing — which is why the
        tool handler captures BEFORE the call crosses to the MCP loop."""
        _enable(monkeypatch)
        span_thread = threading.current_thread()

        def thread_local_inject(carrier):
            if threading.current_thread() is span_thread:
                carrier["traceparent"] = VALID_TP

        self._stub_otel(monkeypatch, thread_local_inject)

        assert mtp.current_traceparent() == VALID_TP  # span's own thread

        seen_elsewhere = []
        other = threading.Thread(
            target=lambda: seen_elsewhere.append(mtp.current_traceparent())
        )
        other.start()
        other.join()
        assert seen_elsewhere == [None]  # the daemon thread would see this


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

class _FakeClient:
    """Just the surface injected_headers touches: a headers mapping."""

    def __init__(self, headers=None):
        self.headers = dict(headers or {})


class TestInjectedHeaders:
    def test_header_exists_exactly_for_the_duration_of_the_block(self):
        client = _FakeClient()
        with mtp.injected_headers(client, VALID_TP):
            assert client.headers["traceparent"] == VALID_TP
        assert "traceparent" not in client.headers

    def test_header_is_removed_even_when_the_rpc_raises(self):
        client = _FakeClient()
        with pytest.raises(RuntimeError):
            with mtp.injected_headers(client, VALID_TP):
                raise RuntimeError("transport dropped")
        assert "traceparent" not in client.headers

    def test_preexisting_header_is_restored_not_dropped(self):
        client = _FakeClient({"traceparent": "00-" + "c" * 32 + "-" + "d" * 16 + "-00"})
        with mtp.injected_headers(client, VALID_TP):
            assert client.headers["traceparent"] == VALID_TP
        assert client.headers["traceparent"].startswith("00-cccc")

    def test_stdio_transport_is_a_noop(self):
        with mtp.injected_headers(None, VALID_TP):
            pass  # nothing to assert — it must simply not raise

    def test_no_traceparent_is_a_noop(self):
        client = _FakeClient({"x": "y"})
        with mtp.injected_headers(client, None):
            assert client.headers == {"x": "y"}
        assert client.headers == {"x": "y"}
