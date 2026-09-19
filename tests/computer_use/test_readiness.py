"""Behavior contracts for bounded readiness checks (RFC #112639)."""

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.readiness import ReadinessResult, verify_readiness


class _StubSession:
    def __init__(self, advertised=True):
        self.advertised = advertised

    def _has_tool(self, name):
        return self.advertised and name == "verify_state"


class _StubBackend:
    def __init__(self, response=None, exc=None, advertised=True):
        self._session = _StubSession(advertised)
        self.calls = []
        self._response = response if response is not None else {"structuredContent": {"status": "satisfied"}}
        self._exc = exc

    def call_tool(self, name, args=None, timeout=30.0):
        self.calls.append({"name": name, "args": dict(args or {}), "timeout": timeout})
        if self._exc is not None:
            raise self._exc
        return self._response


def _expect():
    return [{"window": {"exists": True}}]


def test_not_advertised_returns_unknown_without_driver_call():
    backend = _StubBackend(advertised=False)
    out = verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert out.status == "unknown"
    assert "not advertised" in out.detail
    assert backend.calls == []


def test_session_without_capability_probe_returns_unknown():
    backend = _StubBackend()
    backend._session = object()  # no _has_tool at all
    out = verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert out.status == "unknown"
    assert backend.calls == []


@pytest.mark.parametrize("bad", [None, "x", [], [{"window": {"exists": True}}] * 9, ["not-a-dict"]])
def test_expect_validation(bad):
    with pytest.raises(ValueError):
        verify_readiness(_StubBackend(), pid=1, window_id=2, expect=bad)


def test_bounds_clamped_to_driver_contract():
    backend = _StubBackend()
    verify_readiness(backend, pid=1, window_id=2, expect=_expect(), timeout_ms=99999, stable_samples=9)
    payload = backend.calls[0]["args"]
    assert payload["timeout_ms"] == 10000
    assert payload["stable_samples"] == 5
    # Transport timeout covers the driver-side deadline.
    assert backend.calls[0]["timeout"] >= 10000 / 1000.0

    backend = _StubBackend()
    verify_readiness(backend, pid=1, window_id=2, expect=_expect(), timeout_ms=-5, stable_samples=0)
    payload = backend.calls[0]["args"]
    assert payload["timeout_ms"] == 0
    assert payload["stable_samples"] == 1


@pytest.mark.parametrize("driver_status", ["satisfied", "unsatisfied", "unknown"])
def test_driver_status_mapping(driver_status):
    backend = _StubBackend(response={"structuredContent": {
        "status": driver_status,
        "predicates": [{"index": 0, "status": driver_status}],
    }})
    out = verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert out.status == driver_status
    assert isinstance(out, ReadinessResult)
    assert out.duration_ms >= 0
    assert out.raw["status"] == driver_status


@pytest.mark.parametrize("response", [
    {"structuredContent": {}},
    {"structuredContent": {"status": "bogus"}},
    {"data": "no structured content"},
])
def test_missing_or_bogus_status_maps_unknown(response):
    backend = _StubBackend(response=response)
    out = verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert out.status == "unknown"


def test_transport_error_maps_error_with_single_attempt():
    backend = _StubBackend(exc=RuntimeError("Connection refused"))
    out = verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert out.status == "error"
    assert "Connection refused" in out.detail
    assert len(backend.calls) == 1  # never retried


def test_include_screenshot_defaults_false():
    backend = _StubBackend()
    verify_readiness(backend, pid=1, window_id=2, expect=_expect())
    assert backend.calls[0]["args"]["include_screenshot"] is False

    backend = _StubBackend()
    verify_readiness(backend, pid=1, window_id=2, expect=_expect(), include_screenshot=True)
    assert backend.calls[0]["args"]["include_screenshot"] is True


def test_backend_method_routes_through_call_tool_with_session():
    seen = {}

    class _RecordingSession(_StubSession):
        def call_tool(self, name, payload, timeout=30.0):
            seen.update(payload)
            return {"structuredContent": {"status": "satisfied"}}

    backend = object.__new__(CuaDriverBackend)
    backend._session = _RecordingSession(advertised=True)
    backend._session_id = "sess-123"
    out = backend.verify_readiness(pid=7, window_id=8, expect=_expect(), timeout_ms=500)
    assert out.status == "satisfied"
    assert seen["session"] == "sess-123"  # Hermes-authorized path, session injected
    assert seen["pid"] == 7 and seen["window_id"] == 8
    assert seen["timeout_ms"] == 500
