"""``_cached_tool_availability`` serves the banner's display-only availability from a TTL cache.

Inline, ``compute_toolset_availability`` runs live ``check_fn`` network probes (measured 28–36s on a
slow-egress host) inside ``session.resume`` and blew the Desktop's fixed 30s RPC budget on every
resume of an affected profile. The cache must: never block the caller (cold cache answers ``{}``
while a background thread refreshes), keep serving the warm value within the TTL, refresh in the
background after expiry, and honour ``HERMES_TOOL_AVAILABILITY_TTL_S``.
"""

from __future__ import annotations

import time

import gateway.session_local_inventory as sli


def _reset(monkeypatch, probe_result=None, probe_delay=0.0):
    monkeypatch.setattr(sli, "_availability_cache", {})
    monkeypatch.setattr(sli, "_availability_inflight", set())

    calls = []

    def fake_compute(enabled):
        calls.append(list(enabled))
        if probe_delay:
            time.sleep(probe_delay)
        if isinstance(probe_result, Exception):
            raise probe_result
        return {"unavailable_toolsets": list(enabled)}

    from hermes_cli import banner

    monkeypatch.setattr(banner, "compute_toolset_availability", fake_compute)
    return calls


def test_cold_cache_answers_empty_without_blocking(monkeypatch):
    """Cold cache must return {} immediately — the background probe is slow (28–36s measured)."""
    calls = _reset(monkeypatch, probe_delay=3600)  # would hang the test if awaited

    start = time.monotonic()
    out = sli._cached_tool_availability(["web", "image_gen"])
    elapsed = time.monotonic() - start

    assert out == {}
    assert elapsed < 5, "cold cache blocked on the live probe"
    assert calls == [["image_gen", "web"]]  # probe started in the background (sorted key)


def test_warm_cache_serves_value_without_reprobing(monkeypatch):
    """Within the TTL the warm value is served and no probe runs."""
    _reset(monkeypatch)
    sli._availability_cache[("image_gen", "web")] = (time.time(), {"unavailable_toolsets": ["web"]})

    out = sli._cached_tool_availability(["web", "image_gen"])

    assert out == {"unavailable_toolsets": ["web"]}


def test_expired_cache_refreshes_in_background_and_keeps_serving_stale(monkeypatch):
    """Past the TTL the caller still gets the stale value instantly; refresh runs in background."""
    monkeypatch.setenv("HERMES_TOOL_AVAILABILITY_TTL_S", "0.2")
    _reset(monkeypatch, probe_delay=0.3)
    stale = {"unavailable_toolsets": ["old"]}
    sli._availability_cache[("web",)] = (time.time() - 10, stale)
    sli._availability_inflight.clear()

    out = sli._cached_tool_availability(["web"])
    assert out is stale, "expired entry must serve the stale value, not block"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sli._availability_cache[("web",)][1] is stale:
        time.sleep(0.05)
    refreshed = sli._availability_cache[("web",)][1]
    assert refreshed is not stale and refreshed == {"unavailable_toolsets": ["web"]}


def test_failed_probe_keeps_warm_value_and_clears_inflight(monkeypatch):
    """A crashed probe must neither lose the warm value nor wedge the key as permanently inflight."""
    _reset(monkeypatch, probe_result=RuntimeError("probe exploded"))
    warm = {"unavailable_toolsets": ["web"]}
    sli._availability_cache[("web",)] = (time.time(), warm)

    out = sli._cached_tool_availability(["web"])
    assert out is warm

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and ("web",) in sli._availability_inflight:
        time.sleep(0.05)
    assert ("web",) not in sli._availability_inflight
    assert sli._availability_cache[("web",)][1] is warm
