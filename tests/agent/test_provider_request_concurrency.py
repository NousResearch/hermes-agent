"""Focused tests for provider-level model-REQUEST concurrency.

Verifies that ``provider_max_concurrent_requests`` (a global REQUEST-level
throttle) and ``provider_request_concurrency`` (per-provider overrides) — both
independent of the AGENT-level ``delegation.max_concurrent_children`` cap —
serialize concurrent provider model requests when configured and are
transparent pass-throughs when unset.

Concurrency is asserted via wall-clock timing, not merely "calls happened", so
that a subtle sequential fallback cannot masquerade as concurrency.
"""

import threading
import time

from unittest.mock import patch

from agent import chat_completion_helpers as m


class _Agent:
    def __init__(self, provider=None):
        self.provider = provider


def _spawn(wrapper, n=3, provider=None):
    results = [None] * n
    threads = []
    for i in range(n):
        t = threading.Thread(
            target=lambda i=i: results.__setitem__(
                i, wrapper(_Agent(provider), {"i": i}, make_client=None)
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=15)
    return results


def test_wrapper_is_pass_through_when_no_limit():
    """With no limit configured (global or per-provider) the wrapper must call
    the unlocked body and return its result unchanged."""
    called = []

    def unlocked(agent, api_kwargs, *, make_client):
        called.append(api_kwargs)
        return "result"

    with patch.object(m, "_get_request_semaphore", return_value=None), \
         patch.object(m, "_dispatch_nonstreaming_api_request_unlocked", side_effect=unlocked):
        out = m._dispatch_nonstreaming_api_request(_Agent("some_provider"), {"k": 1}, make_client="mc")
        assert out == "result"
        assert called == [{"k": 1}]


def test_semaphore_none_when_config_unset_or_bad():
    """Invalid / unset values must yield a no-op (None) semaphore for every
    provider, preserving unlimited behavior."""
    for bad in (None, 0, -3, "5", True):
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={
                       "provider_max_concurrent_requests": bad,
                       "provider_request_concurrency": {},
                   }):
            assert m._get_request_semaphore(None) is None
            assert m._get_request_semaphore("x") is None


def test_limit_serializes_concurrent_requests():
    """With a limit of 1, N parallel worker calls for the same provider must
    run sequentially (serialized by the request semaphore). With limit 3 they
    overlap."""

    def run(limit, provider="p"):
        def sleeper(agent, api_kwargs, *, make_client):
            time.sleep(0.25)  # simulate a blocking provider HTTP round-trip
            return api_kwargs

        before = time.monotonic()
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={
                       "provider_max_concurrent_requests": limit,
                       "provider_request_concurrency": {},
                   }), \
             patch.object(m, "_dispatch_nonstreaming_api_request_unlocked", side_effect=sleeper):
            results = _spawn(m._dispatch_nonstreaming_api_request, n=3, provider=provider)
        wall = time.monotonic() - before
        return wall, results

    # limit=1 => 3 sequential sleeps of 0.25s => clearly >= 0.6s
    m._request_semaphores.clear()
    wall1, r1 = run(1)
    assert r1 == [{"i": 0}, {"i": 1}, {"i": 2}], r1
    assert wall1 >= 0.6, f"expected serialization, got {wall1:.2f}s"

    # limit=3 => 3 overlapping sleeps => clearly < 0.6s
    m._request_semaphores.clear()
    wall3, r3 = run(3)
    assert r3 == [{"i": 0}, {"i": 1}, {"i": 2}], r3
    assert wall3 < 0.6, f"expected overlap, got {wall3:.2f}s"


def test_provider_override_wins_over_global():
    """A per-provider entry overrides the global limit for that provider, while
    other providers keep the global limit."""
    cfg = {
        "provider_max_concurrent_requests": 4,
        "provider_request_concurrency": {"alpha": 1, "beta": None},
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        # Provider 'alpha' -> per-provider limit 1 (own semaphore).
        sa = m._get_request_semaphore("alpha")
        assert sa is not None and sa._value == 1
        # Provider 'beta' -> explicit None override -> unlimited.
        assert m._get_request_semaphore("beta") is None
        # Unknown provider / no override -> global limit 4.
        sg = m._get_request_semaphore("gamma")
        assert sg is not None and sg._value == 4
        # providers get independent semaphores.
        assert sa is not sg


def test_resolver_returns_none_on_config_load_error():
    """If config can't be loaded the resolver must not raise; it returns None."""
    before = m._request_semaphores.copy()
    with patch("hermes_cli.config.load_config_readonly",
               side_effect=RuntimeError("boom")):
        assert m._resolve_request_limit("p") is None
        assert m._get_request_semaphore("p") is None


def test_limit_independent_of_agent_concurrency_config():
    """The request-level throttle must not read or depend on
    delegation.max_concurrent_children; the two levels stay separate."""
    cfg = {
        "provider_max_concurrent_requests": 2,
        "provider_request_concurrency": {},
        "delegation": {"max_concurrent_children": 5},
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert m._get_request_semaphore("p")._value == 2

    m._request_semaphores.clear()
    cfg_none = {
        "provider_max_concurrent_requests": None,
        "provider_request_concurrency": {},
        "delegation": {"max_concurrent_children": 5},
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg_none):
        assert m._get_request_semaphore("p") is None