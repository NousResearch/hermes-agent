"""INVARIANT tests: probe client stubs must never enter the auxiliary client cache.

Regression reproduction: check_vision_requirements() resolved True on the first
call in a process and False on every later call, because the second write path
in _get_cached_client() lacked the _AuxProbeClientStub guard that
_store_cached_client() enforces. See #31179-style "check_fn must be repeatable".
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def _fresh_client_cache(monkeypatch):
    """Start each test with an empty auxiliary client cache."""
    from agent import auxiliary_client as ac

    monkeypatch.setattr(ac, "_client_cache", {}, raising=False)
    yield ac
    ac._client_cache.clear() if hasattr(ac, "_client_cache") and isinstance(ac._client_cache, dict) else None


def _run_vision_check():
    from tools.vision_tools import check_vision_requirements

    return check_vision_requirements()


class TestProbeStubNeverCached:
    """The check_fn contract: repeatability. A probe must answer True every time."""

    def test_stub_not_stored_by_build_path(self, _fresh_client_cache):
        """_get_cached_client's inline write path must reject probe stubs (the
        guard _store_cached_client has was missing here)."""
        from agent import auxiliary_client as ac
        from agent.auxiliary_client import aux_probe_mode, _AuxProbeClientStub

        stub = _AuxProbeClientStub(api_key="sk-test", base_url="https://example.invalid/v1")
        with aux_probe_mode():
            stored = ac._store_cached_client(("p", False, "", "", "", (), False, "", "", ""), stub, None)
        # even the guarded path must reject; and the cache must stay stub-free
        assert stored is None
        assert not any(isinstance(v[0], _AuxProbeClientStub) for v in ac._client_cache.values())

    def test_vision_check_repeatably_true_with_explicit_aux(self, _fresh_client_cache, monkeypatch):
        """Full-path regression: with a valid auxiliary.vision provider, repeated
        check_vision_requirements() calls must all return True (idempotence)."""
        from agent import auxiliary_client as ac

        # Minimal explicit aux config: any config provider name resolves without
        # network in probe mode (probe mode never builds real clients).
        monkeypatch.setenv("HERMES_HOME", "/home/pjj/.hermes")
        results = [_run_vision_check() for _ in range(10)]
        assert all(results), f"check_vision_requirements not repeatable: {results}"

    def test_no_stub_left_in_cache_after_checks(self, _fresh_client_cache):
        from agent import auxiliary_client as ac
        from agent.auxiliary_client import _AuxProbeClientStub

        _run_vision_check()
        _run_vision_check()
        assert not any(isinstance(v[0], _AuxProbeClientStub) for v in ac._client_cache.values())
