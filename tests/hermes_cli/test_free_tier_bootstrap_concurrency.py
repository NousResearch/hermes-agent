"""Concurrent startup shares one bootstrap owner without locking out publication."""
import threading
from concurrent.futures import ThreadPoolExecutor

from hermes_cli import free_tier_bootstrap as bootstrap


def test_concurrent_bootstrap_shares_record_and_broadcast(monkeypatch):
    from hermes_cli import anon_auth
    entered, release = threading.Event(), threading.Event()
    bootstrap.reset_for_tests()
    calls = []
    def inventory():
        entered.set()
        assert release.wait(3)
        return True
    monkeypatch.setattr(bootstrap, "_inventory_other_providers", inventory)
    monkeypatch.setattr(bootstrap, "_resolve_inference", lambda: "configured")
    monkeypatch.setattr(bootstrap, "_broadcast", calls.append)
    monkeypatch.setattr(anon_auth, "current_nous_state", lambda: None)
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: False)
    original_wait = bootstrap._done.wait
    waiting = threading.Event()
    def wait(timeout):
        waiting.set()
        return original_wait(timeout)
    monkeypatch.setattr(bootstrap._done, "wait", wait)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(bootstrap.run_bootstrap)
            assert entered.wait(3)
            second = pool.submit(bootstrap.run_bootstrap)
            assert waiting.wait(3)
            release.set()
            assert first.result(timeout=3) is second.result(timeout=3)
        assert len(calls) == 1
    finally:
        release.set()
        bootstrap.reset_for_tests()
