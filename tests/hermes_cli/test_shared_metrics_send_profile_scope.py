"""Profile-scope regression for the shared-metrics sender thread."""

from __future__ import annotations

import threading

from hermes_cli.observability import relay_shared_metrics as mod
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


class _FakeStore:
    pass


class _FakeSubscriber:
    def __init__(self) -> None:
        self.store = _FakeStore()


class _Runtime(mod._Runtime):
    """Minimal runtime that exercises the production send-thread boundary."""

    def __init__(self) -> None:
        self._send_lock = threading.RLock()
        self._send_thread = None
        self.subscriber = _FakeSubscriber()


def test_send_pass_rechecks_consent_in_routed_profile(monkeypatch, tmp_path):
    """The background consent check must stay owned by the profile that started the pass."""
    launch = tmp_path / "launch"
    served = tmp_path / "served"
    launch.mkdir()
    served.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))

    def config_for_active_profile():
        return {
            "telemetry": {
                "shared_metrics": {
                    "enabled": True,
                    "send": get_hermes_home() == served,
                    "endpoint": "https://metrics.test/v1",
                }
            }
        }

    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        config_for_active_profile,
        raising=False,
    )
    # This regression is about the async re-check, not SQLite consent-window bookkeeping.
    monkeypatch.setattr(mod, "_reconcile_store_consent", lambda *_args, **_kwargs: None)

    observed: list[tuple[object, bool]] = []
    checked = threading.Event()

    class _FakeSender:
        def __init__(self, store, endpoint, *, consent_check):
            self.consent_check = consent_check

        def send_pending(self):
            observed.append((get_hermes_home(), self.consent_check()))
            checked.set()

    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
        _FakeSender,
    )

    runtime = _Runtime()
    token = set_hermes_home_override(served)
    try:
        runtime._send_exported_packages()
    finally:
        reset_hermes_home_override(token)

    assert checked.wait(2.0), "the async send pass did not execute"
    runtime._join_send_thread(timeout=2.0)
    assert observed == [(served, True)]
