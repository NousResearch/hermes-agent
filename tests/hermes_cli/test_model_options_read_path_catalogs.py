"""``model.options`` is a READ path: opening the picker must not wait on a provider catalog probe.

A normal open (``refresh=False``) used to run live ``/v1/models`` fetches inline — a cold cache
serialized every authed provider and one degraded provider (hanging endpoint, failed auth probe)
held the whole picker for as long as its probe took (#114215). The open now serves cached/curated
rows and refreshes them off-thread; only an explicit refresh (``refresh=True``, the "Refresh
Models" action) is allowed to block on probes.
"""

import threading
import time

import hermes_cli.models as models_mod
from hermes_cli.inventory import build_model_options_payload, load_picker_context

_DEAD_PROVIDER = "deepseek"


def _picker_env(monkeypatch, tmp_path, *, hung=None):
    """One authed provider visible to the picker, no real network, isolated model-id cache.

    ``hung`` is a slug whose probe blocks on the returned event — a stand-in for a degraded
    provider. Returns ``(live_calls, release_event)``; ``live_calls`` records every live probe.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    release = threading.Event()
    live_calls: list[str] = []

    def fake_provider_model_ids(provider, *, force_refresh=False):
        live_calls.append(provider)
        if provider == hung:
            release.wait(30)
        return ["live-model"]

    monkeypatch.setattr(models_mod, "provider_model_ids", fake_provider_model_ids)
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda *a, **k: {_DEAD_PROVIDER: {"env": ["DEEPSEEK_API_KEY"], "name": "DeepSeek"}},
    )
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {_DEAD_PROVIDER: _DEAD_PROVIDER})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    return live_calls, release


def _drain_background_warms(timeout=10.0) -> None:
    """Let spawned catalog warms finish so a tmp HERMES_HOME can be torn down with no writers left."""
    deadline = time.time() + timeout
    while time.time() < deadline and models_mod._swr_refresh_inflight:
        time.sleep(0.02)


def _row(payload, slug):
    return next((row for row in payload["providers"] if row["slug"] == slug), None)


def test_degraded_provider_cannot_stall_the_open(monkeypatch, tmp_path):
    """One provider whose probe hangs must not hold the picker: the payload comes back while the
    probe is still in flight, the row still renders from its curated list, and the probe runs
    off the read path. (Pre-fix this assertion only ever ran after the probe returned.)"""
    live_calls, release = _picker_env(monkeypatch, tmp_path, hung=_DEAD_PROVIDER)
    box: dict = {}

    def _open():
        box["payload"] = build_model_options_payload(load_picker_context())

    opener = threading.Thread(target=_open, daemon=True)
    opener.start()
    opener.join(timeout=15)
    returned_while_hung = "payload" in box
    release.set()  # never leave the probe hanging, whatever the outcome
    opener.join(timeout=15)
    _drain_background_warms()

    assert returned_while_hung, "the open waited on a degraded provider's catalog probe"
    row = _row(box["payload"], _DEAD_PROVIDER)
    assert row is not None and row["models"], "the degraded row must still render from curated"
    assert "live-model" not in row["models"]
    assert _DEAD_PROVIDER in live_calls, "the degraded provider is still probed — off the read path"


def test_explicit_refresh_still_probes_providers(monkeypatch, tmp_path):
    """``refresh=True`` is the explicit "Refresh Models" action: it is allowed to run live probes."""
    live_calls, _ = _picker_env(monkeypatch, tmp_path)

    payload = build_model_options_payload(load_picker_context(), refresh=True)

    assert _DEAD_PROVIDER in live_calls, "an explicit refresh must still probe provider catalogs"
    assert "live-model" in _row(payload, _DEAD_PROVIDER)["models"]
    _drain_background_warms()
