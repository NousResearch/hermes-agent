"""/api/model/info must bound its context-length probe (#63214).

The resolver chain (``agent.model_metadata.get_model_context_length``) runs
several sequential provider probes, each with its own multi-second timeout, so
an unreachable ``model.base_url`` held the response for tens of seconds — and
the Desktop Model Settings page, which awaits this endpoint inside one
``Promise.all``, showed loading skeletons indefinitely.
"""

from __future__ import annotations

import time


def _config(model: str, provider: str, base_url: str) -> dict:
    return {"model": {"default": model, "provider": provider, "base_url": base_url}}


def test_model_info_degrades_when_the_context_probe_exceeds_its_budget(monkeypatch):
    from hermes_cli.web_routers import models as router

    monkeypatch.setattr(
        router, "_load_config_scoped",
        lambda profile: _config("some-model", "custom-proxy", "http://localhost:9/v1"),
    )
    monkeypatch.setattr(router, "_MODEL_INFO_PROBE_BUDGET_S", 0.2)

    import agent.model_metadata as metadata

    def _hanging_probe(model, base_url="", api_key="", config_context_length=None, provider="", custom_providers=None):
        time.sleep(1.0)
        raise AssertionError("the abandoned probe should never be awaited")

    monkeypatch.setattr(metadata, "get_model_context_length", _hanging_probe)

    started = time.monotonic()
    info = router.get_model_info()
    elapsed = time.monotonic() - started

    # The response degrades to "auto context unknown" instead of hanging…
    assert info["model"] == "some-model"
    assert info["provider"] == "custom-proxy"
    assert info["auto_context_length"] == 0
    # …and it returns within the (shrunk) budget, not the probe's 1s sleep.
    assert elapsed < 0.9


def test_model_info_surfaces_the_context_value_when_the_probe_is_fast(monkeypatch):
    from hermes_cli.web_routers import models as router

    monkeypatch.setattr(
        router, "_load_config_scoped",
        lambda profile: _config("some-model", "custom-proxy", "http://localhost:9/v1"),
    )

    import agent.model_metadata as metadata

    def _fast_probe(model, base_url="", api_key="", config_context_length=None, provider="", custom_providers=None):
        return 131072

    monkeypatch.setattr(metadata, "get_model_context_length", _fast_probe)

    info = router.get_model_info()

    assert info["model"] == "some-model"
    assert info["auto_context_length"] == 131072
    assert info["effective_context_length"] == 131072


def test_model_info_bounds_the_endpoint_key_lookup_behind_a_held_auth_lock(monkeypatch):
    """Resolving the probe key loads the endpoint's credential pool, which persists config-seeded rows
    to auth.json and waits for its lock (an OAuth refresh holds it across a network call). That wait
    is inside the route's budget, not in front of it."""
    import threading

    from hermes_cli.auth import _auth_store_lock
    from hermes_cli.config import save_config
    from hermes_cli.web_routers import models as router

    base = "http://127.0.0.1:47970/v1"
    save_config({"model": {"default": "qwen-lab", "provider": "custom", "base_url": base},
                 "providers": {"lab": {"base_url": base, "api_key": "entry-key-1234"}}})
    monkeypatch.setattr(router, "_MODEL_INFO_PROBE_BUDGET_S", 0.3)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda **kwargs: 131072)
    held, release = threading.Event(), threading.Event()

    def hold():
        with _auth_store_lock():
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        started = time.monotonic()
        info = router.get_model_info()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(5)
    assert info["auto_context_length"] == 0
    assert elapsed < 1.0
