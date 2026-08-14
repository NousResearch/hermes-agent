"""Regression test for #31556.

``list_authenticated_providers()`` used to call ``load_config()`` ~69 times
and ``load_pool()`` ~65 times for a single ``/model`` invocation, because
every provider row re-seeded its own ``CredentialPool`` and re-read
``config.yaml``. The fix memoises ``_load_config_safe()`` per process and
shares ``CredentialPool`` instances across the three provider sections in
one ``list_authenticated_providers`` call.

This test drives the REAL functions against a temp HERMES_HOME with empty
auth/config state and counts the seam calls. The counts are loose enough to
survive reasonable provider list growth, but tight enough that the pre-fix
behaviour (70+ load_config, 70+ load_pool) fails.
"""
import json

import pytest

import agent.credential_pool as cp
import agent.models_dev as mdev
import hermes_cli.config as hc
import hermes_cli.models as models
from hermes_cli.model_switch import list_authenticated_providers


@pytest.fixture
def empty_picker_home(tmp_path, monkeypatch):
    """A minimal Hermes home with no credentials and a stubbed network."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}})
    )
    (home / "config.yaml").write_text("model:\n  provider: openai\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    getattr(cp, "_LOAD_CONFIG_SAFE_CACHE", {}).clear()
    yield home
    hc._LOAD_CONFIG_CACHE.clear()
    getattr(cp, "_LOAD_CONFIG_SAFE_CACHE", {}).clear()


@pytest.fixture(autouse=True)
def _stub_network_and_disk_cache(monkeypatch):
    """Prevent network/model-catalog fetches; the test targets local load cost."""
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr(models, "fetch_ollama_cloud_models", lambda: [])
    monkeypatch.setattr(mdev, "fetch_models_dev", lambda: {})


def _patched_counters(monkeypatch):
    """Count deepcopying ``load_config()``, ``load_config_readonly()`` and
    ``load_pool()`` so the test can pin the post-fix budgets."""
    counts = {"load_config": 0, "load_config_readonly": 0, "load_pool": 0}

    real_load_config = hc.load_config
    real_load_config_ro = hc.load_config_readonly
    real_load_pool = cp.load_pool

    def counting_load_config(*a, **kw):
        counts["load_config"] += 1
        return real_load_config(*a, **kw)

    def counting_load_config_ro(*a, **kw):
        counts["load_config_readonly"] += 1
        return real_load_config_ro(*a, **kw)

    def counting_load_pool(provider):
        counts["load_pool"] += 1
        return real_load_pool(provider)

    monkeypatch.setattr(hc, "load_config", counting_load_config)
    monkeypatch.setattr(hc, "load_config_readonly", counting_load_config_ro)
    monkeypatch.setattr(cp, "load_pool", counting_load_pool)
    return counts


def test_list_authenticated_providers_memoises_config_load(
    empty_picker_home, monkeypatch
):
    """Deepcopying ``load_config()`` must not run once per provider row."""
    counts = _patched_counters(monkeypatch)
    providers = list_authenticated_providers()
    assert isinstance(providers, list)

    # The fix routes every ``_load_config_safe()`` through
    # ``load_config_readonly()``. The direct ``load_config()`` calls come from
    # pre-existing unrelated paths (model_catalog, vertex_adapter,
    # is_provider_explicitly_configured) and are not the per-row config
    # deepcopy the issue targets. The target metric is ``load_config_readonly``,
    # which should be called at most once per process for the config load.
    assert counts["load_config_readonly"] <= 2, (
        f"load_config_readonly called {counts['load_config_readonly']}x — "
        "regression reintroduces repeated config loads"
    )
    # Deepcopying ``load_config()`` should also stay bounded; the model_catalog
    # and vertex paths add a few pre-existing calls, but the per-row
    # credential-pool expansion is gone.
    assert counts["load_config"] <= 10, (
        f"deepcopying load_config called {counts['load_config']}x — "
        "regression reintroduces per-row config deepcopy"
    )


def test_list_authenticated_providers_reuses_pool_per_provider(
    empty_picker_home, monkeypatch
):
    """The same provider must not be ``load_pool``-ed in every section."""
    counts = _patched_counters(monkeypatch)
    providers = list_authenticated_providers()
    assert isinstance(providers, list)

    # Pre-fix: 70+ load_pool calls. Post-fix: one per distinct provider
    # checked (empty auth store still has to inspect the built-in roster).
    assert counts["load_pool"] <= 45, (
        f"load_pool called {counts['load_pool']}x — "
        "regression reintroduces repeated pool seeding per provider"
    )
