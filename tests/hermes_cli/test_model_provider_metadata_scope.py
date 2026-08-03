"""Context-scoped provider metadata regression tests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@contextmanager
def _hermes_home(path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _provider_profile(name, alias, display_name, description):
    return SimpleNamespace(
        name=name,
        aliases=(alias,),
        auth_type="api_key",
        display_name=display_name,
        description=description,
    )


def _profile_for_current_home():
    from hermes_constants import get_hermes_home

    suffix = get_hermes_home().name
    return _provider_profile(f"scoped-provider-{suffix}", "scoped-provider", f"Scoped Provider {suffix}", f"Provider for {suffix}")


def _assert_scoped_provider(models, suffix):
    expected = f"scoped-provider-{suffix}"
    assert models.normalize_provider("scoped-provider") == expected
    assert models._PROVIDER_LABELS[expected] == f"Scoped Provider {suffix}"
    assert expected in models._KNOWN_PROVIDER_NAMES
    assert expected in {entry.slug for entry in models.CANONICAL_PROVIDERS}
    return expected


def _compatibility_views(models):
    return (
        models.CANONICAL_PROVIDERS,
        models._PROVIDER_ALIASES,
        models._PROVIDER_LABELS,
        models._KNOWN_PROVIDER_NAMES,
    )


def _count_refresh_hooks(monkeypatch, models, providers):
    original = models._refresh_canonical_providers_from_plugins
    calls = []

    def counted_refresh():
        calls.append(None)
        original()

    hooks = [counted_refresh if hook is original else hook for hook in providers._PROVIDER_REFRESH_HOOKS]
    monkeypatch.setattr(providers, "_PROVIDER_REFRESH_HOOKS", hooks)
    return original, calls


@pytest.fixture
def scoped_provider_discovery(monkeypatch):
    import providers

    monkeypatch.setattr(providers, "list_providers", lambda: [_profile_for_current_home()])
    monkeypatch.setattr(providers, "is_plugin_managed_provider_id", lambda provider_id: False)
    monkeypatch.setattr(providers, "is_provider_plugin_active", lambda provider_id: False)


def test_provider_metadata_snapshots_stay_bound_to_their_profile(
    tmp_path,
    scoped_provider_discovery,
):
    import hermes_cli.models as models

    compatibility_views = _compatibility_views(models)
    snapshots = {}
    for suffix, other in (("a", "b"), ("b", "a")):
        with _hermes_home(tmp_path / suffix):
            models._refresh_canonical_providers_from_plugins()
            snapshots[suffix] = models._provider_metadata_snapshot()
            _assert_scoped_provider(models, suffix)
            assert f"scoped-provider-{other}" not in models._KNOWN_PROVIDER_NAMES

    snapshot_a, snapshot_b = snapshots["a"], snapshots["b"]
    assert snapshot_a is not snapshot_b
    with pytest.raises(TypeError):
        snapshot_a.provider_aliases["new-alias"] = "scoped-provider-a"

    # A refresh in profile B must not mutate or replace profile A's snapshot.
    with _hermes_home(tmp_path / "a"):
        assert models._provider_metadata_snapshot() is snapshot_a
        _assert_scoped_provider(models, "a")
        assert "scoped-provider-b" not in models._KNOWN_PROVIDER_NAMES
        canonical_view, aliases_view, labels_view, known_names_view = compatibility_views
        assert aliases_view.get("scoped-provider") == "scoped-provider-a"
        assert labels_view.get("scoped-provider-a") == "Scoped Provider a"
        assert "scoped-provider-a" in known_names_view
        assert "scoped-provider-a" in {entry.slug for entry in canonical_view}

    # Existing ``from hermes_cli.models import ...`` references stay live.
    assert all(
        original is current
        for original, current in zip(compatibility_views, _compatibility_views(models))
    )


def test_provider_metadata_compatibility_views_preserve_collection_reads(
    tmp_path,
    scoped_provider_discovery,
):
    import hermes_cli.models as models

    with _hermes_home(tmp_path / "collection-profile"):
        models._refresh_canonical_providers_from_plugins()
        snapshot = models._provider_metadata_snapshot()

        expected_names = set(snapshot.known_provider_names)
        known_names = models._KNOWN_PROVIDER_NAMES
        for left, right in ((set(known_names), expected_names), (known_names, expected_names), (expected_names, known_names)):
            assert left == right
        for actual in (known_names | {"sentinel"}, {"sentinel"} | known_names):
            assert actual == expected_names | {"sentinel"}
        assert set(models._canonical_slugs) == set(snapshot.canonical_slugs)

        expected_canonical = list(snapshot.canonical_providers)
        assert models.CANONICAL_PROVIDERS == expected_canonical
        assert expected_canonical == models.CANONICAL_PROVIDERS


def test_cached_profile_views_do_not_reenter_provider_discovery(
    tmp_path,
    scoped_provider_discovery,
    monkeypatch,
):
    import hermes_cli.models as models
    import providers

    profiles = [(tmp_path / f"hot-{suffix}", f"hot-{suffix}") for suffix in ("a", "b")]
    for home, _ in profiles:
        with _hermes_home(home):
            models._refresh_canonical_providers_from_plugins()

    identity_calls = 0

    def discovery_identity():
        nonlocal identity_calls
        identity_calls += 1
        raise AssertionError("hot metadata read re-entered provider discovery")

    _, refresh_calls = _count_refresh_hooks(monkeypatch, models, providers)
    monkeypatch.setattr(providers, "get_provider_discovery_identity", discovery_identity)

    for _ in range(20):
        for home, suffix in profiles:
            with _hermes_home(home):
                _assert_scoped_provider(models, suffix)
                assert dict(models._PROVIDER_LABELS)
                assert set(models._KNOWN_PROVIDER_NAMES)

    assert identity_calls == 0
    assert refresh_calls == []


def test_provider_activation_invalidation_refreshes_same_scope_snapshot(
    tmp_path,
    scoped_provider_discovery,
    monkeypatch,
):
    import hermes_cli.models as models
    import providers

    active_name = "activation-provider-a"
    monkeypatch.setattr(
        providers,
        "list_providers",
        lambda: [_provider_profile(active_name, "activation-provider", active_name, f"Provider {active_name}")],
    )
    original_hook, hook_calls = _count_refresh_hooks(monkeypatch, models, providers)

    with _hermes_home(tmp_path / "activation-profile"):
        original_hook()
        snapshot_a = models._provider_metadata_snapshot()
        assert models.normalize_provider("activation-provider") == active_name

        active_name = "activation-provider-b"
        hook_calls.clear()
        providers.invalidate_provider_discovery()

        snapshot_b = models._provider_metadata_snapshot()
        assert len(hook_calls) == 1
        assert snapshot_b is not snapshot_a
        assert models.normalize_provider("activation-provider") == active_name


def test_concurrent_profile_refreshes_do_not_cross_contaminate(
    tmp_path,
    scoped_provider_discovery,
):
    import hermes_cli.models as models

    barrier = threading.Barrier(2)
    errors = []

    def worker(home, suffix):
        try:
            with _hermes_home(home):
                barrier.wait(timeout=5)
                for _ in range(25):
                    models._refresh_canonical_providers_from_plugins()
                    _assert_scoped_provider(models, suffix)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(tmp_path / suffix, suffix))
        for suffix in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
