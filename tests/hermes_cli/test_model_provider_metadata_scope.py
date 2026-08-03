"""Context-scoped provider metadata regression tests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@contextmanager
def _hermes_home(path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _profile_for_current_home():
    from hermes_constants import get_hermes_home

    suffix = get_hermes_home().name
    return SimpleNamespace(
        name=f"scoped-provider-{suffix}",
        aliases=("scoped-provider",),
        auth_type="api_key",
        display_name=f"Scoped Provider {suffix}",
        description=f"Provider for {suffix}",
    )


@pytest.fixture
def scoped_provider_discovery(monkeypatch):
    import providers
    monkeypatch.setattr(
        providers,
        "list_providers",
        lambda: [_profile_for_current_home()],
    )
    monkeypatch.setattr(
        providers,
        "is_plugin_managed_provider_id",
        lambda provider_id: False,
    )
    monkeypatch.setattr(
        providers,
        "is_provider_plugin_active",
        lambda provider_id: False,
    )


def test_provider_metadata_snapshots_stay_bound_to_their_profile(
    tmp_path,
    scoped_provider_discovery,
):
    import hermes_cli.models as models

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    compatibility_views = (
        models.CANONICAL_PROVIDERS,
        models._PROVIDER_ALIASES,
        models._PROVIDER_LABELS,
        models._KNOWN_PROVIDER_NAMES,
    )

    with _hermes_home(home_a):
        models._refresh_canonical_providers_from_plugins()
        snapshot_a = models._provider_metadata_snapshot()
        assert models.normalize_provider("scoped-provider") == "scoped-provider-a"
        assert "scoped-provider-a" in {
            entry.slug for entry in models.CANONICAL_PROVIDERS
        }
        assert "scoped-provider-b" not in {
            entry.slug for entry in models.CANONICAL_PROVIDERS
        }

    with _hermes_home(home_b):
        models._refresh_canonical_providers_from_plugins()
        snapshot_b = models._provider_metadata_snapshot()
        assert models.normalize_provider("scoped-provider") == "scoped-provider-b"
        assert "scoped-provider-b" in models._KNOWN_PROVIDER_NAMES

    assert snapshot_a is not snapshot_b
    with pytest.raises(TypeError):
        snapshot_a.provider_aliases["new-alias"] = "scoped-provider-a"

    # A refresh in profile B must not mutate or replace profile A's snapshot.
    with _hermes_home(home_a):
        assert models._provider_metadata_snapshot() is snapshot_a
        assert models.normalize_provider("scoped-provider") == "scoped-provider-a"
        assert "scoped-provider-b" not in models._KNOWN_PROVIDER_NAMES
        canonical_view, aliases_view, labels_view, known_names_view = (
            compatibility_views
        )
        assert aliases_view.get("scoped-provider") == "scoped-provider-a"
        assert labels_view.get("scoped-provider-a") == "Scoped Provider a"
        assert "scoped-provider-a" in known_names_view
        assert "scoped-provider-a" in {
            entry.slug for entry in canonical_view
        }

    # Existing ``from hermes_cli.models import ...`` references stay live.
    current_views = (
        models.CANONICAL_PROVIDERS,
        models._PROVIDER_ALIASES,
        models._PROVIDER_LABELS,
        models._KNOWN_PROVIDER_NAMES,
    )
    assert all(
        original is current
        for original, current in zip(compatibility_views, current_views)
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
        assert set(models._KNOWN_PROVIDER_NAMES) == expected_names
        assert models._KNOWN_PROVIDER_NAMES == expected_names
        assert expected_names == models._KNOWN_PROVIDER_NAMES
        assert models._KNOWN_PROVIDER_NAMES | {"sentinel"} == (
            expected_names | {"sentinel"}
        )
        assert {"sentinel"} | models._KNOWN_PROVIDER_NAMES == (
            expected_names | {"sentinel"}
        )
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

    home_a = tmp_path / "hot-a"
    home_b = tmp_path / "hot-b"
    for home in (home_a, home_b):
        with _hermes_home(home):
            models._refresh_canonical_providers_from_plugins()

    identity_calls = 0
    refresh_calls = 0

    def discovery_identity():
        nonlocal identity_calls
        identity_calls += 1
        raise AssertionError("hot metadata read re-entered provider discovery")

    original_hook = models._refresh_canonical_providers_from_plugins

    def counted_refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        original_hook()

    hooks = list(providers._PROVIDER_REFRESH_HOOKS)
    hooks = [
        counted_refresh if hook is original_hook else hook
        for hook in hooks
    ]
    monkeypatch.setattr(providers, "_PROVIDER_REFRESH_HOOKS", hooks)
    monkeypatch.setattr(
        providers,
        "get_provider_discovery_identity",
        discovery_identity,
    )

    for _ in range(20):
        with _hermes_home(home_a):
            assert models.normalize_provider("scoped-provider") == (
                "scoped-provider-hot-a"
            )
            assert dict(models._PROVIDER_LABELS)
            assert set(models._KNOWN_PROVIDER_NAMES)
        with _hermes_home(home_b):
            assert models.normalize_provider("scoped-provider") == (
                "scoped-provider-hot-b"
            )
            assert dict(models._PROVIDER_LABELS)
            assert set(models._KNOWN_PROVIDER_NAMES)

    assert identity_calls == 0
    assert refresh_calls == 0


def test_provider_activation_invalidation_refreshes_same_scope_snapshot(
    tmp_path,
    scoped_provider_discovery,
    monkeypatch,
):
    import hermes_cli.models as models
    import providers

    home = tmp_path / "activation-profile"
    home.mkdir()
    active_name = "activation-provider-a"

    def active_profiles():
        return [
            SimpleNamespace(
                name=active_name,
                aliases=("activation-provider",),
                auth_type="api_key",
                display_name=active_name,
                description=f"Provider {active_name}",
            )
        ]

    monkeypatch.setattr(providers, "list_providers", active_profiles)
    original_hook = models._refresh_canonical_providers_from_plugins
    hook_calls = 0

    def counted_refresh():
        nonlocal hook_calls
        hook_calls += 1
        original_hook()

    hooks = list(providers._PROVIDER_REFRESH_HOOKS)
    hooks = [
        counted_refresh if hook is original_hook else hook
        for hook in hooks
    ]
    monkeypatch.setattr(providers, "_PROVIDER_REFRESH_HOOKS", hooks)

    with _hermes_home(home):
        original_hook()
        snapshot_a = models._provider_metadata_snapshot()
        assert models.normalize_provider("activation-provider") == active_name

        active_name = "activation-provider-b"
        hook_calls = 0
        providers.invalidate_provider_discovery()

        snapshot_b = models._provider_metadata_snapshot()
        assert hook_calls == 1
        assert snapshot_b is not snapshot_a
        assert models.normalize_provider("activation-provider") == active_name


def test_concurrent_profile_refreshes_do_not_cross_contaminate(
    tmp_path,
    scoped_provider_discovery,
):
    import hermes_cli.models as models

    barrier = threading.Barrier(2)
    errors = []

    def worker(home, expected_provider):
        try:
            with _hermes_home(home):
                barrier.wait(timeout=5)
                for _ in range(25):
                    models._refresh_canonical_providers_from_plugins()
                    assert (
                        models.normalize_provider("scoped-provider")
                        == expected_provider
                    )
                    assert expected_provider in models._KNOWN_PROVIDER_NAMES
                    assert expected_provider in {
                        entry.slug for entry in models.CANONICAL_PROVIDERS
                    }
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(
            target=worker,
            args=(tmp_path / "a", "scoped-provider-a"),
        ),
        threading.Thread(
            target=worker,
            args=(tmp_path / "b", "scoped-provider-b"),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
