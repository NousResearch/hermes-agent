"""The assembly's doors: a profile's scope, one shared backend per profile with released references, the owned
read-only reader, and health and location answered without opening a store."""

from __future__ import annotations

import asyncio

import pytest

import persistence_assembly
from hermes_constants import get_hermes_home, hermes_home_key
from persistence import ProfileScope, StorageHealth, StorageProbe, StorageState, StoreUnavailable
from persistence.sqlite import _registry
from tests.persistence_sqlite.conftest import profile_home


def test_profile_scope_resolves_a_named_profile_and_a_pinned_home():
    home = profile_home("alpha")
    scope = persistence_assembly.profile_scope("alpha")
    assert scope == persistence_assembly.profile_scope(home=home)
    assert (scope.profile_dir, scope.name) == (home.resolve(), "alpha")
    assert scope.hermes_home_key == hermes_home_key(home)
    elsewhere = persistence_assembly.profile_scope(home=get_hermes_home() / "not-a-profile")
    assert elsewhere.name == "default"


def test_handles_on_one_profile_share_a_backend_and_release_on_close():
    home = profile_home("alpha")
    scope = persistence_assembly.profile_scope(home=home)
    first = persistence_assembly.open_persistence(scope)
    second = persistence_assembly.open_persistence(ProfileScope(profile_dir=home, name="other-label"))
    assert first.scope == second.scope and persistence_assembly._shared[scope.hermes_home_key][1] == 2
    first.state_meta.set("k", "v")
    assert first.holder() is second.holder()
    first.close()
    first.close()
    assert _registry.stats()["live_generations"] == 1
    assert second.state_meta.get("k") == "v"
    second.close()
    assert _registry.stats()["live_generations"] == 0 and persistence_assembly._shared == {}
    with persistence_assembly.open_persistence(scope) as third:
        assert third.state_meta.get("k") == "v"


def test_a_closed_handle_refuses_use_while_another_handle_holds_the_backend():
    scope = persistence_assembly.profile_scope(home=profile_home("alpha"))
    first = persistence_assembly.open_persistence(scope)
    second = persistence_assembly.open_persistence(scope)
    first.state_meta.set("k", "v")
    first.close()
    with pytest.raises(RuntimeError, match="persistence handle used after close"):
        first.state_meta.set("k", "stale")
    with pytest.raises(RuntimeError, match="persistence handle used after close"):
        first.ready()
    assert second.state_meta.get("k") == "v"
    second.close()


def test_a_handle_is_bound_to_its_own_profile():
    alpha, beta = profile_home("alpha"), profile_home("beta")
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=alpha)) as one:
        with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=beta)) as two:
            one.state_meta.set("k", "alpha")
            assert two.state_meta.get("k") is None
            assert one.holder() is not two.holder()


def test_awaitable_form_through_the_handle():
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=profile_home("alpha"))) as handle:
        asyncio.run(handle.aio.state_meta.set("k", "v"))
        assert handle.state_meta.get("k") == "v"


def test_eager_open_raises_the_typed_store_error_and_releases_its_reference():
    home = profile_home("broken")
    (home / "state.db").mkdir()
    with pytest.raises(StoreUnavailable):
        persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home), eager=True)
    assert persistence_assembly._shared == {}


def test_reader_is_read_only_owned_and_never_creates():
    home = profile_home("beta")
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home)) as writer:
        writer.state_meta.set("k", "v")
    for target in (home, home / "state.db", "beta"):
        with persistence_assembly.open_persistence_reader(target) as reader:
            assert reader.read_only and reader.scope.name == "beta"
            assert reader.state_meta.get("k") == "v"
    missing = get_hermes_home() / "profiles" / "gamma"
    with persistence_assembly.open_persistence_reader(missing) as reader:
        with pytest.raises(StoreUnavailable):
            reader.state_meta.get("k")
    assert not (missing / "state.db").exists()


def test_health_and_location_doors_answer_without_opening_a_store():
    home = profile_home("delta")
    scope = persistence_assembly.profile_scope(home=home)
    health = persistence_assembly.storage_health(scope)
    assert isinstance(health, StorageHealth)
    assert health.probe() == StorageProbe(status="ok", detail="not initialized")
    assert health.storage_state() is StorageState.OK
    assert persistence_assembly.store_location(scope) == home.resolve() / "state.db"
    assert not (home / "state.db").exists() and _registry.stats()["live_generations"] == 0
    with persistence_assembly.open_persistence(scope) as handle:
        handle.state_meta.set("k", "v")
        assert health.probe() == handle.health.probe() == StorageProbe(status="ok", detail=None)


def test_close_stores_under_sweeps_and_the_handle_reacquires():
    home = profile_home("epsilon")
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home)) as handle:
        handle.state_meta.set("k", "v")
        assert persistence_assembly.close_stores_under(home) == 1
        assert handle.state_meta.get("k") == "v"


def _record_resolution(monkeypatch) -> list[tuple]:
    """Every provider resolution and process-config read the assembly makes from here on."""
    import hermes_cli.config

    calls: list[tuple] = []
    provider_for_home, load_config_readonly = persistence_assembly._provider_for_home, hermes_cli.config.load_config_readonly

    def recorded_provider(home):
        calls.append(("provider", home))
        return provider_for_home(home)

    def recorded_config(*args, **kwargs):
        calls.append(("config",))
        return load_config_readonly(*args, **kwargs)

    monkeypatch.setattr(persistence_assembly, "_provider_for_home", recorded_provider)
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", recorded_config)
    return calls


def _listing(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


def test_opening_reads_no_configuration_and_touches_no_file(monkeypatch):
    calls = _record_resolution(monkeypatch)
    home = get_hermes_home()
    before = _listing(home)
    handle = persistence_assembly.open_persistence(persistence_assembly.profile_scope())
    with persistence_assembly.open_persistence_reader(home) as reader:
        assert (handle.read_only, reader.read_only) == (False, True)
        assert handle.scope == reader.scope == persistence_assembly.profile_scope(home=home)
        assert calls == [] and _listing(home) == before
        handle.state_meta.set("k", "v")
        # The store open reads its own database settings after the assembly resolved the provider.
        assert calls[:2] == [("provider", home.resolve()), ("config",)] and "state.db" in _listing(home)
        assert reader.state_meta.get("k") == "v"
    handle.close()
    assert persistence_assembly._shared == {}


def test_a_handle_closed_before_use_resolves_nothing(monkeypatch):
    calls = _record_resolution(monkeypatch)
    home = profile_home("alpha")
    handle = persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home))
    handle.close()
    handle.close()
    with persistence_assembly.open_persistence_reader(home):
        pass
    assert calls == [] and persistence_assembly._shared == {} and _listing(home) == []
    with pytest.raises(RuntimeError, match="used after close"):
        handle.state_meta.get("k")
    assert calls == [] and _listing(home) == []


def test_an_unknown_provider_raises_at_first_use_and_at_ready(monkeypatch):
    unknown = "unknown persistence provider 'bogus'"
    home = profile_home("zeta")
    (home / "config.yaml").write_text("persistence:\n  provider: bogus\n", encoding="utf-8")
    scope = persistence_assembly.profile_scope(home=home)
    handle = persistence_assembly.open_persistence(scope)
    assert (handle.scope, handle.read_only) == (scope, False)
    with pytest.raises(ValueError, match=unknown):
        handle.state_meta.get("k")
    with pytest.raises(ValueError, match=unknown):
        handle.ready()
    handle.close()
    with pytest.raises(ValueError, match=unknown):
        persistence_assembly.open_persistence(scope, eager=True)
    assert persistence_assembly._shared == {}
    with persistence_assembly.open_persistence_reader(home) as reader:
        assert reader.read_only
        with pytest.raises(ValueError, match=unknown):
            reader.ready()
        with pytest.raises(ValueError, match=unknown):
            reader.state_meta.get("k")
    assert not (home / "state.db").exists()
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {"persistence": {"provider": "bogus"}})
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope()) as current:
        with pytest.raises(ValueError, match=unknown):
            current.ready()
    assert persistence_assembly._shared == {}


def test_a_reader_close_closes_its_backend_once_and_the_exit_closes_it_too():
    home = profile_home("beta")
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home)) as writer:
        writer.state_meta.set("k", "v")
    with persistence_assembly.open_persistence_reader(home) as reader:
        assert reader.state_meta.get("k") == "v"
        reader.close()
        reader.close()
        with pytest.raises(RuntimeError, match="used after close"):
            reader.state_meta.get("k")
    with persistence_assembly.open_persistence_reader(home) as exited:
        assert exited.state_meta.get("k") == "v"
    with pytest.raises(RuntimeError, match="used after close"):
        exited.state_meta.get("k")


def test_live_stores_lends_every_open_store_and_pins_it_for_the_block():
    alpha, beta = profile_home("alpha"), profile_home("beta")
    one = persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=alpha))
    two = persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=beta))
    one.state_meta.set("k", "alpha")
    two.ready()
    with persistence_assembly.live_stores() as stores:
        assert sorted(store.scope.name for store in stores) == ["alpha", "beta"]
        lent = {store.scope.name: store for store in stores}
        assert lent["alpha"].holder() is one.holder()
        assert lent["alpha"].search.retry_deferred_search_recovery() is False
        # The last owner releases inside the block: the lent handle still reads through the pinned store.
        one.close()
        assert _registry.stats()["live_generations"] == 2
        assert lent["alpha"].state_meta.get("k") == "alpha"
    assert _registry.stats()["live_generations"] == 1
    with persistence_assembly.live_stores() as stores:
        assert [store.scope.name for store in stores] == ["beta"]
        holder = stores[0].holder()
    assert holder is two.holder() and _registry.stats()["total_refcounts"] == 1
    two.close()
    with persistence_assembly.live_stores() as stores:
        assert stores == []


def test_a_lent_handle_never_opens_its_store_again():
    home = profile_home("alpha")
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=home)) as owner:
        owner.ready()
        with persistence_assembly.live_stores() as stores:
            assert persistence_assembly.close_stores_under(home) == 1
            # The sweep closed the store under the lent handle; the retry answers False on it, as on a closed store.
            assert stores[0].search.retry_deferred_search_recovery() is False
            assert _registry.stats()["live_generations"] == 0
