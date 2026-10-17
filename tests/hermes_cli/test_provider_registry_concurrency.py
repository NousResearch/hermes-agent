"""Concurrent plugin-provider registration must not break PROVIDER_REGISTRY iteration (#134594).

Plugin-load workers mirror provider rows into ``PROVIDER_REGISTRY`` from threads while the engine
iterates it (env-key auto-detect, auth-status fallbacks, metric attribution…). On a bare dict that
race ends in ``RuntimeError: dictionary changed size during iteration`` and the whole plugin load
fails. ``_ConcurrentProviderRegistry`` keeps single-key reads lock-free but serves every iteration
path from a key snapshot, so a concurrent insert cannot invalidate an in-flight iterator.
"""

import threading

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    _ConcurrentProviderRegistry,
)


def _row(name: str) -> ProviderConfig:
    return ProviderConfig(
        id=name,
        name=name,
        auth_type="api_key",
        api_key_env_vars=(f"{name.upper()}_API_KEY",),
    )


class TestSnapshotIteration:
    def test_in_flight_iterator_sees_only_the_snapshot(self):
        reg = _ConcurrentProviderRegistry({"p0": _row("p0")})
        it = iter(reg)
        reg["p1"] = _row("p1")
        assert sorted(it) == ["p0"]

    def test_items_values_keys_all_survive_insert_during_iteration(self):
        for view in ("items", "values", "keys"):
            reg = _ConcurrentProviderRegistry({"p0": _row("p0")})
            it = iter(getattr(reg, view)())
            reg["p1"] = _row("p1")
            assert list(it)  # must not raise; exact contents are the snapshot

    def test_bare_dict_counterpart_does_raise(self):
        """Counterproof that the guarded scenario is real: a bare dict's in-flight iterator
        deterministically blows up on a concurrent insert — the exact #134594 failure."""
        rows = {"p0": _row("p0")}
        it = iter(rows)
        rows["p1"] = _row("p1")
        with pytest.raises(RuntimeError):
            next(it)


class TestConcurrentRegistration:
    def test_reader_loops_never_raise_while_writer_registers(self):
        reg = _ConcurrentProviderRegistry({f"p{i}": _row(f"p{i}") for i in range(20)})
        stop = threading.Event()

        def writer() -> None:
            for i in range(200):
                if stop.is_set():
                    return
                reg[f"plugin-{i}"] = _row(f"plugin-{i}")

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            while thread.is_alive():
                sum(1 for _ in reg.items())
                list(reg.values())
                set(reg)
                sorted(reg.keys())
        finally:
            stop.set()
            thread.join()
        assert sum(1 for k in reg if k.startswith("plugin-")) == 200

    def test_register_plugin_provider_racing_an_iterating_reader(self, monkeypatch):
        """The issue's exact shape: ``register_plugin_provider`` on loader threads while a reader
        iterates the real registry object (monkeypatched so the global one stays pristine)."""
        from hermes_cli import auth, auth_plugin_providers

        monkeypatch.setattr(
            auth,
            "PROVIDER_REGISTRY",
            _ConcurrentProviderRegistry(dict(auth.PROVIDER_REGISTRY)),
        )

        class _Profile:
            aliases: tuple = ()

            def __init__(self, name: str) -> None:
                self.name = name
                self.auth_type = "api_key"
                self.env_vars = (f"{name.upper()}_API_KEY",)
                self.display_name = ""
                self.base_url = ""

        stop = threading.Event()

        def loader(n: int) -> None:
            i = 0
            while not stop.is_set() and i < 100:
                auth_plugin_providers.register_plugin_provider(
                    _Profile(f"plug-{n}-{i}")
                )
                i += 1

        threads = [threading.Thread(target=loader, args=(n,)) for n in range(3)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                sum(1 for _ in auth.PROVIDER_REGISTRY.items())
        finally:
            stop.set()
            for t in threads:
                t.join()
        assert sum(1 for k in auth.PROVIDER_REGISTRY if k.startswith("plug-")) == 300


class TestRegistryDictProtocol:
    """The fixture patterns used across the suite (copy via type(), clear+update restore, pop)
    keep working on the new storage."""

    def test_copy_clear_update_pop_roundtrip(self):
        reg = _ConcurrentProviderRegistry({"p0": _row("p0"), "p1": _row("p1")})
        snapshot = type(reg)(reg)
        reg.pop("p0")
        assert "p0" not in reg
        reg.clear()
        assert len(reg) == 0
        reg.update(snapshot)
        assert set(reg) == {"p0", "p1"}
        assert reg.get("missing") is None
        assert reg["p1"].id == "p1"

    def test_module_registry_is_the_concurrent_storage(self):
        assert isinstance(PROVIDER_REGISTRY, _ConcurrentProviderRegistry)
        assert "anthropic" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY.get("anthropic").auth_type
