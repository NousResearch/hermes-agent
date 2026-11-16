"""Snapshot-derived caches and reloads cannot publish revoked values as current."""
import threading
from concurrent.futures import ThreadPoolExecutor

from agent import secret_scope as ss
import hermes_secret_routing as routing

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_concurrent_proxy_builds_keep_values_and_version_together(store):
    provider, home = store
    provider.seed_external(home, {"RACE_SECRET": "initial-marker"})
    provider.refresh(home)
    entered, release = threading.Event(), threading.Event()
    blocked = [None]

    def builder(path):
        snapshot = routing._PROFILE_SNAPSHOT.get()[1]
        result = ss._build_profile_secret_scope(path)
        if snapshot.version == blocked[0]:
            entered.set()
            assert release.wait(10)
        return result

    scope = ss.LiveSecretScope.for_profile(home, provider, builder)
    provider._remote(home)["env"] = {"RACE_SECRET": "revoked-marker"}
    provider.refresh(home)
    blocked[0] = provider.current_snapshot(home).version
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(scope.get, "RACE_SECRET")
        try:
            assert entered.wait(10)
            provider._remote(home)["env"] = {}
            provider.refresh(home)
            def read_new():
                import sys

                def trace(frame, event, arg):
                    if (frame.f_code is ss.LiveSecretScope.refresh.__code__ and event == "line"
                            and frame.f_lineno == ss.LiveSecretScope.refresh.__code__.co_firstlineno + 10):
                        release.set()
                        old.result(timeout=10)
                    return trace

                sys.settrace(trace)
                try:
                    return scope.get("RACE_SECRET")
                finally:
                    sys.settrace(None)

            newer = pool.submit(read_new)
            assert newer.result(timeout=10) is None
        finally:
            release.set()
        old.result(timeout=10)
        assert newer.result(timeout=10) is None
    assert scope.get("RACE_SECRET") is None


def test_build_cannot_tag_old_values_with_a_new_snapshot(store):
    provider, home = store
    provider.seed_external(home, {"RACE_SECRET": "revoked-marker"})
    provider.refresh(home)
    changed = [False]

    def builder(path):
        result = ss._build_profile_secret_scope(path)
        if not changed[0]:
            changed[0] = True
            provider._remote(home)["env"] = {}
            provider.refresh(home)
        return result

    scope = ss.LiveSecretScope.for_profile(home, provider, builder)
    assert scope.get("RACE_SECRET") is None


def test_reload_reconciles_a_removed_name(store, monkeypatch):
    import os
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_secret_port import SecretSnapshot

    provider, home = store
    monkeypatch.delenv("REMOVED_SECRET", raising=False)
    provider.seed_external(home, {"REMOVED_SECRET": "removed-marker"})
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    cell = provider.cells[provider._key(home)]
    cell.install(SecretSnapshot(cell.current.version + 1, {}, {}))
    monkeypatch.setattr(provider, "boot", lambda home: None)
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert "REMOVED_SECRET" not in os.environ


def test_randomized_snapshot_publication_converges_without_revoked_values(store, monkeypatch):
    import random
    from hermes_secret_port import SecretSnapshot
    from hermes_cli.env_loader import load_hermes_dotenv

    provider, home = store
    provider.refresh(home)
    scope = ss.build_profile_secret_scope(home)
    cell = provider.cells[provider._key(home)]
    start = threading.Barrier(9)
    monkeypatch.setattr(provider, "boot", lambda home: None)

    def worker(seed):
        rng = random.Random(seed)
        start.wait(timeout=10)
        for index in range(50):
            version = 2 + seed * 50 + index
            values = {"STRESS_SECRET": "stress-marker"} if rng.randrange(2) else {}
            snapshot = SecretSnapshot(version, values, {})
            cell.install(snapshot)
            if rng.randrange(2):
                routing.publish(snapshot, home=home, consumer=scope._install_snapshot)
            else:
                load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
            current = cell.current
            if not current.env:
                assert scope.get("STRESS_SECRET") is None or cell.current.version != current.version

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, seed) for seed in range(8)]
        start.wait(timeout=10)
        for future in futures:
            future.result(timeout=20)
    latest = SecretSnapshot(1000, {}, {})
    cell.install(latest)
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert scope.get("STRESS_SECRET") is None
