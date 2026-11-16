"""The env mirror rejects a stale snapshot even when explicit loads race."""

import os
import threading

import hermes_secret_routing as routing
from hermes_cli.env_loader import load_hermes_dotenv

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_late_snapshot_cannot_revert_process_env(store, monkeypatch):
    provider, home = store
    provider.seed_external(home, {"PUBLISH_KEY": "first-marker"})
    monkeypatch.delenv("PUBLISH_KEY", raising=False)
    load_hermes_dotenv(hermes_home=home)
    provider.seed_external(home, {"PUBLISH_KEY": "second-marker"})
    provider.refresh(home)
    monkeypatch.setattr(provider, "boot", lambda home: None)
    stale = provider.current_snapshot(home)
    entered, release = threading.Event(), threading.Event()
    original = provider.current_snapshot
    errors = []

    def snapshot(path):
        if threading.current_thread() is late:
            entered.set()
            assert release.wait(10)
            return stale
        return original(path)

    def publish():
        try:
            load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(provider, "current_snapshot", snapshot)
    late = threading.Thread(target=publish, daemon=True)
    late.start()
    try:
        assert entered.wait(10)
        provider.seed_external(home, {"PUBLISH_KEY": "third-marker"})
        provider.refresh(home)
        load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    finally:
        release.set()
        late.join(10)
    assert not errors and not late.is_alive()
    assert os.environ["PUBLISH_KEY"] == "third-marker"
    assert routing._PUBLICATIONS[provider._key(home)][1]["PUBLISH_KEY"] == "third-marker"
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["PUBLISH_KEY"] == "third-marker"
