import pytest

import plugins.memory.hindsight as hindsight_module
from plugins.memory.hindsight import HindsightMemoryProvider


def _set_config(monkeypatch, config):
    monkeypatch.setattr(hindsight_module, "_load_config", lambda: config)


def _mark_local_deps_missing(monkeypatch):
    monkeypatch.setattr(
        hindsight_module,
        "_missing_local_dependencies",
        lambda: ["hindsight", "hindsight_embed"],
        raising=False,
    )


def _mark_cloud_client_missing(monkeypatch):
    monkeypatch.setattr(hindsight_module, "_has_cloud_client", lambda: False, raising=False)


@pytest.mark.parametrize(
    ("config", "mark_missing"),
    [
        ({"mode": "local"}, _mark_local_deps_missing),
        ({"mode": "cloud", "apiKey": "test-key"}, _mark_cloud_client_missing),
    ],
)
def test_is_available_false_when_required_dependency_missing(monkeypatch, config, mark_missing):
    _set_config(monkeypatch, config)
    mark_missing(monkeypatch)

    provider = HindsightMemoryProvider()

    assert provider.is_available() is False


def test_initialize_raises_before_starting_local_thread_when_embedded_deps_missing(monkeypatch):
    _set_config(monkeypatch, {"mode": "local"})
    _mark_local_deps_missing(monkeypatch)

    started = False

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            nonlocal started
            started = True

    monkeypatch.setattr(hindsight_module.threading, "Thread", FakeThread)

    provider = HindsightMemoryProvider()

    with pytest.raises(RuntimeError, match="Run `hermes memory setup`"):
        provider.initialize("session-1")

    assert started is False


def test_initialize_raises_when_cloud_client_missing(monkeypatch):
    _set_config(monkeypatch, {"mode": "cloud", "apiKey": "test-key"})
    _mark_cloud_client_missing(monkeypatch)

    provider = HindsightMemoryProvider()

    with pytest.raises(RuntimeError, match="hindsight_client"):
        provider.initialize("session-1")
