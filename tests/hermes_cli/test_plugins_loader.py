import threading

import pytest

from hermes_cli import plugins_loader


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.3)
    abandoned = []

    class Context:
        def __init__(self, name):
            self.name = name

        def _abandon_load(self):
            abandoned.append(self.name)

    observed_threads = []
    release = threading.Event()

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline(
            "done", Context("done"), lambda: observed_threads.append(threading.current_thread()),
        )
        plugins_loader.run_with_load_deadline("hung", Context("hung"), release.wait)

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("outer", Context("outer"), outer_load)
    release.set()

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
    # The outer timeout abandons only contexts still loading, not a nested load that already finished.
    assert abandoned == ["outer", "hung"]


def test_default_load_timeout_stays_generous_and_consistent(tmp_path, monkeypatch):
    """The shipped default must stay generous to slow-but-legitimate imports (#140621): cold imports of
    bundled platform SDKs and the dashboard auth providers exceed 10s on slow storage, and an auth
    provider that times out keeps a gated (non-loopback) dashboard from starting at all for that boot.
    The config default and the loader constant are independent literals, so pin them to each other —
    they must not drift apart — and pin the resolver to the same value with no user config present."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert DEFAULT_CONFIG["plugins"]["load_timeout_seconds"] == plugins_loader._LOAD_TIMEOUT_SECS
    assert plugins_loader._LOAD_TIMEOUT_SECS >= 30.0
    assert plugins_loader._resolve_plugin_load_timeout() == plugins_loader._LOAD_TIMEOUT_SECS
