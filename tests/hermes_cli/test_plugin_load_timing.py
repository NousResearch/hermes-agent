"""Local import/register timing must preserve loader results and profile isolation."""

import logging
from types import SimpleNamespace

import pytest

from hermes_cli import plugins_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("timeout,outcome", [(0, "success"), (10, "success"), (0, "error"), (10, "error"), (10, "timeout")])
def test_load_timing_preserves_outcome(monkeypatch, caplog, timeout, outcome):
    now = [100.0]
    monkeypatch.setattr(plugins_loader, "monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: timeout)
    caplog.set_level(logging.INFO, logger="hermes_cli.plugins")
    failure = ValueError("private credential /private/plugin/source")
    abandoned = []
    ctx = SimpleNamespace(_abandon_load=lambda: abandoned.append(True))
    sentinel = object()

    def load():
        now[0] += 0.125
        if outcome == "error":
            raise failure
        return sentinel

    if outcome == "timeout":
        # Deterministic timeout decision, with no live abandoned thread or sleeps.
        class HungWorker:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def join(self, seconds):
                now[0] += seconds

            def is_alive(self):
                return True

        monkeypatch.setattr(plugins_loader.threading, "Thread", HungWorker)
        monkeypatch.setattr(plugins_loader, "_ABANDONED_LOADERS", [])
        with pytest.raises(plugins_loader.PluginLoadTimeout):
            plugins_loader.run_with_load_deadline("demo", ctx, load)
        assert abandoned == [True]
    elif outcome == "error":
        with pytest.raises(ValueError) as caught:
            plugins_loader.run_with_load_deadline("demo", ctx, load)
        assert caught.value is failure
    else:
        assert plugins_loader.run_with_load_deadline("demo", ctx, load) is sentinel

    records = [r for r in caplog.records if r.msg.startswith("Plugin load ")]
    assert len(records) == 1
    assert records[0].args[0] == "demo"
    assert records[0].args[2] == outcome
    assert records[0].args[3] == pytest.approx(10000.0 if outcome == "timeout" else 125.0)
    assert "private credential" not in records[0].getMessage()
    assert "/private/plugin/source" not in records[0].getMessage()


def test_nested_real_load_timing_is_profile_scoped(tmp_path, monkeypatch, caplog):
    """Real imports/register(), nested workers and A→B→A retain their owning profile."""
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 10)
    caplog.set_level(logging.INFO, logger="hermes_cli.plugins")
    plugin = tmp_path / "fixture"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(
        "from hermes_cli.plugins_loader import run_with_load_deadline\n"
        "def register(ctx):\n"
        "    run_with_load_deadline('nested', ctx, lambda: 42)\n",
        encoding="utf-8",
    )
    homes = [root / "profiles" / "alpha", root / "profiles" / "beta"]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(home)
        try:
            manager = PluginManager(scope_key=str(home))
            manifest = PluginManifest(name="timing-fixture", path=plugin, source="user")
            manager._load_plugin(manifest)
            assert manager._plugins["timing-fixture"].enabled
        finally:
            reset_hermes_home_override(token)
    records = [r for r in caplog.records if r.msg.startswith("Plugin load ")]
    assert [(r.args[0], r.args[1], r.args[2]) for r in records] == [
        (key, profile, "success") for profile in ("alpha", "beta", "alpha")
        for key in ("nested", "timing-fixture")
    ]
    assert all(r.args[3] >= 0 for r in records)
    assert str(tmp_path) not in "\n".join(r.getMessage() for r in records)
