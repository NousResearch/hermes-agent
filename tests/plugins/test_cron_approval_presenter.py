"""Load the real plugin API under two isolated homes and retire its presenters."""

import json
from pathlib import Path

import pytest

from cron.approval_relay import presenters_for
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


PLUGIN = '''
from contextlib import contextmanager
from hermes_constants import get_hermes_home

def register(ctx):
    @contextmanager
    def present(request, lifetime):
        assert str(get_hermes_home().resolve()) == request.profile_home
        yield
    ctx.cron_approvals.register_presenter(present)
'''


def test_discovery_and_unload_keep_profile_presenters_separate(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / "first", tmp_path / "second"]
    managers = []
    for home in homes:
        root = home / "plugins" / "guard"
        root.mkdir(parents=True)
        (root / "plugin.yaml").write_text("name: guard\nversion: 0.1.0\n")
        (root / "__init__.py").write_text(PLUGIN)
        (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["guard"]}}))
        monkeypatch.setenv("HERMES_HOME", str(home))
        token = set_hermes_home_override(home)
        try:
            manager = PluginManager()
            manager.discover_and_load()
            assert "guard" in presenters_for(str(home))
            managers.append(manager)
        finally:
            reset_hermes_home_override(token)
    try:
        first = presenters_for(str(homes[0]))["guard"]
        second = presenters_for(str(homes[1]))["guard"]
        assert first is not second
        managers[0].unload("guard")
        assert first.closed.is_set()
        assert not second.closed.is_set()
        assert presenters_for(str(homes[0])) == {}
    finally:
        for manager in managers:
            manager.unload("guard")


def test_abandoned_context_cannot_replace_live_presenter(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manager = PluginManager()
    manifest = PluginManifest(name="guard", version="0.1.0")
    abandoned = PluginContext(manifest, manager)
    stale_service = abandoned.cron_approvals
    abandoned._abandon_load()
    current = PluginContext(manifest, manager)
    live = current.cron_approvals.register_presenter(lambda *_: None)
    try:
        with pytest.raises(RuntimeError):
            stale_service.register_presenter(lambda *_: None)
        assert not live.closed.is_set()
        assert presenters_for(str(tmp_path)) == {"guard": live}
    finally:
        manager.unload("guard")
        live.dispose()


def test_unloaded_service_cannot_reregister(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manager = PluginManager()
    service = PluginContext(PluginManifest(name="guard", version="0.1.0"), manager).cron_approvals
    registration = service.register_presenter(lambda *_: None)
    manager.unload("guard")
    assert registration.closed.is_set()
    with pytest.raises(RuntimeError):
        service.register_presenter(lambda *_: None)
