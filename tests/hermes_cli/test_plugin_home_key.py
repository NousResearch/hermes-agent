"""Regression coverage for the per-profile plugin-manager cache key."""

from pathlib import Path

from hermes_cli import plugins


def test_plugin_home_key_resolution_is_cached(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    plugins._plugin_home_key_cache.clear()
    original_resolve = plugins.Path.resolve
    calls = 0

    def counting_resolve(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(plugins.Path, "resolve", counting_resolve)

    assert plugins._plugin_home_key_for(str(home)) == home
    first_call_count = calls
    assert plugins._plugin_home_key_for(str(home)) == home
    assert calls == first_call_count


def test_plugin_home_key_preserves_unresolved_path_when_resolution_fails(monkeypatch, tmp_path):
    home = tmp_path / "unavailable"
    plugins._plugin_home_key_cache.clear()
    original_resolve = plugins.Path.resolve
    attempts = 0

    def unavailable_then_available(path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("unavailable")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(plugins.Path, "resolve", unavailable_then_available)

    assert plugins._plugin_home_key_for(str(home)) == Path(home)
    assert plugins._plugin_home_key_for(str(home)) == home
    assert attempts == 2
