"""The plugin-host model-provider cache answers the caller but never recreates a profile home
that ``hermes profile delete``/``rename`` already removed."""

from __future__ import annotations

import pytest

from hermes_cli import plugin_host_profiles


def _plugin_dir(tmp_path):
    plugin_dir = tmp_path / "plug" / "providerx"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    return plugin_dir


@pytest.mark.parametrize("move", ("delete", "rename"))
def test_cache_write_refuses_a_dead_profile_home(tmp_path, monkeypatch, move):
    from tests.fixtures.dead_profile_home import dead_named_home

    home = dead_named_home(tmp_path / "dead", move)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugin_host_profiles, "_extract",
                        lambda _dir, name: {"profiles": [{"name": "x"}], "module_name": name})

    payload = plugin_host_profiles._cached_extraction(_plugin_dir(tmp_path), "providerx")

    assert payload == {"profiles": [{"name": "x"}], "module_name": "providerx"}
    assert not home.exists()


def test_cache_still_written_and_reused_for_a_live_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "live" / ".hermes" / "profiles" / "b"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    extracted = []
    monkeypatch.setattr(plugin_host_profiles, "_extract",
                        lambda _dir, name: extracted.append(name) or {"profiles": [{"name": "x"}], "module_name": name})

    plugin_dir = _plugin_dir(tmp_path)
    plugin_host_profiles._cached_extraction(plugin_dir, "providerx")
    plugin_host_profiles._cached_extraction(plugin_dir, "providerx")

    assert len(list((home / "cache" / "plugin_host" / "model-providers").glob("providerx-*.json"))) == 1
    assert extracted == ["providerx"], "the second listing is served from the cache, not re-extracted"