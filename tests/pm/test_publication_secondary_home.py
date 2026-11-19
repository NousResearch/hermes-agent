"""Plugin publication accepts every dependency home, not only the default root.

``dependency_homes()`` is the default home plus each live profile. A profile may resolve outside
the default root (for example ``profiles/<name>`` symlinked to another volume), so validating a
publication against ``dependency_home_root()`` alone refused installs into that profile. Paths
outside every dependency home are still refused.
"""
import pytest

from pm.publication import PluginSelection, StagedPlugin

pytestmark = pytest.mark.platforms("linux", "macos")


def _symlinked_profile(tmp_path, monkeypatch):
    """A live profile whose real directory is outside the default home."""
    home = tmp_path / "home"
    (home / "profiles").mkdir(parents=True)
    external = tmp_path / "external" / "worker"
    (external / "plugins").mkdir(parents=True)
    (external / "config.yaml").write_text("plugins:\n  enabled: [old]\n", encoding="utf-8")  # identity marker
    profile = home / "profiles" / "worker"
    profile.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return profile


def _staged(tmp_path, name="probe"):
    staged = tmp_path / "staging" / name
    (staged / "pkg").mkdir(parents=True)
    (staged / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (staged / "pkg" / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    return staged


def _plugin(staged, target, name="probe"):
    return {
        "staged": str(staged), "target": str(target), "target_digest": None,
        "old_metadata": {}, "new_metadata": {name: {"revision": "new"}},
    }


def test_staged_plugin_accepts_a_target_in_a_secondary_home(tmp_path, monkeypatch):
    profile = _symlinked_profile(tmp_path, monkeypatch)
    target = profile / "plugins" / "probe"

    plugin = StagedPlugin(_plugin(_staged(tmp_path), target))

    assert plugin.target == target


def test_staged_plugin_outside_every_home_is_refused(tmp_path, monkeypatch):
    _symlinked_profile(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere" / "plugins"
    elsewhere.mkdir(parents=True)

    with pytest.raises(ValueError, match="escape or overlap their home"):
        StagedPlugin(_plugin(_staged(tmp_path), elsewhere / "probe"))


def test_plugin_selection_accepts_a_secondary_home(tmp_path, monkeypatch):
    profile = _symlinked_profile(tmp_path, monkeypatch)

    selection = PluginSelection({"home": str(profile), "enabled": ["new"], "disabled": []})

    assert selection.path == profile.resolve() / "config.yaml"


def test_plugin_selection_outside_every_home_is_refused(tmp_path, monkeypatch):
    _symlinked_profile(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(ValueError, match="outside Hermes state"):
        PluginSelection({"home": str(elsewhere), "enabled": ["new"], "disabled": []})
