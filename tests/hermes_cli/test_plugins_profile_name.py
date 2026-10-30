"""Plugin context profile-name behavior tests."""

from pathlib import Path

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class TestPluginContextProfileName:
    """ctx.profile_name resolves from HERMES_HOME in every context."""

    def _ctx(self):
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        return PluginContext(manifest, mgr)

    def test_default_profile(self, tmp_path, monkeypatch):
        """HERMES_HOME at the root resolves to 'default'."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert self._ctx().profile_name == "default"

    def test_named_profile(self, tmp_path, monkeypatch):
        """HERMES_HOME under profiles/<name> resolves to that name."""
        prof = tmp_path / ".hermes" / "profiles" / "coder"
        prof.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(prof))
        assert self._ctx().profile_name == "coder"
