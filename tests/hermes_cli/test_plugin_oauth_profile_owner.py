"""Profile copies share a path-derived owner, not a manifest's claimed name."""
import json
from pathlib import Path

import pytest

from hermes_cli.plugin_oauth import _declared_clients
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tests.hermes_cli.test_platform_plugin_oauth import _plugin, _entry


@pytest.mark.parametrize("launch,selected", [("default", "goonbot"), ("goonbot", "default"), ("goonbot", "sage")])
def test_same_plugin_across_bound_profiles_keeps_own_oauth_variables(tmp_path, monkeypatch, launch, selected):
    root = tmp_path / ".hermes"
    homes = {"default": root, "goonbot": root / "profiles/goonbot", "sage": root / "profiles/sage"}
    dirs = {}
    for name, home in homes.items():
        dirs[name] = _plugin(home, "native", _entry("https://login.example.com"),
                             names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET"))
        (home / "config.yaml").write_text("{}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(homes[launch]))
    assert len(_declared_clients(dirs[launch])) == 1
    token = set_hermes_home_override(homes[selected])
    try:
        assert len(_declared_clients(dirs[launch])) == 1
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("other", ["impostor", "category/native"])
def test_profile_copy_does_not_exempt_other_directory_or_claimed_same_name(tmp_path, monkeypatch, other):
    root = tmp_path / ".hermes"
    selected = root / "profiles/goonbot"
    own = _plugin(root, "native", _entry("https://login.example.com"),
                  names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET"))
    _plugin(selected, "native", _entry("https://login.example.com"),
            names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET"))
    foreign = _plugin(selected, other, names=("NATIVE_CLIENT_SECRET",))
    manifest = json.loads((foreign / "plugin.yaml").read_text())
    manifest["name"] = "native"  # A self-reported name must not grant ownership.
    (foreign / "plugin.yaml").write_text(json.dumps(manifest))
    (root / "config.yaml").write_text("{}")
    (selected / "config.yaml").write_text("{}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(selected)
    try:
        with pytest.raises(ValueError, match="NATIVE_CLIENT_SECRET is declared by another plugin"):
            _declared_clients(own)
    finally:
        reset_hermes_home_override(token)
