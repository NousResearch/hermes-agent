"""Real default, custom and active-profile roots all discover sibling members."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml
import pytest

import pm.plugins_state as pstate
import pm.workspace as ws


def _write_enabled(home: Path, enabled: list) -> None:
    home.mkdir(parents=True, exist_ok=True)
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"plugins": {"enabled": enabled}}, f)


def _make_dep_plugin(plugins_dir: Path, name: str) -> Path:
    plug = plugins_dir / name
    plug.mkdir(parents=True)
    (plug / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (plug / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["pyfiglet==1.0.2"]\n',
        encoding="utf-8",
    )
    return plug


@pytest.mark.parametrize("layout", ["default", "custom", "profile"])
def test_home_layout_joins_sibling_union(tmp_path, monkeypatch, layout):
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / (".hermes" if layout == "default" else "data-root")
    active = root / "profiles/active" if layout == "profile" else root
    if layout == "default":
        monkeypatch.delenv("HERMES_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_HOME", str(active))
    active.mkdir(parents=True)
    sibling = root / "profiles/sibling"
    _write_enabled(sibling, ["dep-plug"])
    member = _make_dep_plugin(sibling / "plugins", "dep-plug")
    (root / "profiles/README.txt").write_text("not a profile")
    assert hermes_constants.get_default_hermes_root() == root
    assert ws.enabled_member_dirs() == [member]
    assert pstate.enabled_plugins_ordered() == {sibling / "plugins": ["dep-plug"]}
