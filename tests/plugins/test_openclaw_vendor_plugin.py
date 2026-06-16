"""Tests for the openclaw-vendor Hermes plugin bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "openclaw-vendor"
REPO_ROOT = PLUGIN_DIR.parents[1]


def load_plugin():
    package_name = "openclaw_vendor_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def __init__(self) -> None:
        self.cli_commands = {}

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def test_register_exposes_cli_only(tmp_path, monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(module.core, "plugin_dir", lambda: PLUGIN_DIR)
    monkeypatch.setattr(
        module.core,
        "get_hermes_home",
        lambda: tmp_path / ".hermes",
    )
    monkeypatch.setattr(
        module.core,
        "display_hermes_home",
        lambda: str(tmp_path / ".hermes"),
    )

    ctx = _FakeContext()
    module.register(ctx)
    assert "openclaw-vendor" in ctx.cli_commands


def test_read_skill_name_from_frontmatter(tmp_path):
    module = load_plugin()
    skill_dir = tmp_path / "hypura"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: hypura-harness\n---\n",
        encoding="utf-8",
    )
    assert module.core.read_skill_name(skill_dir) == "hypura-harness"


def test_discover_skills_from_fake_mirror(tmp_path, monkeypatch):
    module = load_plugin()
    mirror = tmp_path / "vendor" / "openclaw-mirror"
    ext = mirror / "extensions" / "hypura-harness" / "skills" / "hypura"
    ext.mkdir(parents=True)
    (ext / "SKILL.md").write_text(
        "---\nname: hypura-harness\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module.core, "vendor_mirror_root", lambda: mirror)
    monkeypatch.setattr(
        module.core,
        "get_hermes_home",
        lambda: tmp_path / ".hermes",
    )

    sources = module.core.discover_skill_sources()
    assert len(sources) == 1
    assert sources[0]["skill_name"] == "hypura-harness"
    assert sources[0]["extension"] == "hypura-harness"


def test_install_links_skills(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    mirror = tmp_path / "vendor" / "openclaw-mirror"
    skill_src = mirror / "extensions" / "hypura-harness" / "skills" / "hypura"
    skill_src.mkdir(parents=True)
    (skill_src / "SKILL.md").write_text(
        "---\nname: hypura-harness\n---\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module.core, "vendor_mirror_root", lambda: mirror)
    monkeypatch.setattr(module.core, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: home)
    monkeypatch.setattr(module.core, "display_hermes_home", lambda: str(home))

    result = module.core.install(force=True)
    assert result["ok"] is True
    assert (home / "skills" / "hypura-harness" / "SKILL.md").is_file()


def test_list_units_includes_packages_when_mirror_present(monkeypatch):
    module = load_plugin()
    mirror = REPO_ROOT / "vendor" / "openclaw-mirror"
    if not mirror.is_dir():
        return
    monkeypatch.setattr(module.core, "repo_root", lambda: REPO_ROOT)
    listing = module.core.list_units()
    assert listing["ok"] is True
    assert listing["mirror_present"] is True
    pkg_ids = {p["id"] for p in listing["packages"]}
    assert "AI-Scientist" in pkg_ids
    assert "ShinkaEvolve" in pkg_ids
    ext_ids = {e["id"] for e in listing["extensions"]}
    assert "hypura-harness" in ext_ids
