"""Tests for the book-to-skill Hermes plugin bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "book-to-skill"


def load_plugin():
    package_name = "book_to_skill_test_plugin"
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
    assert "book-to-skill" in ctx.cli_commands


def test_materialize_skill_symlink(tmp_path, monkeypatch):
    module = load_plugin()
    src = tmp_path / "upstream"
    src.mkdir()
    (src / "SKILL.md").write_text("---\nname: book-to-skill\n---\n", encoding="utf-8")
    dst = tmp_path / "skills" / "book-to-skill"

    result = module.core._materialize_skill_source(src, dst)
    assert result["ok"] is True
    assert dst.exists()
    assert (dst / "SKILL.md").is_file()


def test_sync_requires_vendor(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(module.core, "vendor_root", lambda: home / "missing-vendor")
    monkeypatch.setattr(module.core, "user_skill_path", lambda: home / "skills" / "book-to-skill")

    result = module.core.sync_skill_link()
    assert result["ok"] is False
    assert "install" in result["error"].lower()


def test_install_links_fake_vendor(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    vendor = tmp_path / "vendor" / "book-to-skill"
    vendor.mkdir(parents=True)
    (vendor / "SKILL.md").write_text("---\nname: book-to-skill\n---\n", encoding="utf-8")

    monkeypatch.setattr(module.core, "clone_upstream", lambda **_: {"ok": True, "skipped": True})
    monkeypatch.setattr(module.core, "vendor_root", lambda: vendor)
    monkeypatch.setattr(module.core, "user_skill_path", lambda: home / "skills" / "book-to-skill")
    monkeypatch.setattr(module.core, "generated_skills_dir", lambda: home / "skills")
    monkeypatch.setattr(module.core, "display_hermes_home", lambda: str(home))

    result = module.core.install()
    assert result["ok"] is True
    assert (home / "skills" / "book-to-skill" / "SKILL.md").is_file()
