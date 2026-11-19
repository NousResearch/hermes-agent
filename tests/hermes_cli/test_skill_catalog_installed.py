"""Installed provider rows join by actual skills-root-relative path, never title."""
import asyncio
import json

from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginManager
from tests.hermes_cli.test_skill_catalog_contract import PLUGIN, make_plugin


def _skill(home, relative, name):
    path = home / "skills" / relative
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: local\n---\n\nApproved\n")


def _provider(home, target="nested/folder"):
    from hermes_cli.skill_catalog.identity import source_id
    identifier = source_id("fixture-catalog", "team") + ":example-skill"
    record = dict(identifier=identifier, name="Unrelated title", target_relative_path=target,
                  installed_version="1", latest_version="2", managed_state="installed",
                  installation_revision="install-1", actions=["update", "uninstall"])
    body = PLUGIN.replace('    def installed(self, request):\n        return {"ok": True, "data": {"items": [], "next_cursor": None}}',
                          '    def installed(self, request):\n        return ' + repr({"ok": True, "data": {"items": [record], "next_cursor": None}}))
    make_plugin(home, body=body)
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    return manager, record


def test_installed_joins_exact_path_and_keeps_stock_rows(_isolate_hermes_home):
    from hermes_cli.web_routers.skills import get_skills
    home = get_hermes_home()
    _skill(home, "nested/folder", "shown-name")
    _skill(home, "other", "Unrelated title")
    manager, record = _provider(home)
    try:
        rows = {row["name"]: row for row in asyncio.run(get_skills())}
        managed = rows["shown-name"]
        assert managed["provenance"] == "provider"
        for key, value in record.items():
            if key != "name":
                assert managed[key] == value
        assert managed["provider"] == dict(id="team", plugin_id="fixture-catalog", label="Team skills",
            version="1", author=None, managed_state="installed", scan_verdict="unknown",
            actions=["update", "uninstall"])
        assert managed["settlement"] == "settled"
        assert rows["Unrelated title"]["provenance"] == "agent"
        assert "provider" not in rows["Unrelated title"]
    finally:
        manager.unload()
    disabled = asyncio.run(get_skills())
    assert all(row["provenance"] == "agent" and "identifier" not in row for row in disabled)


def test_external_or_symlinked_paths_do_not_become_managed(_isolate_hermes_home, tmp_path):
    from hermes_cli.web_routers.skills import get_skills
    home = get_hermes_home()
    outside = tmp_path / "outside"
    _skill(outside, "actual", "outside-name")
    (home / "skills").mkdir(exist_ok=True)
    (home / "skills" / "link").symlink_to(outside / "skills" / "actual", target_is_directory=True)
    manager, _record = _provider(home, "link")
    try:
        rows = asyncio.run(get_skills())
        assert all("provider" not in row for row in rows)
    finally:
        manager.unload()


def test_empty_registry_does_not_call_provider_worker(_isolate_hermes_home, monkeypatch):
    from hermes_cli.skill_catalog import worker
    from hermes_cli.web_routers.skills import get_skills
    home = get_hermes_home()
    _skill(home, "local", "local")
    def forbidden(*args, **kwargs):
        raise AssertionError("providerless request created a worker")
    monkeypatch.setattr(worker, "call", forbidden)
    result = asyncio.run(get_skills())
    assert result == [dict(name="local", description="local", category=None, locked=False,
                           enabled=True, usage=0, provenance="agent")]
    assert json.dumps(result)
