"""Installed provider rows join by actual skills-root-relative path, never title."""
import asyncio
import json

import pytest

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
        from hermes_cli.skill_catalog.validation import CatalogError
        with pytest.raises(CatalogError) as caught:
            asyncio.run(get_skills())
        assert caught.value.code == 'ownership_state_unavailable'
        assert (outside / 'skills/actual/SKILL.md').is_file()
    finally:
        manager.unload()


@pytest.mark.parametrize('failure', ['invalid_scan', 'read_error'])
def test_live_failed_projection_never_downgrades_managed_rows_to_local(_isolate_hermes_home, monkeypatch, failure):
    from hermes_cli.web_routers.skills import get_skills
    from hermes_cli.skill_catalog.registry import snapshot
    from hermes_cli.skill_catalog.validation import CatalogError
    home = get_hermes_home()
    _skill(home, 'nested/folder', 'shown-name')
    manager, _record = _provider(home)
    entry, = snapshot()
    saved = entry.mutations.projection.directory.joinpath('claims.json').read_text()
    try:
        if failure == 'invalid_scan':
            # A fresh installed read cannot grant ownership after its projection lost authority.
            (entry.mutations.projection.directory / 'claims.json').write_text('{corrupt')
        else:
            def unavailable(request):
                raise RuntimeError('provider offline')
            monkeypatch.setattr(entry.provider, 'installed', unavailable)
        with pytest.raises(CatalogError) as caught:
            asyncio.run(get_skills())
        assert caught.value.code == 'ownership_state_unavailable'
    finally:
        if failure == 'invalid_scan':
            # Restore the known document only to let the ordinary lifecycle release leases.
            entry.mutations.projection.directory.joinpath('claims.json').write_text(saved)
        manager.unload()


def test_installed_row_reports_reconciling_local_commit(_isolate_hermes_home):
    import uuid
    from hermes_cli.web_routers.skills import get_skills
    from hermes_cli.skill_catalog.registry import snapshot
    from tests.hermes_cli.test_skill_catalog_projection import review
    home = get_hermes_home()
    _skill(home, 'nested/folder', 'shown-name')
    manager, record = _provider(home)
    entry, = snapshot()
    try:
        plan = review(entry, target='nested/folder', action='update',
                      identifier=record['identifier'], installation_revision='install-1')
        admission = entry.mutations.projection.reserve(entry, plan)
        op = uuid.uuid4().hex
        entry.mutations.bind({key: value for key, value in admission.items() if key != 'generation'}
                             | {'operation_id': op})
        with entry.mutations.commit(admission, op) as commit:
            commit.settle(dict(operation_id=op, identifier=record['identifier'], action='update',
                state='pending', outcome=None, managed_state='installed', message='',
                installation_revision='install-2'))
        rows = asyncio.run(get_skills())
        assert rows[0]['settlement'] == 'reconciling'
        assert rows[0]['provenance'] == 'provider'
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
