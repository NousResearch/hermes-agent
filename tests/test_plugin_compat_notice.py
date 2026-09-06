"""hermes_cli.plugin_compat: detect plugins on old import paths, tell the user, disable after the date.

Kept with the compat layer (tests/test_compat_manifest_targets.py); deleted with it.
"""
from __future__ import annotations

import datetime as dt
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_compat as pc

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "compat_manifest.json").exists(), reason="compat layer removed")

MANIFEST = {"tools.web_tools": {"prefers_gateway": "tools.tool_backend_helpers.prefers_gateway"},
            "hermes_cli.kanban_db": {"connect": "hermes_cli.kanban_db_connect.connect"}}


@pytest.mark.parametrize("src, expect", [
    ("from tools.web_tools import prefers_gateway\n", ["tools.web_tools.prefers_gateway"]),
    ("import tools.web_tools\nx = tools.web_tools.prefers_gateway()\n", ["tools.web_tools.prefers_gateway"]),
    ("import tools.web_tools as wt\nwt.prefers_gateway()\n", ["tools.web_tools.prefers_gateway"]),
    ("from unittest.mock import patch\npatch('hermes_cli.kanban_db.connect')\n", ["hermes_cli.kanban_db.connect"]),
    ("from tools.web_tools import web_search\n", []),                       # live name: not a hit
    ("from tools.tool_backend_helpers import prefers_gateway\n", []),      # already migrated
])
def test_scan_source_finds_every_import_form(src, expect):
    hits = pc.scan_source(src, "p.py", MANIFEST)
    assert [h.old for h in hits] == expect
    for h in hits:
        assert h.new == MANIFEST[h.old.rsplit(".", 1)[0]][h.old.rsplit(".", 1)[1]]


def test_scan_plugin_walks_dir_and_skips_tests(tmp_path):
    (tmp_path / "__init__.py").write_text("from tools.web_tools import prefers_gateway\n")
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "m.py").write_text("import hermes_cli.kanban_db as k\nk.connect()\n")
    (tmp_path / "tests").mkdir(); (tmp_path / "tests" / "t.py").write_text("from tools.web_tools import prefers_gateway\n")
    hits = pc.scan_plugin(tmp_path, MANIFEST)
    assert sorted(h.file for h in hits) == ["__init__.py", "sub/m.py"]


def _manifest(name, path, source="user"):
    return SimpleNamespace(name=name, path=str(path), source=source)


def test_compat_report_only_external_plugins_with_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "load_manifest", lambda: MANIFEST)
    monkeypatch.setattr(pc, "_write_report_file", lambda r: None)
    good = tmp_path / "good"; good.mkdir(); (good / "__init__.py").write_text("x = 1\n")
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "__init__.py").write_text("from tools.web_tools import prefers_gateway\n")
    bundled = tmp_path / "bundled"; bundled.mkdir(); (bundled / "__init__.py").write_text("from tools.web_tools import prefers_gateway\n")
    report = pc.compat_report([_manifest("good", good), _manifest("bad", bad), _manifest("ours", bundled, "bundled")], force=True)
    assert list(report) == ["bad"] and report["bad"][0].old == "tools.web_tools.prefers_gateway"


def test_disable_only_after_the_date_and_not_when_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "load_manifest", lambda: MANIFEST)
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "__init__.py").write_text("from tools.web_tools import prefers_gateway\n")
    m = _manifest("bad", bad)
    before, after = pc.COMPAT_REMOVAL_DATE - dt.timedelta(days=1), pc.COMPAT_REMOVAL_DATE
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: False)
    assert pc.disable_reason(m, today=before) is None
    reason = pc.disable_reason(m, today=after)
    assert reason and pc.COMPAT_REMOVAL in reason and "hermes plugins compat" in reason
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: True)
    assert pc.disable_reason(m, today=after) is None
    good = tmp_path / "good"; good.mkdir(); (good / "__init__.py").write_text("x=1\n")
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: False)
    assert pc.disable_reason(_manifest("good", good), today=after) is None


def test_summary_lines_name_plugins_and_the_date():
    report = {"alpha": [pc.Hit("a.py", 1, "x.y", "z.y")], "beta": [pc.Hit("b.py", 2, "x.y", "z.y"), pc.Hit("b.py", 3, "x.q", "z.q")]}
    before = pc.COMPAT_REMOVAL_DATE - dt.timedelta(days=3)
    head, tail = pc.summary_lines(report, today=before)
    assert "2 plugins" in head and "alpha (1)" in head and "beta (2)" in head and pc.COMPAT_REMOVAL in head and "3 days" in head
    assert "hermes plugins compat" in tail
    head_after, _ = pc.summary_lines(report, today=pc.COMPAT_REMOVAL_DATE)
    assert "DISABLED" in head_after
    assert pc.summary_lines({}) == []


def test_report_file_written_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "report_file_path", lambda: tmp_path / "r.json")
    pc._write_report_file({"p": [pc.Hit("a.py", 1, "x.y", "z.y")]})
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["plugins"]["p"][0]["old"] == "x.y" and data["removal_date"] == pc.COMPAT_REMOVAL and len(data["lines"]) == 2
    pc._write_report_file({})
    assert not (tmp_path / "r.json").exists()


def test_loader_skips_hitting_plugin_after_date(tmp_path, monkeypatch):
    """PluginManager records the reason and never imports the plugin."""
    from hermes_cli.plugins import PluginManager
    monkeypatch.setattr(pc, "load_manifest", lambda: MANIFEST)
    monkeypatch.setattr(pc, "removal_in_effect", lambda today=None: True)
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: False)
    plugin = tmp_path / "plugins" / "oldpaths"; plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: oldpaths\nversion: 0.1\ndescription: t\n")
    (plugin / "__init__.py").write_text(textwrap.dedent("""
        from tools.web_tools import prefers_gateway
        LOADED = True
        def register(ctx):
            raise AssertionError("must not be imported/registered")
    """))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mgr = PluginManager(scope_key=str(tmp_path))
    from hermes_cli.plugins_manifest import PluginManifest
    real = PluginManifest(name="oldpaths", version="0.1", description="t", source="user", path=str(plugin))
    mgr._load_plugin(real)
    loaded = next(lp for lp in mgr._plugins.values() if lp.manifest.name == "oldpaths")
    assert not loaded.enabled and loaded.error and pc.COMPAT_REMOVAL in loaded.error


def test_discovery_refreshes_report_file(tmp_path, monkeypatch):
    """The Desktop modal reads the report the `serve` backend's discovery wrote — discovery itself must
    write it (not only the CLI banner / doctor / update paths), and clear it once the plugin is fixed."""
    from hermes_cli.plugins import PluginManager
    monkeypatch.setattr(pc, "load_manifest", lambda: MANIFEST)
    monkeypatch.setattr(pc, "removal_in_effect", lambda today=None: False)
    monkeypatch.setattr(pc, "report_file_path", lambda: tmp_path / "r.json")
    plugin = tmp_path / "plugins" / "oldpaths"; plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: oldpaths\nversion: 0.1\ndescription: t\n")
    (plugin / "__init__.py").write_text("from tools.web_tools import prefers_gateway\ndef register(ctx):\n    pass\n")
    from hermes_cli.plugins_manifest import PluginManifest
    real = PluginManifest(name="oldpaths", version="0.1", description="t", source="user", path=str(plugin))
    mgr = PluginManager(scope_key=str(tmp_path))
    mgr._refresh_plugin_compat_report([real])
    data = json.loads((tmp_path / "r.json").read_text())
    assert list(data["plugins"]) == ["oldpaths"] and data["in_effect"] is False
    (plugin / "__init__.py").write_text("from tools.tool_backend_helpers import prefers_gateway\ndef register(ctx):\n    pass\n")
    mgr._refresh_plugin_compat_report([real])
    assert not (tmp_path / "r.json").exists()


def test_scan_root_never_falls_back_to_cwd(tmp_path, monkeypatch):
    """Windows dir paths and ``module:attr`` entry points used to collapse to ``.`` and scan the
    launch directory; an entry point must resolve to its installed package, everything else to None."""
    monkeypatch.setattr(pc, "load_manifest", lambda: MANIFEST)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stray.py").write_text("from tools.web_tools import prefers_gateway\n")
    assert pc.plugin_hits(SimpleNamespace(source="directory", path=r"C:\Users\alice\plugin", name="w")) == []
    assert pc.plugin_hits(SimpleNamespace(source="entrypoint", path="no_such_pkg_xyz:register", name="e")) == []
    pkg = tmp_path / "site" / "vendor_plugin"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from tools.web_tools import prefers_gateway\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    hits = pc.plugin_hits(SimpleNamespace(source="entrypoint", path="vendor_plugin:register", name="v"))
    assert [h.file for h in hits] == ["__init__.py"]


def test_allow_override_requires_literal_true_and_notice_reports_it(monkeypatch):
    assert pc.allow_deprecated_imports({"plugins": {pc.ALLOW_KEY: "false"}}) is False
    assert pc.allow_deprecated_imports({"plugins": {pc.ALLOW_KEY: 1}}) is False
    assert pc.allow_deprecated_imports({"plugins": {pc.ALLOW_KEY: True}}) is True
    report = {"bad": [pc.Hit("x.py", 1, "tools.web_tools.prefers_gateway", "tools.tool_backend_helpers.prefers_gateway")]}
    after = pc.COMPAT_REMOVAL_DATE
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: True)
    assert "force-loaded" in pc.summary_lines(report, today=after)[0]
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: False)
    assert "DISABLED" in pc.summary_lines(report, today=after)[0]
