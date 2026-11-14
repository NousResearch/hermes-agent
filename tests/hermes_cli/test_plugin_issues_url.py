"""Offline issue-tracker discovery and CLI support output (refs #84941)."""

import json
import logging
from types import SimpleNamespace

import pytest

from hermes_cli import dump, plugin_catalog, plugins_cmd, plugins_manifest, plugins_provenance


UPSTREAM_ISSUES = "https://github.com/NousResearch/hermes-agent/issues"


@pytest.fixture(autouse=True)
def offline_resolution(monkeypatch):
    def refuse_live_catalog(*args, **kwargs):
        raise AssertionError("Issue-tracker resolution must stay offline")

    monkeypatch.setattr(plugin_catalog, "load_catalog_live", refuse_live_catalog)
    monkeypatch.setattr(plugins_manifest, "_catalog_repo", lambda name: "")
    monkeypatch.setattr(plugins_provenance, "_git_origin_url", lambda path: None)


@pytest.mark.parametrize("url", ["https://example.com/bugs", "http://example.com/bugs"])
def test_explicit_url_wins(url, tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_manifest, "_catalog_repo", lambda name: "https://github.com/catalog/repo")
    manifest = {"issues_url": url, "homepage": "https://github.com/home/repo"}
    assert plugins_manifest.resolve_plugin_issues_url("sample", manifest, tmp_path) == url


@pytest.mark.parametrize("value", ["javascript:alert(1)", "file:///bugs", "git@github.com:owner/repo", 42, None])
def test_invalid_url_is_ignored(value, tmp_path):
    manifest = {"issues_url": value, "homepage": "https://github.com/home/repo"}
    assert plugins_manifest.resolve_plugin_issues_url("sample", manifest, tmp_path) == "https://github.com/home/repo/issues"


@pytest.mark.parametrize("value", ["https://example.com/bugs", "http://example.com/bugs", "mailto:bugs@example.com", 42])
def test_manifest_parsing_recognizes_and_normalizes_url(value, tmp_path, caplog):
    import hermes_yaml as yaml

    manifest_file = tmp_path / "plugin.yaml"
    manifest_file.write_text(yaml.safe_dump({"name": "sample", "manifest_version": 2, "issues_url": value}))
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manifest = plugins_manifest.parse_manifest_file(manifest_file, tmp_path, "user", "")
    expected = value if isinstance(value, str) and value.startswith(("https://", "http://")) else ""
    assert manifest is not None
    assert manifest.issues_url == expected
    assert "unknown manifest field" not in caplog.text
    assert plugins_manifest.resolve_plugin_issues_url("sample", manifest, tmp_path) == expected


def test_catalog_fallback_precedes_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_manifest, "_catalog_repo", lambda name: "https://github.com/catalog/repo.git")
    monkeypatch.setattr(plugins_provenance, "_git_origin_url", lambda path: "git@github.com:origin/repo.git")
    assert plugins_manifest.resolve_plugin_issues_url("sample", {}, tmp_path) == "https://github.com/catalog/repo/issues"


@pytest.mark.parametrize("origin", [
    "git@github.com:owner/repo", "git@github.com:owner/repo.git",
    "https://github.com/owner/repo", "https://github.com/owner/repo.git",
    "ssh://git@github.com/owner/repo", "ssh://git@github.com/owner/repo.git",
])
def test_origin_fallback_precedes_homepage(origin, tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_manifest, "_catalog_repo", lambda name: "https://gitlab.com/catalog/repo")
    monkeypatch.setattr(plugins_provenance, "_git_origin_url", lambda path: origin)
    manifest = {"homepage": "https://github.com/home/repo"}
    assert plugins_manifest.resolve_plugin_issues_url("sample", manifest, tmp_path) == "https://github.com/owner/repo/issues"


@pytest.mark.parametrize(("homepage", "expected"), [
    ("https://github.com/owner/repo", "https://github.com/owner/repo/issues"),
    ("https://github.com/owner/repo.git/", "https://github.com/owner/repo/issues"),
    ("https://example.com/plugin", ""),
    ("https://github.com/owner/repo/tree/main", ""),
    ("https://github.com/owner", ""),
    ("", ""),
])
def test_homepage_requires_github_repo_root(homepage, expected):
    assert plugins_manifest.resolve_plugin_issues_url("sample", {"homepage": homepage}, None) == expected


def test_bundled_plugin_reports_upstream():
    manifest = plugins_manifest.PluginManifest(name="sample", source="bundled")
    assert plugins_manifest.resolve_plugin_issues_url("sample", manifest, None) == UPSTREAM_ISSUES


@pytest.fixture
def installed_plugins(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "plugins"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))
    (home / "config.yaml").write_text("plugins:\n  enabled: [sample, unknown]\n  disabled: [disabled]\n")
    for name, tracker in (("sample", "https://example.com/bugs"), ("unknown", ""), ("disabled", ""), ("inactive", "")):
        plugin = root / name
        plugin.mkdir()
        (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.2.3\nissues_url: '{tracker}'\n")
    bundled = tmp_path / "bundled" / "builtin"
    bundled.mkdir(parents=True)
    (bundled / "plugin.yaml").write_text("name: builtin\nversion: 2.0\nkind: backend\n")
    return root


@pytest.mark.parametrize(("name", "expected"), [("sample", "https://example.com/bugs"), ("unknown", ""), ("builtin", UPSTREAM_ISSUES)])
def test_show_issue_tracker(name, expected, installed_plugins, capsys):
    plugins_cmd.cmd_show(name)
    output = capsys.readouterr().out
    if expected:
        assert f"Report issues: {expected}" in output
        assert output.index("Key:") < output.index("Report issues:")
    else:
        assert "Report issues:" not in output


def test_json_list_has_issue_tracker(installed_plugins, monkeypatch, capsys):
    from hermes_cli import plugins_cmd_catalog

    monkeypatch.setattr(plugins_cmd_catalog, "resolved_removed_entries", list)
    plugins_cmd.cmd_list(SimpleNamespace(json=True))
    entries = {entry["name"]: entry for entry in json.loads(capsys.readouterr().out)}
    assert entries["sample"]["issues_url"] == "https://example.com/bugs"
    assert entries["unknown"]["issues_url"] == ""
    assert entries["builtin"]["issues_url"] == UPSTREAM_ISSUES


@pytest.fixture
def isolated_dump(monkeypatch):
    monkeypatch.setattr(dump, "load_hermes_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(dump, "display_hermes_home", lambda: "(test home)")
    monkeypatch.setattr(dump, "_version_line", lambda root: "(test version)")
    monkeypatch.setattr(dump, "_gateway_status", lambda: "stopped")
    monkeypatch.setattr(dump, "_api_key_lines", lambda show_keys: [])


def test_dump_lists_only_enabled_plugins_without_paths(installed_plugins, isolated_dump, capsys):
    dump.run_dump(SimpleNamespace())
    output = capsys.readouterr().out
    assert "plugins:\n" in output
    assert "  sample 1.2.3 user https://example.com/bugs" in output
    assert "  unknown 1.2.3 user -" in output
    assert f"  builtin 2.0 bundled {UPSTREAM_ISSUES}" in output
    assert "  disabled " not in output
    assert "  inactive " not in output
    assert "/Users/" not in output
    assert str(installed_plugins.parent.parent) not in output


def test_dump_survives_resolver_exception(installed_plugins, isolated_dump, monkeypatch, capsys):
    def broken_resolver(*args):
        raise RuntimeError("resolver failed")

    monkeypatch.setattr(plugins_manifest, "resolve_plugin_issues_url", broken_resolver)
    dump.run_dump(SimpleNamespace())
    output = capsys.readouterr().out
    assert "plugins:\n  (unavailable)" in output
    assert "--- end dump ---" in output


def test_catalog_index_parsed_once_per_mtime(tmp_path, monkeypatch):
    monkeypatch.undo()  # use the real _catalog_repo; only the catalog source is faked below
    calls = []
    entries = [SimpleNamespace(name="a", repo="https://github.com/o/a"), SimpleNamespace(name="b", repo="https://github.com/o/b")]
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: tmp_path)
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda root=None: calls.append(root) or entries)
    plugins_manifest._CATALOG_REPOS.clear()
    assert [plugins_manifest._catalog_repo(n) for n in ("a", "b", "a", "missing")] == [
        "https://github.com/o/a", "https://github.com/o/b", "https://github.com/o/a", ""]
    assert calls == [tmp_path]
