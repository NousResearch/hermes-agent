"""#124058: catalog ``known_issues`` is an informational field.

The catalog (``plugin-catalog/hindsight.yaml``) documents in prose that
``local_embedded`` mode is unsupported on PM-managed Hermes with the current
pin (it calls the retired lazy-install path for ``hindsight-all`` and loops
update takeovers). The field is machine-readable but informational: it must
parse and round-trip, and the install summaries must surface the text — it
must never block an install (the guard belongs at the mode-selection seam;
see teknium1's review on #124037 and #122341 / #123771).
"""

from pathlib import Path

from hermes_cli.plugin_catalog import (
    PluginCatalogEntry,
    entry_capability_summary,
    entry_from_mapping,
    load_catalog,
)


def _entry(**overrides):
    base = dict(
        name="hindsight",
        repo="https://github.com/vectorize-io/hindsight",
        sha="176f8c2de1369f569c489b831d143b78128b5535",
        tier="community",
        category="memory",
    )
    base.update(overrides)
    return entry_from_mapping(base, "test-entry")


def test_known_issues_parse_round_trip(tmp_path):
    """known_issues parses from dict and YAML into the entry and to_dict rewrites it identically."""
    entry = _entry(known_issues=["First issue.", "Second issue."])
    assert isinstance(entry, PluginCatalogEntry)
    assert entry.known_issues == ["First issue.", "Second issue."]
    assert entry.to_dict()["known_issues"] == ["First issue.", "Second issue."]

    # YAML → entry → to_dict: structural round-trip on a scratch catalog, so
    # the test never pins the prose of the live hindsight entry.
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "hindsight.yaml").write_text(
        "name: hindsight\n"
        "repo: https://github.com/vectorize-io/hindsight\n"
        "sha: 176f8c2de1369f569c489b831d143b78128b5535\n"
        "description: test entry\n"
        "maintainer: vectorize-io\n"
        "known_issues:\n"
        "  - 'Trap one.'\n"
        "  - 'Trap two.'\n"
    )
    loaded = load_catalog(catalog_dir)
    assert len(loaded) == 1
    assert loaded[0].known_issues == ["Trap one.", "Trap two."]
    assert loaded[0].to_dict()["known_issues"] == ["Trap one.", "Trap two."]

    # Missing key defaults to an empty list; to_dict always emits the key.
    assert _entry().known_issues == []
    assert _entry().to_dict()["known_issues"] == []


def test_install_summary_shows_known_issues(monkeypatch):
    """The dashboard install result and capability summary surface the text; install still proceeds."""
    from hermes_cli.plugins_cmd_install import dashboard_install_plugin

    entry = _entry(known_issues=["Local embedded mode is not supported on PM-managed Hermes."])
    monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.get_live_catalog_entry", lambda _n: entry)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._resolve_git_url",
        lambda _i: ("https://github.com/vectorize-io/hindsight.git", None))
    monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.raise_if_removed", lambda *a: None)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd_catalog.install_catalog_entry",
        lambda *a, **k: (Path("/tmp/hindsight-target"), {"manifest": 1}, "hindsight"))
    monkeypatch.setattr("hermes_cli.plugins_cmd._python_dependency_summary", lambda _t, w: None)
    monkeypatch.setattr("hermes_cli.plugins_cmd._set_plugin_enabled", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.plugins_cmd._missing_env_specs", lambda _m: [])
    monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda _n: {})

    result = dashboard_install_plugin("", force=False, enable=False, catalog_name="hindsight")
    assert result["ok"] is True  # informational — the install is not refused
    assert result["known_issues"] == ["Local embedded mode is not supported on PM-managed Hermes."]
    assert any(w.startswith("Known issue:") for w in result["warnings"])

    # The catalog capability summary (shown at install/enable prompts and in the UI) carries the text.
    summary = entry_capability_summary(entry)
    assert "Known issues:" in summary
    assert "Local embedded mode is not supported on PM-managed Hermes." in summary
