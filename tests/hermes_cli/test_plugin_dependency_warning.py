"""Discovery's missing-dependency warning covers the install surface, not one field.

``install_requirements`` — the union of the manifest's ``pip_dependencies``/
``python_dependencies`` aliases and a real pyproject's dependencies — is exactly
what ``hermes pm`` stages into the venv, so a requirement missing from it is the
same broken import the loader trips over (#136519). Markers survive that surface
on purpose (uv evaluates them per target), so the probe must skip requirements
whose marker does not hold on the running interpreter.
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli.plugins_loader import PluginLoaderMixin
from hermes_cli.plugins_manifest import PluginManifest


def _manifest_with(tmp_path: Path, manifest_yaml: str) -> PluginManifest:
    plugin_dir = tmp_path / "probe-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(manifest_yaml, encoding="utf-8")
    return PluginManifest(name="probe-plugin", version="1.0", path=str(plugin_dir))


def _probe(manifest: PluginManifest, caplog) -> list[str]:
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins_loader"):
        PluginLoaderMixin()._warn_python_dependencies(manifest)
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


def test_pip_dependencies_alias_warns_when_missing(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path, "name: probe-plugin\npip_dependencies: [slixmpp==1.15.0]\n"
    )
    warnings = _probe(manifest, caplog)
    assert any("slixmpp==1.15.0" in w and "pm repair" in w for w in warnings), warnings


def test_python_dependencies_field_still_warns(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path,
        "name: probe-plugin\npython_dependencies: [never-install-without-consent==1]\n",
    )
    warnings = _probe(manifest, caplog)
    assert any("never-install-without-consent==1" in w for w in warnings), warnings


def test_both_fields_warn_once_without_duplicates(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path,
        "name: probe-plugin\n"
        "python_dependencies: [left-pad-py]\n"
        "pip_dependencies: [left-pad-py, right-pad-py]\n",
    )
    warnings = _probe(manifest, caplog)
    assert len(warnings) == 1, warnings
    assert warnings[0].count("left-pad-py") == 1, warnings
    assert "right-pad-py" in warnings[0], warnings


def test_external_runtime_plugin_is_not_warned(tmp_path, caplog, monkeypatch):
    """An external-runtime plugin owns its interpreter; PM stages nothing, so missing
    requirements there are not Hermes' to report — whichever field declares them."""
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path,
        "name: probe-plugin\npython_runtime: external\n"
        "python_dependencies: [slixmpp]\npip_dependencies: [slixmpp]\n",
    )
    assert _probe(manifest, caplog) == []


def test_marker_not_applicable_here_does_not_warn(tmp_path, caplog, monkeypatch):
    """Markers survive ``install_requirements`` on purpose (uv evaluates them per target
    Python/platform); a requirement whose marker never holds on the running interpreter
    is not missing here, and ``pm repair`` could not clear a false warning for it."""
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path,
        'name: probe-plugin\npython_dependencies: [\'pywin32; python_version < "3"\']\n',
    )
    assert _probe(manifest, caplog) == []


def test_marker_applicable_here_still_warns(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(
        tmp_path,
        'name: probe-plugin\npip_dependencies: [\'slixmpp; python_version >= "3"\']\n',
    )
    warnings = _probe(manifest, caplog)
    assert any("slixmpp" in w and "pm repair" in w for w in warnings), warnings


def test_satisfied_dependencies_stay_quiet(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: True)
    manifest = _manifest_with(
        tmp_path, "name: probe-plugin\npip_dependencies: [slixmpp]\n"
    )
    assert _probe(manifest, caplog) == []


def test_no_declaration_at_all_stays_quiet(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_loader._dist_installed", lambda req: False)
    manifest = _manifest_with(tmp_path, "name: probe-plugin\n")
    assert _probe(manifest, caplog) == []
