"""``hermes plugins doctor`` warns when Desktop runs a stale copy of a unified package's desktop half."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _plugin(root: Path) -> Path:
    plugin = root / "plugins" / "uni"
    (plugin / "desktop").mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: uni\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugin / "desktop" / "plugin.js").write_text("export default { id: 'uni' } // v1\n", encoding="utf-8")
    return plugin


def _copy(home: Path, plugin: Path) -> Path:
    copy = home / "desktop-plugins" / "uni"
    copy.mkdir(parents=True)
    (copy / "plugin.js").write_bytes((plugin / "desktop" / "plugin.js").read_bytes())
    (copy / ".hermes-package.json").write_text(
        json.dumps({"package": "uni", "source": str(plugin / "desktop"), "sourceMtimeMs": 1}), encoding="utf-8")
    return copy


def _stale_warnings(report) -> list[str]:
    return [f.message for f in report.findings if "stale copy" in f.message]


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_doctor_warns_when_the_desktop_copy_is_stale(home: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = _plugin(home)
    copy = _copy(home, plugin)
    assert _stale_warnings(doctor_plugin(plugin)) == []

    (plugin / "desktop" / "plugin.js").write_text("export default { id: 'uni' } // v2\n", encoding="utf-8")
    report = doctor_plugin(plugin)

    [warning] = _stale_warnings(report)
    assert str(copy) in warning
    assert "Rescan" in warning
    assert report.ok  # a warning, never a CI failure


def test_doctor_is_silent_without_a_desktop_copy(home: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    assert _stale_warnings(doctor_plugin(_plugin(home))) == []
