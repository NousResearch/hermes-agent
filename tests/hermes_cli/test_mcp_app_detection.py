"""Backend-native, read-only catalog app discovery; no app or MCP is launched."""

import os
from pathlib import Path
import sys

import pytest


def test_native_discovery_returns_only_curated_boundary_matches(tmp_path, monkeypatch):
    from hermes_cli import mcp_app_detection as detection

    root = tmp_path / "applications"
    root.mkdir()
    if sys.platform == "darwin":
        (root / "Vendor" / "FIXTURE Paint 5.1.app").mkdir(parents=True)
        (root / "SuperFixtureCode.app").mkdir()
        (root / "Personal Secret.app").mkdir()
    elif sys.platform == "win32":
        (root / "Vendor" / "FIXTURE Paint 5.1").mkdir(parents=True)
        (root / "SuperFixtureCode").mkdir()
        (root / "Personal Secret").mkdir()
    else:
        (root / "org.example.editor.desktop").write_text(
            "[Desktop Entry]\nName=FIXTURE Paint 5.1\nExec=do-not-run\n", encoding="utf-8")
        (root / "SuperFixtureCode.desktop").write_text("[Desktop Entry]\nName=Personal Secret\n")
    monkeypatch.setattr(detection, "_application_roots", lambda: [root])
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    (tmp_path / "bin").mkdir()
    criteria = {"paint": ["Fixture Paint", "Missing App"], "code": ["FixtureCode"], "legacy": []}
    result = detection.discover_catalog_apps(criteria)
    assert result == {"matches": {"paint": ["Fixture Paint"], "code": [], "legacy": []},
                      "discovery": {"scope": "backend", "status": "ok", "platform": sys.platform}}
    # A PATH signal is exact, not a substring or shell invocation. Executable content is never run.
    candidate = tmp_path / "bin" / ("fixturecode.exe" if os.name == "nt" else "fixturecode")
    candidate.write_text("this is not an executable program", encoding="utf-8")
    candidate.chmod(0o755)
    assert detection.discover_catalog_apps(criteria)["matches"]["code"] == ["FixtureCode"]


def test_discovery_failure_or_budget_is_unknown_not_absence(tmp_path, monkeypatch):
    from hermes_cli import mcp_app_detection as detection

    monkeypatch.setattr(detection, "_application_roots", lambda: [tmp_path])
    monkeypatch.setenv("PATH", "")
    criteria = {f"entry{i}": [f"App{i}"] for i in range(300)}
    limited = detection.discover_catalog_apps(criteria)
    assert limited["discovery"]["status"] == "unavailable"
    assert all(matches == [] for matches in limited["matches"].values())

    criteria = {"demo": ["Fixture Paint"]}
    assert detection.discover_catalog_apps(criteria)["discovery"]["status"] == "ok"
    (tmp_path / "ordinary-file").touch()
    monkeypatch.setattr(detection, "_application_roots", lambda: [tmp_path / "ordinary-file"])
    assert detection.discover_catalog_apps(criteria)["discovery"]["status"] == "unavailable"
    monkeypatch.setattr(detection, "_application_roots", lambda: [tmp_path / "missing"])
    assert detection.discover_catalog_apps(criteria)["discovery"]["status"] == "unavailable"

    def denied(_path):
        raise PermissionError("private filesystem path must not escape")

    monkeypatch.setattr(detection.os, "scandir", denied)
    assert detection.discover_catalog_apps(criteria) == {
        "matches": {"demo": []},
        "discovery": {"scope": "backend", "status": "unavailable", "platform": sys.platform},
    }
    with pytest.raises(ValueError, match="suggest.applications"):
        detection.discover_catalog_apps({"demo": ["/bin/echo"]})
