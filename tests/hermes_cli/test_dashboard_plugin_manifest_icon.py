"""A dashboard plugin manifest's ``icon`` is documented as a Lucide icon name
(a string, ``Puzzle`` when absent or unknown).  The discovery layer already
validates ``tab``, ``slots`` and ``api``; ``icon`` used to pass through
untouched, so a manifest carrying an object (``{"type": "image", ...}``) was
served as-is by ``GET /api/dashboard/plugins``.  The web UI looks the value up
in ``ICON_MAP[name] ?? Puzzle`` and shrugs, but a typed client that decodes the
list (the iOS app) fails the whole list on the one odd element.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.web_server_dashboard as dashboard


def _write_manifest(root: Path, name: str, manifest: dict) -> None:
    dashboard_dir = root / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "manifest.json").write_text(json.dumps({"name": name, **manifest}), encoding="utf-8")


def _icon_of(root: Path, name: str):
    plugins = dashboard._discover_dashboard_plugins(search_dirs=[(root, "user")])
    return next(p["icon"] for p in plugins if p["name"] == name)


def test_string_icon_is_kept(tmp_path):
    _write_manifest(tmp_path, "kept", {"icon": "Sparkles"})
    assert _icon_of(tmp_path, "kept") == "Sparkles"


def test_missing_icon_defaults_to_puzzle(tmp_path):
    _write_manifest(tmp_path, "missing", {})
    assert _icon_of(tmp_path, "missing") == "Puzzle"


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "image", "src": "dist/icon.png", "alt": "logo"},
        ["Sparkles"],
        42,
        True,
        None,
        "",
        "   ",
    ],
    ids=["object", "list", "int", "bool", "null", "empty", "blank"],
)
def test_non_string_or_blank_icon_falls_back_to_puzzle(tmp_path, bad):
    _write_manifest(tmp_path, "odd", {"icon": bad})
    assert _icon_of(tmp_path, "odd") == "Puzzle"
