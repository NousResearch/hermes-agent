"""Regression for #132431 — stale TS-sibling .js must not survive into builds."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hermes_cli.source_build import clean_stale_typescript_sibling_js


def _touch(path: Path, text: str = "// stale\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_removes_untracked_js_beside_ts(tmp_path: Path) -> None:
    shared = tmp_path / "apps" / "shared" / "src"
    _touch(shared / "index.ts", "export const x = 1\n")
    stale = shared / "index.js"
    _touch(stale, "export const old = true\n")

    with patch("hermes_cli.source_build._git_tracked_paths", return_value=set()):
        removed = clean_stale_typescript_sibling_js(tmp_path)

    assert removed == [stale]
    assert not stale.exists()
    assert (shared / "index.ts").exists()


def test_keeps_tracked_js_beside_ts(tmp_path: Path) -> None:
    desktop = tmp_path / "apps" / "desktop" / "src"
    _touch(desktop / "plugin.ts", "export {}\n")
    kept = desktop / "plugin.js"
    _touch(kept, "module.exports = {}\n")

    with patch(
        "hermes_cli.source_build._git_tracked_paths",
        return_value={"apps/desktop/src/plugin.js"},
    ):
        removed = clean_stale_typescript_sibling_js(tmp_path)

    assert removed == []
    assert kept.exists()


def test_ignores_js_without_ts_sibling(tmp_path: Path) -> None:
    shared = tmp_path / "apps" / "shared" / "src"
    lone = shared / "only-js.js"
    _touch(lone)

    with patch("hermes_cli.source_build._git_tracked_paths", return_value=set()):
        removed = clean_stale_typescript_sibling_js(tmp_path)

    assert removed == []
    assert lone.exists()


def test_removes_tsx_sibling_js(tmp_path: Path) -> None:
    desktop = tmp_path / "apps" / "desktop" / "src" / "components"
    _touch(desktop / "Button.tsx", "export const Button = () => null\n")
    stale = desktop / "Button.js"
    _touch(stale)

    with patch("hermes_cli.source_build._git_tracked_paths", return_value=set()):
        removed = clean_stale_typescript_sibling_js(tmp_path)

    assert removed == [stale]
    assert not stale.exists()
