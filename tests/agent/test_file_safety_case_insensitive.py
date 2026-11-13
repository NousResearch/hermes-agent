"""Credential denies hold for case variants on case-insensitive filesystems (#51474).

macOS and Windows open ``mcp-tokens/`` for ``MCP-TOKENS/``, while realpath / ``Path.resolve``
keep the typed case, so exact-string deny comparisons let the variant through. Linux keeps the
two apart, so the denies stay exact there. Dummy files only, in a throwaway home.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs
from gateway.platforms import base as gateway_base


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Profile HERMES_HOME under a distinct global root, plus a throwaway user home."""
    root = tmp_path.resolve() / "hermes_root"
    profile = root / "profiles" / "coder"
    user_home = tmp_path.resolve() / "user"
    for d in (profile / "mcp-tokens", root / "auth", user_home / ".ssh"):
        d.mkdir(parents=True)
    (profile / "mcp-tokens" / "github.json").write_text("dummy", encoding="utf-8")
    (root / "auth.json").write_text("dummy", encoding="utf-8")
    (user_home / ".ssh" / "id_ed25519").write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    return root, profile, user_home


def _variants(root: Path, profile: Path, user_home: Path) -> dict[str, str]:
    return {
        "read: credential file": str(root / "AUTH.JSON"),
        "read: credential dir": str(profile / "MCP-Tokens" / "github.json"),
        "read: project .env": str(root.parent / "project" / ".ENV"),
        "write: protected subpath": str(profile / "MCP-TOKENS" / "new.json"),
        "write: home credential dir": str(user_home / ".SSH" / "authorized_keys"),
    }


@pytest.mark.platforms("macos", "windows")
def test_case_variants_of_credential_paths_are_denied(hermes_layout):
    paths = _variants(*hermes_layout)
    for label, path in paths.items():
        if label.startswith("read"):
            assert fs.get_read_block_error(path), f"{label} readable: {path}"
        else:
            assert fs.is_write_denied(path), f"{label} writable: {path}"


@pytest.mark.platforms("macos", "windows")
def test_case_variant_of_home_credential_is_not_media_deliverable(hermes_layout):
    *_, user_home = hermes_layout
    variant = (user_home / ".SSH" / "id_ed25519").resolve()
    assert gateway_base._path_under_denied_prefix(variant) is True
    assert gateway_base._path_under_denied_prefix((user_home / "report.pdf").resolve()) is False


@pytest.mark.platforms("linux")
def test_case_variants_are_distinct_files_on_linux(hermes_layout):
    root, profile, user_home = hermes_layout
    assert fs.get_read_block_error(str(root / "AUTH.JSON")) is None
    assert fs.is_write_denied(str(profile / "MCP-TOKENS" / "new.json")) is False
    assert fs.is_write_denied(str(user_home / ".SSH" / "authorized_keys")) is False
    assert fs.is_write_denied(str(user_home / ".ssh" / "authorized_keys")) is True
