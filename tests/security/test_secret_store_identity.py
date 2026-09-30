"""A Hermes secret store is refused for what it is, not for how the path to it is spelled.

Every consumer (file-tool read and write guards, gateway chat delivery, dashboard Files) must
refuse a store reached by another path: a hardlink, a store directory behind a symlink, or, on a
case-insensitive filesystem, a case variant.
"""

import os
from pathlib import Path

import pytest

from agent.file_safety import get_read_block_error, is_write_denied
from gateway.platforms.base import validate_media_delivery_path
from hermes_cli.web_routers.files import _is_sensitive_path


def _let_through(path: Path, *, write: bool = True) -> list[str]:
    """The consumers that would hand ``path`` over (or let it be written)."""
    checks = {
        "read_file": get_read_block_error(str(path)) is None,
        "chat delivery": validate_media_delivery_path(str(path)) is not None,
        "dashboard Files": not _is_sensitive_path(path),
    }
    if write:
        checks["write_file"] = not is_write_denied(str(path))
    return [f"{path.name} via {name}" for name, allowed in checks.items() if allowed]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    return home


def test_store_reached_through_a_hardlink_or_symlinked_directory_is_refused(home, tmp_path):
    (home / ".env").write_text("FAKE_KEY=synthetic")
    outside = tmp_path / "outside"
    outside.mkdir()
    hardlink = outside / "notes.txt"
    os.link(home / ".env", hardlink)
    # A store directory moved behind a symlink: the resolved path has no store name in it.
    external = tmp_path / "external-tokens"
    external.mkdir()
    (home / "mcp-tokens").symlink_to(external, target_is_directory=True)
    (external / "server.json").write_text('{"access_token": "synthetic"}')

    leaks = _let_through(hardlink) + _let_through(external / "server.json")

    assert leaks == []
    plain = outside / "report.txt"
    plain.write_text("ordinary")
    assert _let_through(plain) == [f"{plain.name} via {name}" for name in
                                   ("read_file", "chat delivery", "dashboard Files", "write_file")]


@pytest.mark.platforms("macos", "windows")
def test_case_variant_of_a_store_is_refused_on_a_case_insensitive_filesystem(home):
    (home / "auth.json").write_text('{"token": "synthetic"}')
    (home / "mcp-tokens").mkdir()
    (home / "mcp-tokens" / "server.json").write_text('{"access_token": "synthetic"}')

    leaks = _let_through(home / "AUTH.JSON", write=False) + _let_through(home / "MCP-TOKENS" / "server.json")
    # Not created yet: the write would create the .env the loader reads at startup.
    if not is_write_denied(str(home / ".ENV")):
        leaks.append(".ENV via write_file")

    assert leaks == []
